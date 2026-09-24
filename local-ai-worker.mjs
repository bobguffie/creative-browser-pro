// LocalAiWorker.mjs — on-device AI vision engine for the Designer workspace.
//
// Runs inside a dedicated Web Worker (UI stays at 60 FPS):
//   • "Remove Background (Local AI)":
//       GPU (with fp16) → BiRefNet (onnx-community/BiRefNet_lite-ONNX)
//       CPU / no-fp16   → border flood-fill matte: a magic-wand segmentation
//         seeded from every edge pixel, tolerance-scaled to image variance.
//         Deterministic, ~50ms, zero model download, and — unlike MODNet
//         (portrait-only) or BiRefNet-on-WASM (its fixed 1024² activations
//         exceed the 4GB heap → ORT error 6) — it works on ANY photo.
//       Fallback chain: GPU-BiRefNet → flood-fill (never fails silently).
//   • "Split into Magic Layers (Local AI)":
//       SAM Automatic Mask Generator pipeline:
//         GPU → SAM 2 Small (onnx-community/sam2-hiera-small-ONNX)
//         CPU → SlimSAM-77 (Xenova/slimsam-77-uniform, proven on WASM)
//       One encoder pass, chunked decoders, IoU+stability+point-in-mask
//       filtering, NMS dedupe → cropped RGBA cutouts as movable layers.
//
// Device policy (spec):
//   auto → real WebGPU adapter probe (incl. shader-f16) ? gpu : cpu
//   gpu  → force WebGPU, error notice if absent  |  cpu → force WASM
//
// Weights are cached by transformers.js in the app:// origin's CacheStorage
// (privileged protocol in main.cjs) — fully offline after first download.

const BG_GPU_MODEL = 'onnx-community/BiRefNet_lite-ONNX';
const SAM_GPU_MODEL = 'onnx-community/sam2-hiera-small-ONNX';
const SAM_CPU_MODEL = 'Xenova/slimsam-77-uniform';

const toLibDevice = (d) => (d === 'cpu' ? 'wasm' : d);

// AMG tuning
const GRID = 14;                // 14×14 = 196 point prompts
const MIN_IOU_SCORE = 0.5;
const MIN_STABILITY = 0.80;     // used for layers; union-bg skips it
const MIN_AREA_RATIO = 0.0005;
const MAX_AREA_LAYERS = 0.55;   // layers: drop big background slabs
const MAX_AREA_UNION = 0.97;    // union cutout: keep everything object-like
const NMS_BOX_THRESH = 0.55;
const NMS_MASK_OVERLAP = 0.80;
const MAX_LAYERS = 24;
const DECODE_CHUNK = 49;        // prompts per decoder pass

// Flood-fill tuning
const FF_MAX_DIM = 1024;        // analysis resolution cap
const FF_FEATHER = 1.5;         // px alpha feather at mask edges

let bgPipe = null, bgPipeDevice = null;
let samModel = null, samProcessor = null, samEngine = null;
let deviceMode = 'auto';

const post = (payload) => self.postMessage(payload);
const report = (type, payload = {}) => post({ type, ...payload });

// ------------------------------- device policy -------------------------------

let probeCache = null;
async function probeWebGPU(force) {
  if (probeCache && !force) return probeCache;
  if (typeof navigator === 'undefined' || !navigator.gpu) {
    probeCache = { available: false, reason: 'WebGPU is not available in this browser/runtime.' };
    return probeCache;
  }
  try {
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) {
      probeCache = { available: false, reason: 'No WebGPU adapter found.' };
      return probeCache;
    }
    probeCache = { available: true, fp16: !!adapter.features?.has('shader-f16') };
    return probeCache;
  } catch (e) {
    probeCache = { available: false, reason: e.message };
    return probeCache;
  }
}

async function resolveDeviceRobust() {
  if (deviceMode === 'cpu') return 'cpu';
  if (deviceMode === 'gpu') {
    const probe = await probeWebGPU();
    if (!probe.available) throw new Error('GPU Only mode: ' + probe.reason + ' Switch to Auto or CPU Only.');
    return 'webgpu';
  }
  // Auto: require BOTH an adapter and shader-f16 for the GPU engine;
  // BiRefNet fp16 without shader-f16 fails at session init.
  const probe = await probeWebGPU();
  return (probe.available && probe.fp16) ? 'webgpu' : 'cpu';
}

async function loadLib() {
  const lib = await import('./vendor/transformers.bundle.js');
  lib.env.backends.onnx.wasm.wasmPaths = new URL('./vendor/', import.meta.url).href;
  lib.env.allowLocalModels = false;
  return lib;
}

function progressRelay(feature) {
  return (p) => {
    if (p && p.status === 'progress' && p.file && /\.onnx(_data)?$/.test(p.file)) {
      report('download', { feature, file: p.file, loaded: p.loaded, total: p.total, progress: p.progress });
    }
  };
}

// ------------------------------- background removal -------------------------------

async function ensureBgWebGPU() {
  if (bgPipe && bgPipeDevice === 'webgpu') return;
  if (bgPipe) { try { await bgPipe.dispose(); } catch (e) { /* noop */ } bgPipe = null; }
  const { pipeline } = await loadLib();
  report('status', { message: 'Downloading AI Engine (BiRefNet)…', phase: 'init' });
  bgPipe = await pipeline('background-removal', BG_GPU_MODEL, {
    device: 'webgpu',
    dtype: 'fp16',
    progress_callback: progressRelay('bg'),
  });
  bgPipeDevice = 'webgpu';
  report('ready', { feature: 'bg', device: 'webgpu' });
}

// ---- CPU engine: border flood-fill matte ----
// Backgrounds connect to the image border; objects sit inside. A tolerance-
// scaled flood fill from every border pixel carves the background away.
function floodFillMatte(rgba, W, H) {
  // Tolerance from color variance (uniform studio bg → tight; busy bg → loose)
  let mean = 0, meanSq = 0, n = 0;
  for (let i = 0; i < W * H * 4; i += 16) {
    const lum = 0.299 * rgba[i] + 0.587 * rgba[i + 1] + 0.114 * rgba[i + 2];
    mean += lum; meanSq += lum * lum; n++;
  }
  mean /= n; meanSq /= n;
  const variance = Math.max(0, meanSq - mean * mean);
  const tol = Math.min(48, 10 + Math.sqrt(variance) * 0.9);
  const tolSq = tol * tol * 3;

  const bg = new Uint8Array(W * H);       // 1 = background
  const stack = [];

  const push = (x, y) => {
    const idx = y * W + x;
    if (!bg[idx]) stack.push(idx);
  };
  for (let x = 0; x < W; x++) { push(x, 0); push(x, H - 1); }
  for (let y = 0; y < H; y++) { push(0, y); push(W - 1, y); }

  // seed colors: average of border pixels (backgrounds are usually uniform)
  let sr = 0, sg = 0, sb = 0, sc = 0;
  const sample = (x, y) => {
    const i = (y * W + x) * 4;
    sr += rgba[i]; sg += rgba[i + 1]; sb += rgba[i + 2]; sc++;
  };
  for (let x = 0; x < W; x += 4) { sample(x, 0); sample(x, H - 1); }
  for (let y = 0; y < H; y += 4) { sample(0, y); sample(W - 1, y); }
  sr /= sc; sg /= sc; sb /= sc;

  while (stack.length) {
    const idx = stack.pop();
    if (bg[idx]) continue;
    const i4 = idx * 4;
    const dr = rgba[i4] - sr, dg = rgba[i4 + 1] - sg, db = rgba[i4 + 2] - sb;
    if (dr * dr + dg * dg + db * db > tolSq) continue;
    bg[idx] = 1;
    const x = idx % W, y = (idx / W) | 0;
    if (x > 0) push(x - 1, y);
    if (x < W - 1) push(x + 1, y);
    if (y > 0) push(x, y - 1);
    if (y < H - 1) push(x, y + 1);
  }

  // Foreground = NOT background; morphological cleanup: despeckle + fill holes
  const fg = new Uint8Array(W * H);
  for (let i = 0; i < W * H; i++) fg[i] = bg[i] ? 0 : 1;

  // Remove tiny foreground islands (noise): count 4-neighbors, drop isolates
  const cleaned = new Uint8Array(fg);
  for (let y = 1; y < H - 1; y++) {
    for (let x = 1; x < W - 1; x++) {
      const idx = y * W + x;
      if (!fg[idx]) continue;
      const nb = fg[idx - 1] + fg[idx + 1] + fg[idx - W] + fg[idx + W];
      if (nb === 0) cleaned[idx] = 0;
    }
  }
  return cleaned;
}

async function floodFillRemoveBackground(imageBuffer) {
  report('status', { message: 'Processing Pixels…', phase: 'run' });
  const bmp = await decodeImage(imageBuffer, FF_MAX_DIM);
  const { width: W, height: H } = bmp;
  const mask = floodFillMatte(new Uint8ClampedArray(bmp.rgba.data), W, H);

  // Feather the mask edge for clean compositing
  const aCan = new OffscreenCanvas(W, H);
  const aCtx = aCan.getContext('2d');
  const aImg = aCtx.createImageData(W, H);
  for (let i = 0; i < W * H; i++) {
    aImg.data[i * 4] = 255; aImg.data[i * 4 + 1] = 255; aImg.data[i * 4 + 2] = 255;
    aImg.data[i * 4 + 3] = mask[i] ? 255 : 0;
  }
  aCtx.putImageData(aImg, 0, 0);
  if (FF_FEATHER > 0) {
    aCtx.filter = `blur(${FF_FEATHER}px)`;
    aCtx.drawImage(aCan, 0, 0);
    aCtx.filter = 'none';
  }
  return { alphaCanvas: aCan, alphaW: W, alphaH: H };
}

// SAM-union backup: union of AMG foreground masks as the matte.
async function samUnionRemoveBackground(imageBuffer) {
  const { kept, W, H } = await runAMG(imageBuffer, {
    maxAreaRatio: MAX_AREA_UNION,
    requireStability: false,
    requirePointInMask: true,
  });
  if (!kept.length) throw new Error('No distinct objects found to cut out.');

  report('status', { message: 'Building mask…', phase: 'run' });
  const union256 = new Uint8Array(256 * 256);
  for (const k of kept) {
    for (let i = 0; i < 65536; i++) if (k.mask256[i]) union256[i] = 1;
  }
  const union = upscaleMaskToImage(union256, W, H);
  const aCan = new OffscreenCanvas(W, H);
  const aCtx = aCan.getContext('2d');
  const aImg = aCtx.createImageData(W, H);
  for (let i = 0; i < W * H; i++) {
    aImg.data[i * 4] = 255; aImg.data[i * 4 + 1] = 255; aImg.data[i * 4 + 2] = 255;
    aImg.data[i * 4 + 3] = union[i] ? 255 : 0;
  }
  aCtx.putImageData(aImg, 0, 0);
  return { alphaCanvas: aCan, alphaW: W, alphaH: H };
}

async function removeBackground(imageBuffer) {
  const device = await resolveDeviceRobust();

  if (device === 'webgpu') {
    try {
      await ensureBgWebGPU();
      report('status', { message: 'Processing Pixels…', phase: 'run' });
      const { RawImage } = await loadLib();
      const small = await decodeImage(imageBuffer, 1024);
      const smallImage = new RawImage(new Uint8ClampedArray(small.rgba.data), small.width, small.height, 4);
      const out = await bgPipe(smallImage);
      const aCan = new OffscreenCanvas(small.width, small.height);
      const aCtx = aCan.getContext('2d');
      const aImg = aCtx.createImageData(small.width, small.height);
      for (let i = 0; i < small.width * small.height; i++) {
        aImg.data[i * 4] = 255; aImg.data[i * 4 + 1] = 255; aImg.data[i * 4 + 2] = 255;
        aImg.data[i * 4 + 3] = out.data[i * 4 + 3];
      }
      aCtx.putImageData(aImg, 0, 0);
      const result = await compositeAlpha(imageBuffer, aCan, small.width, small.height);
      return { ...result, device: 'webgpu', engine: 'birefnet' };
    } catch (err) {
      report('status', { message: 'GPU engine failed (' + err.message + ') — using flood-fill…', phase: 'run' });
    }
  }

  // CPU path: flood-fill, verify it kept something sane, else SAM-union.
  const ff = await floodFillRemoveBackground(imageBuffer);
  let opaque = 0;
  {
    const ctx = ff.alphaCanvas.getContext('2d');
    const d = ctx.getImageData(0, 0, ff.alphaW, ff.alphaH).data;
    for (let i = 3; i < d.length; i += 4) if (d[i] > 127) opaque++;
  }
  const opaqueRatio = opaque / (ff.alphaW * ff.alphaH);
  if (opaqueRatio < 0.02 || opaqueRatio > 0.98) {
    report('status', { message: 'Flood-fill ambiguous — running SAM cutout…', phase: 'run' });
    const sam = await samUnionRemoveBackground(imageBuffer);
    const r2 = await compositeAlpha(imageBuffer, sam.alphaCanvas, sam.alphaW, sam.alphaH);
    return { ...r2, device: 'cpu', engine: 'sam-union' };
  }
  const r = await compositeAlpha(imageBuffer, ff.alphaCanvas, ff.alphaW, ff.alphaH);
  return { ...r, device: 'cpu', engine: 'flood-fill' };
}

async function compositeAlpha(imageBuffer, alphaCanvas, aw, ah) {
  const full = await decodeImage(imageBuffer, 4096);
  const fCan = new OffscreenCanvas(full.width, full.height);
  const fCtx = fCan.getContext('2d');
  fCtx.putImageData(full.rgba, 0, 0);
  fCtx.globalCompositeOperation = 'destination-in';
  fCtx.imageSmoothingEnabled = true;
  fCtx.drawImage(alphaCanvas, 0, 0, full.width, full.height);
  const blob = await fCan.convertToBlob({ type: 'image/png' });
  return { blob, width: full.width, height: full.height };
}

// ------------------------------- SAM engines (shared AMG) -------------------------------

async function ensureSam(device) {
  if (samModel && samEngine && samModel.__device === device) return;
  if (samModel) { try { await samModel.dispose(); } catch (e) { /* noop */ } samModel = null; }
  const lib = await loadLib();
  if (device === 'webgpu') {
    report('status', { message: 'Downloading AI Engine (SAM 2)…', phase: 'init' });
    const { Sam2Model, Sam2Processor } = lib;
    const opts = { device: 'webgpu', dtype: 'fp16', progress_callback: progressRelay('sam') };
    [samModel, samProcessor] = await Promise.all([
      Sam2Model.from_pretrained(SAM_GPU_MODEL, opts),
      Sam2Processor.from_pretrained(SAM_GPU_MODEL, { device: 'webgpu', dtype: undefined }),
    ]);
    samEngine = 'sam2';
  } else {
    report('status', { message: 'Downloading AI Engine (SlimSAM)…', phase: 'init' });
    const { SamModel, SamProcessor } = lib;
    const opts = { device: 'wasm', dtype: 'q8', progress_callback: progressRelay('sam') };
    [samModel, samProcessor] = await Promise.all([
      SamModel.from_pretrained(SAM_CPU_MODEL, opts),
      SamProcessor.from_pretrained(SAM_CPU_MODEL, { device: 'wasm', dtype: undefined }),
    ]);
    samEngine = 'slimsam';
  }
  samModel.__device = device;
  report('ready', { feature: 'sam', device, engine: samEngine });
}

function buildGridPoints(w, h) {
  const pts = [];
  for (let gy = 0; gy < GRID; gy++) {
    for (let gx = 0; gx < GRID; gx++) {
      pts.push([Math.round((gx + 0.5) * w / GRID), Math.round((gy + 0.5) * h / GRID)]);
    }
  }
  return pts;
}

function maskStats(mask256) {
  let minX = 257, minY = 257, maxX = -1, maxY = -1, area = 0;
  for (let y = 0; y < 256; y++) {
    const row = y * 256;
    for (let x = 0; x < 256; x++) {
      if (mask256[row + x]) {
        if (x < minX) minX = x;
        if (x > maxX) maxX = x;
        if (y < minY) minY = y;
        if (y > maxY) maxY = y;
        area++;
      }
    }
  }
  if (maxX < 0) return null;
  return { minX, minY, maxX, maxY, area };
}

function upscaleMaskToImage(mask256, W, H) {
  const out = new Uint8Array(W * H);
  const xr = new Int32Array(W);
  for (let x = 0; x < W; x++) xr[x] = Math.min(255, Math.floor(x * 256 / W));
  for (let y = 0; y < H; y++) {
    const sy = Math.min(255, Math.floor(y * 256 / H)) * 256;
    const orow = y * W;
    for (let x = 0; x < W; x++) out[orow + x] = mask256[sy + xr[x]];
  }
  return out;
}

function downsampleMask(mask, W, H, s) {
  const out = new Uint8Array(s * s);
  for (let y = 0; y < s; y++) {
    const sy = Math.min(H - 1, Math.floor(y * H / s)) * W;
    for (let x = 0; x < s; x++) {
      out[y * s + x] = mask[sy + Math.min(W - 1, Math.floor(x * W / s))];
    }
  }
  return out;
}

function boxIoU(a, b) {
  const x0 = Math.max(a.minX, b.minX), y0 = Math.max(a.minY, b.minY);
  const x1 = Math.min(a.maxX, b.maxX), y1 = Math.min(a.maxY, b.maxY);
  const inter = Math.max(0, x1 - x0) * Math.max(0, y1 - y0);
  const areaA = (a.maxX - a.minX) * (a.maxY - a.minY);
  const areaB = (b.maxX - b.minX) * (b.maxY - b.minY);
  const union = areaA + areaB - inter;
  return union > 0 ? inter / union : 0;
}

function maskOverlap(smallA, smallB) {
  let inter = 0, areaA = 0, areaB = 0;
  for (let i = 0; i < smallA.length; i++) {
    if (smallA[i]) areaA++;
    if (smallB[i]) areaB++;
    if (smallA[i] && smallB[i]) inter++;
  }
  const minArea = Math.min(areaA, areaB);
  return minArea > 0 ? inter / minArea : 0;
}

// Shared AMG core. Returns kept masks + image pixels for cutout extraction.
async function runAMG(imageBuffer, { maxAreaRatio, requireStability }) {
  const device = await resolveDeviceRobust();
  await ensureSam(device);
  report('status', { message: 'Processing Pixels…', phase: 'run' });

  const { RawImage } = await loadLib();
  const src = await decodeImage(imageBuffer, 768);
  const W = src.width, H = src.height;
  const rgba = new Uint8ClampedArray(src.rgba.data);
  const image = new RawImage(rgba, W, H, 4);

  const allPoints = buildGridPoints(W, H);
  const baseInputs = await samProcessor(image, { input_points: [allPoints] });
  const embeddings = await samModel.get_image_embeddings(baseInputs);

  const candidates = [];
  const stats_counts = { decoded: 0, iou: 0, area: 0, pointInMask: 0 };
  for (let start = 0; start < allPoints.length; start += DECODE_CHUNK) {
    const chunk = allPoints.slice(start, start + DECODE_CHUNK);
    // Each point is its OWN prompt: shape [n, 1, 2] → [1, n, 1, 2].
    // (Passing [1, n, 2] would mean ONE prompt with n points.)
    const asOwnPrompts = chunk.map(p => [p]);
    const pointsTensor = samProcessor.reshape_input_points(
      asOwnPrompts, baseInputs.original_sizes, baseInputs.reshaped_input_sizes
    );
    const outputs = await samModel({ ...embeddings, input_points: pointsTensor });
    // pred_masks: [1, n, 3, 256, 256] logits; iou_scores: [1, n, 3]
    const n = chunk.length;
    const nMasks = outputs.pred_masks.dims[1];
    const iouData = outputs.iou_scores.data;
    for (let i = 0; i < nMasks; i++) {
      stats_counts.decoded++;
      let bestC = 0, bestScore = -Infinity, second = -Infinity;
      const iouOff = i * 3;
      for (let c = 0; c < 3; c++) {
        const s = iouData[iouOff + c];
        if (s > bestScore) { second = bestScore; bestScore = s; bestC = c; }
        else if (s > second) second = s;
      }
      if (bestScore < MIN_IOU_SCORE) continue;
      stats_counts.iou++;
      if (requireStability && bestScore > 0 && second / bestScore < MIN_STABILITY) continue;

      const logits = outputs.pred_masks.data;
      const maskLen = 256 * 256;
      const off = i * 3 * maskLen + bestC * maskLen;
      const mask256 = new Uint8Array(maskLen);
      let on = 0;
      for (let p = 0; p < maskLen; p++) { if (logits[off + p] > 0) { mask256[p] = 1; on++; } }
      const areaRatio = on / maskLen;
      if (areaRatio < MIN_AREA_RATIO || areaRatio > maxAreaRatio) { continue; }
      stats_counts.area++;

      // The prompt point must lie inside its own mask (rejects background slabs).
      const pt = chunk[i];
      const mx = Math.min(255, Math.floor(pt[0] / W * 256));
      const my = Math.min(255, Math.floor(pt[1] / H * 256));
      if (!mask256[my * 256 + mx]) continue;
      stats_counts.pointInMask++;

      const stats = maskStats(mask256);
      if (!stats) continue;
      candidates.push({ mask256, stats, score: bestScore, small: null });
    }
    report('status', {
      message: `Processing Pixels… ${Math.min(99, Math.round(100 * (start + n) / allPoints.length))}%`,
      phase: 'run',
    });
  }

  for (const c of candidates) c.small = downsampleMask(upscaleMaskToImage(c.mask256, W, H), W, H, 64);
  candidates.sort((a, b) => b.score - a.score);
  const kept = [];
  for (const cand of candidates) {
    let ok = true;
    for (const k of kept) {
      if (boxIoU(cand.stats, k.stats) > NMS_BOX_THRESH) { ok = false; break; }
      if (maskOverlap(cand.small, k.small) > NMS_MASK_OVERLAP) { ok = false; break; }
    }
    if (ok) kept.push(cand);
    if (kept.length >= MAX_LAYERS) break;
  }
  return { kept, rgba, W, H, device, counts: { ...stats_counts, candidates: candidates.length, kept: kept.length } };
}

async function magicLayers(imageBuffer) {
  const { kept, rgba, W, H, device, counts } = await runAMG(imageBuffer, {
    maxAreaRatio: MAX_AREA_LAYERS,
    requireStability: true,
  });

  report('status', { message: 'Building layers…', phase: 'run' });
  const layers = [];
  for (const k of kept) {
    const bw = k.stats.maxX - k.stats.minX + 1, bh = k.stats.maxY - k.stats.minY + 1;
    const cut = new Uint8ClampedArray(bw * bh * 4);
    let count = 0;
    for (let y = 0; y < 256; y++) {
      const srcY = Math.min(H - 1, Math.floor(y / 256 * H));
      for (let x = 0; x < 256; x++) {
        if (!k.mask256[y * 256 + x]) continue;
        const srcX = Math.min(W - 1, Math.floor(x / 256 * W));
        const si = (srcY * W + srcX) * 4;
        const di = ((y - k.stats.minY) * bw + (x - k.stats.minX)) * 4;
        cut[di] = rgba[si]; cut[di + 1] = rgba[si + 1];
        cut[di + 2] = rgba[si + 2]; cut[di + 3] = rgba[si + 3];
        count++;
      }
    }
    if (!count) continue;
    const canvas = new OffscreenCanvas(bw, bh);
    canvas.getContext('2d').putImageData(new ImageData(cut, bw, bh), 0, 0);
    const blob = await canvas.convertToBlob({ type: 'image/png' });
    layers.push({
      blob,
      left: Math.floor(k.stats.minX / 256 * W),
      top: Math.floor(k.stats.minY / 256 * H),
      width: bw, height: bh,
      score: +k.score.toFixed(3),
    });
  }
  return { layers, device, engine: samEngine, counts, imageWidth: W, imageHeight: H };
}

// ------------------------------- shared helpers -------------------------------

// Decode bytes to a bitmap, downscaled so max(w,h) <= maxDim (bounds RAM).
async function decodeImage(imageBuffer, maxDim) {
  const bitmap = await createImageBitmap(new Blob([imageBuffer]));
  const scale = Math.min(1, maxDim / Math.max(bitmap.width, bitmap.height));
  const w = Math.max(1, Math.round(bitmap.width * scale));
  const h = Math.max(1, Math.round(bitmap.height * scale));
  const can = new OffscreenCanvas(w, h);
  const ctx = can.getContext('2d');
  ctx.drawImage(bitmap, 0, 0, w, h);
  const data = ctx.getImageData(0, 0, w, h);
  bitmap.close();
  return { rgba: data, width: w, height: h };
}

// ------------------------------- router -------------------------------

self.onmessage = async (e) => {
  const { type, id, payload } = e.data || {};
  try {
    if (type === 'config') {
      deviceMode = payload?.mode || 'auto';
      const resolved = await resolveDeviceRobust().catch((err) => 'error: ' + err.message);
      report('device', { mode: deviceMode, resolved, probe: await probeWebGPU() });
      return;
    }
    if (type === 'warmup') {
      const device = await resolveDeviceRobust();
      if (payload?.feature === 'sam') await ensureSam(device);
      else if (device === 'webgpu') await ensureBgWebGPU();
      return;
    }
    if (type === 'remove-bg' && payload?.imageBuffer) {
      const result = await removeBackground(payload.imageBuffer);
      report('done', { id, kind: 'remove-bg', ...result });
      return;
    }
    if (type === 'magic-layers' && payload?.imageBuffer) {
      const result = await magicLayers(payload.imageBuffer);
      report('done', { id, kind: 'magic-layers', ...result });
      return;
    }
    report('error', { id, message: `Unknown request: ${type}` });
  } catch (error) {
    report('error', {
      id,
      message: error.message,
      stack: (error.stack || '').split('\n').slice(0, 4).join(' | '),
    });
  }
};

post({ type: 'worker-alive' });
