<div align="center">

# 🎨 Creative Browser Pro

**A 4-quadrant desktop browser with a built-in AI design studio — everything runs on your machine.**

[![Release](https://img.shields.io/github/v/release/bobguffie/creative-browser-2?label=%E2%AC%87%EF%B8%8F%20Download&color=success)](https://github.com/bobguffie/creative-browser-2/releases/latest)
[![Build](https://github.com/bobguffie/creative-browser-2/actions/workflows/build.yml/badge.svg)](https://github.com/bobguffie/creative-browser-2/actions/workflows/build.yml)

**[⬇️ DOWNLOAD THE LATEST APPIMAGE](https://github.com/bobguffie/creative-browser-2/releases/latest)**

*Linux · ~430 MB · No installation — just download, `chmod +x`, and run*

</div>

---

## What is it?

A desktop app that puts everything a creator needs on one screen:

| Quadrant | Purpose |
|----------|---------|
| 🔍 **Search** | Live Google |
| 🖌️ **AI Assets** | Bing Image Creator |
| 🖼️ **Reference** | Pinterest (or any site) |
| 🎨 **Designer** | Full Fabric.js design studio — built in |

Every quadrant is a real browser view: type a URL in its bar and press **Enter**.

## ✨ Local AI (on-device, private)

Your images **never leave your computer**. Models download once (~100-200MB),
then run entirely offline on your GPU or CPU:

| Tool | What it does |
|------|-------------|
| 🪄 **Remove Background** | BiRefNet segmentation (WebGPU) or edge flood-fill matte (CPU) → transparent PNG |
| 🧩 **Magic Object Layers** | SAM 2 Automatic Mask Generator slices a photo into movable object layers — Canva-style |
| ⚙️ **Auto / GPU / CPU toggle** | WebGPU with fp16 when available; graceful WASM fallback |

Plus the full studio: shapes, text, brushes (pencil/circle/spray/pattern),
clone stamp, spot healing, color key, slice/weld, frames, layers panel,
filters, undo/redo, PNG export.

## 🚀 Quick start

**Option A — download the AppImage (recommended):**

Grab it from the [Releases page](https://github.com/bobguffie/creative-browser-2/releases/latest):

```bash
chmod +x CreativeBrowser-*.AppImage
./CreativeBrowser-*.AppImage
```

**Option B — run from source:**

```bash
git clone https://github.com/bobguffie/creative-browser-2.git
cd creative-browser-2
npm install
npm start
```

> 💡 On Linux machines with GPU driver trouble, run:
> `npm start -- --no-sandbox --disable-gpu --in-process-gpu`

## 🔑 Optional: cloud AI background removal

The ✨ **Remove BG** button (cloud) needs a free Gemini API key:

```bash
cp .env.example .env   # then paste your key from aistudio.google.com/apikey
```

The 🪄 **Local AI** buttons need no key at all.

## 🛠️ Building from source

```bash
npm run dist        # → release/*.AppImage
```

CI does this automatically on every push — see the
[Actions tab](https://github.com/bobguffie/creative-browser-2/actions) or the
[latest release](https://github.com/bobguffie/creative-browser-2/releases/latest).

## 📦 Tech

Electron 30 · Fabric.js 5.3 · Transformers.js 4 (ONNX Runtime Web) ·
SAM 2 / SlimSAM / BiRefNet · models cached in browser CacheStorage
