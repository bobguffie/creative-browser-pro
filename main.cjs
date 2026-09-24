const { app, BrowserWindow, BrowserView, ipcMain, session, nativeImage, protocol, net } = require('electron');
const path = require('path');

// GPU switches: some Linux drivers need help; zero-copy was removed (caused
// blank/artifacted views on several machines).
app.commandLine.appendSwitch('no-sandbox');
app.commandLine.appendSwitch('disable-setuid-sandbox');
app.commandLine.appendSwitch('ignore-gpu-blocklist');
app.commandLine.appendSwitch('enable-gpu-rasterization');

registerAppProtocol();

let mainWindow;
let views = [];
let uiView;
let expandedId = -1; // which view is fullscreen-expanded, -1 = none
let relayoutTimer = null;
const GAP = 5;
const NAV_HEIGHT = 35;

const CHROME_UA = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36';

// Serve only app files over the app:// protocol. The workspace loads from
// app:// so it gets a real origin — required for CacheStorage, which caches
// the ~100MB+ of local AI model weights persistently across relaunches.
const APP_FILES = new Set(['workspace.html', 'controls.html', 'preload.js', 'local-ai-worker.mjs', 'icon.png']);

function registerAppProtocol() {
  protocol.registerSchemesAsPrivileged([
    { scheme: 'app', privileges: { standard: true, secure: true, supportFetchAPI: true, stream: true } }
  ]);
}

function handleAppProtocol() {
  protocol.handle('app', (request) => {
    const url = new URL(request.url);
    const rel = decodeURIComponent(url.pathname).replace(/^\//, '');
    const isVendor = rel.startsWith('vendor/');
    if (!APP_FILES.has(rel) && !isVendor) return new Response('Not found', { status: 404 });
    if (rel.includes('..')) return new Response('Bad request', { status: 400 });
    return net.fetch('file://' + path.join(__dirname, rel)).then((resp) => {
      // Cross-origin isolation → SharedArrayBuffer → ORT multi-threaded WASM.
      const headers = new Headers(resp.headers);
      headers.set('Cross-Origin-Opener-Policy', 'same-origin');
      headers.set('Cross-Origin-Embedder-Policy', 'credentialless');
      return new Response(resp.body, { status: resp.status, headers });
    });
  });
}

// Strip framing/CSP headers ONLY for the quadrant sites (not session-wide).
const FRAMED_HOSTS = /(^|\.)google\.|(^|\.)bing\.com|(^|\.)pinterest\./;

function createView(url, isLocal = false) {
  const view = new BrowserView({
    webPreferences: {
      nodeIntegration: false,        // never give remote pages Node
      contextIsolation: true,        // required for the preload bridge
      sandbox: false,                // preload uses ipcRenderer
      webSecurity: isLocal,          // only remote views relax same-origin for canvas work
      spellcheck: true,
      preload: path.join(__dirname, 'preload.js'),
    }
  });

  view.webContents.setWindowOpenHandler(({ url: openUrl }) => {
    view.webContents.loadURL(openUrl);
    return { action: 'deny' };
  });

  view.webContents.on('context-menu', (e, props) => {
    const { Menu, MenuItem } = require('electron');
    const menu = new Menu();
    menu.append(new MenuItem({ label: 'Copy', role: 'copy' }));
    menu.append(new MenuItem({ label: 'Paste', role: 'paste' }));
    menu.append(new MenuItem({ label: 'Select All', role: 'selectAll' }));

    if (props.misspelledWord) {
      menu.append(new MenuItem({ type: 'separator' }));
      props.dictionarySuggestions.forEach(suggestion => {
        menu.append(new MenuItem({
          label: suggestion,
          click: () => view.webContents.replaceMisspelling(suggestion)
        }));
      });
    }

    if (props.hasImageContents || props.mediaType === 'image') {
      menu.append(new MenuItem({ type: 'separator' }));
      menu.append(new MenuItem({
        label: 'Send Image to Workspace',
        click: () => { if (views[3]) views[3].webContents.send('clipboard-image', props.srcURL); }
      }));
    }

    menu.append(new MenuItem({ type: 'separator' }));
    menu.append(new MenuItem({ label: 'Inspect', click: () => view.webContents.openDevTools({ mode: 'detach' }) }));
    menu.popup(mainWindow);
  });

  if (!isLocal) {
    view.webContents.setUserAgent(CHROME_UA);
    view.webContents.loadURL(url);
  } else if (url.startsWith('app://')) {
    view.webContents.loadURL(url);
  } else {
    view.webContents.loadFile(url);
  }

  return view;
}

function updateLayout() {
  if (!mainWindow) return;
  const bounds = mainWindow.getContentBounds();
  const w = bounds.width;
  const h = bounds.height;

  if (expandedId >= 0 && views[expandedId]) {
    // Fullscreen: expanded view fills the window, UI bar is hidden.
    uiView.setBounds({ x: 0, y: 0, width: 0, height: 0 });
    views[expandedId].setBounds({ x: 0, y: 0, width: w, height: h });
    return;
  }

  if (uiView) uiView.setBounds({ x: 0, y: 0, width: w, height: h });

  const midX = Math.floor((w - GAP) / 2);
  const midY = Math.floor((h - GAP) / 2);

  views[0].setBounds({ x: 0, y: NAV_HEIGHT, width: midX, height: midY - NAV_HEIGHT });
  views[1].setBounds({ x: midX + GAP, y: NAV_HEIGHT, width: w - (midX + GAP), height: midY - NAV_HEIGHT });
  views[2].setBounds({ x: 0, y: midY + GAP + NAV_HEIGHT, width: midX, height: h - (midY + GAP + NAV_HEIGHT) });
  views[3].setBounds({ x: midX + GAP, y: midY + GAP + NAV_HEIGHT, width: w - (midX + GAP), height: h - (midY + GAP + NAV_HEIGHT) });
}

// Window bounds can be stale during maximize/unmaximize transitions, so
// relayout is coalesced and re-run once more after the WM settles.
function requestLayout() {
  clearTimeout(relayoutTimer);
  relayoutTimer = setTimeout(updateLayout, 60);
  setTimeout(updateLayout, 300); // final settle pass (idempotent)
}

function createWindow() {
  // Strip framing headers only for the quadrant hosts, leave the rest of the
  // session's security headers intact.
  session.defaultSession.webRequest.onHeadersReceived((details, callback) => {
    if (!FRAMED_HOSTS.test(details.url)) {
      callback({ cancel: false });
      return;
    }
    const headers = { ...details.responseHeaders };
    ['x-frame-options', 'X-Frame-Options', 'content-security-policy', 'Content-Security-Policy'].forEach(h => delete headers[h]);
    callback({ cancel: false, responseHeaders: headers });
  });

  const icon = nativeImage.createFromPath(path.join(__dirname, 'icon.png'));

  mainWindow = new BrowserWindow({
    width: 1400,
    height: 900,
    title: 'Professional Creative Browser',
    backgroundColor: '#121212',
    icon: icon.isEmpty() ? undefined : icon,
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true,
      sandbox: false,
      preload: path.join(__dirname, 'preload.js'),
    }
  });

  mainWindow.webContents.on('context-menu', (e, props) => {
    const { Menu, MenuItem } = require('electron');
    const menu = new Menu();
    menu.append(new MenuItem({ label: 'Copy', role: 'copy' }));
    menu.append(new MenuItem({ label: 'Paste', role: 'paste' }));
    menu.append(new MenuItem({ label: 'Select All', role: 'selectAll' }));
    menu.popup(mainWindow);
  });

  uiView = new BrowserView({
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true,
      sandbox: false,
      preload: path.join(__dirname, 'preload.js'),
    }
  });
  uiView.webContents.loadURL('app://local/controls.html');
  mainWindow.addBrowserView(uiView);

  const urls = [
    'https://www.google.com',
    'https://www.bing.com/images/create',
    'https://www.pinterest.com',
    'app://local/workspace.html'
  ];

  views = urls.map((url, index) => createView(url, index === 3));
  views.forEach(view => mainWindow.addBrowserView(view));

  updateLayout();

  mainWindow.on('resize', requestLayout);
  mainWindow.on('maximize', requestLayout);
  mainWindow.on('unmaximize', requestLayout);
}

// --- IPC LISTENERS ---

ipcMain.on('nav', (event, data) => {
  const view = views[data && data.id];
  if (view && typeof data.url === 'string') {
    let targetUrl = data.url.trim();
    if (!targetUrl.startsWith('http://') && !targetUrl.startsWith('https://')) {
      targetUrl = 'https://' + targetUrl;
    }
    view.webContents.loadURL(targetUrl).catch(err => console.error("Nav Error:", err.message));
  }
});

ipcMain.on('expand-view', (event, id) => {
  if (typeof id !== 'number' || !views[id]) return;
  expandedId = id;
  mainWindow.setTopBrowserView(views[id]);
  updateLayout();
});

ipcMain.on('reset-layout', () => {
  expandedId = -1;
  requestLayout();
});

// CORS-free image fetch for the workspace (drops, context-menu sends, AI).
// Runs in the main process where cross-origin rules don't taint canvases.
ipcMain.on('fetch-image', async (event, url) => {
  const wc = event.sender;
  if (typeof url !== 'string' || !/^https?:\/\//.test(url)) {
    wc.send('fetch-image-result', { ok: false, error: 'Unsupported URL.' });
    return;
  }
  try {
    const response = await net.fetch(url, { headers: { 'User-Agent': CHROME_UA } });
    if (!response.ok) throw new Error('HTTP ' + response.status);
    const buffer = await response.arrayBuffer();
    const contentType = response.headers.get('content-type') || 'image/png';
    wc.send('fetch-image-result', { ok: true, buffer, contentType });
  } catch (err) {
    wc.send('fetch-image-result', { ok: false, error: 'Fetch failed: ' + err.message });
  }
});

ipcMain.on('image-dropped', (event, url) => {
  if (views[3] && typeof url === 'string') {
    views[3].webContents.send('clipboard-image', url);
  }
});

// AI background removal: the API key lives in the MAIN process (from .env),
// never in the renderer. The workspace sends a base64 PNG; we reply privately.
ipcMain.on('ai-remove-bg-request', async (event, data) => {
  const wc = event.sender;
  if (!data || typeof data.base64 !== 'string') {
    wc.send('ai-remove-bg-result', { ok: false, error: 'Bad request.' });
    return;
  }
  try {
    const fs = require('fs');
    const dotenv = require('dotenv');
    dotenv.config({ path: path.join(__dirname, '.env') });
    dotenv.config({ path: path.join(__dirname, '.env.local') });
    const key = process.env.GEMINI_API_KEY;
    if (!key) {
      wc.send('ai-remove-bg-result', { ok: false, error: 'No GEMINI_API_KEY configured. Add it to the .env file next to the app.' });
      return;
    }
    const { GoogleGenAI } = require('@google/genai');
    const ai = new GoogleGenAI({ apiKey: key });
    const response = await ai.models.generateContent({
      model: 'gemini-2.5-flash-image',
      contents: {
        parts: [
          { inlineData: { data: data.base64, mimeType: 'image/png' } },
          { text: 'Remove the background from this image. Return only the subject with a transparent background. Output ONLY the edited image.' }
        ]
      }
    });
    let resultBase64 = '';
    for (const part of (response.candidates?.[0]?.content?.parts || [])) {
      if (part.inlineData) { resultBase64 = part.inlineData.data; break; }
    }
    if (!resultBase64) {
      wc.send('ai-remove-bg-result', { ok: false, error: 'The model returned no image.' });
      return;
    }
    wc.send('ai-remove-bg-result', { ok: true, resultBase64 });
  } catch (error) {
    console.error('AI remove-bg error:', error.message);
    wc.send('ai-remove-bg-result', { ok: false, error: 'AI request failed: ' + error.message });
  }
});

app.whenReady().then(() => {
  handleAppProtocol();
  createWindow();
});
app.on('window-all-closed', () => { if (process.platform !== 'darwin') app.quit(); });
