// Preload: contextIsolation-safe bridge, shared by ALL views (remote and local).
// Sending is restricted to an allowlist; receiving is restricted to three
// app-internal channels. No keys or objects ever pass through here.

const { contextBridge, ipcRenderer } = require('electron');

const SEND_ONLY = new Set([
  'nav',
  'expand-view',
  'reset-layout',
  'image-dropped',
  'ai-remove-bg-request',
  'fetch-image',
]);

const RECEIVE_ONLY = new Set([
  'clipboard-image',     // main -> workspace: image URL from another view's context menu
  'ai-remove-bg-result', // main -> workspace: AI processing result
  'fetch-image-result',  // main -> workspace: fetched image bytes
]);

contextBridge.exposeInMainWorld('electron', {
  send: (channel, data) => {
    if (SEND_ONLY.has(channel)) {
      ipcRenderer.send(channel, data);
    }
  },
  on: (channel, func) => {
    if (RECEIVE_ONLY.has(channel) && typeof func === 'function') {
      ipcRenderer.on(channel, (event, ...args) => func(...args));
    }
  },
});
