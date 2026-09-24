# Creative Browser

A desktop app (Electron) that shows four live web quadrants — Search, AI image
generation, visual references, and a built-in Fabric.js design workspace — with
a shared navigation bar and cross-quadrant image tools.

## Run

```bash
npm install
cp .env.example .env    # optional: add your GEMINI_API_KEY for AI Remove BG
npm start
```

> On Linux, if the window fails to launch on some GPU drivers, run:
> `npm start -- --disable-gpu --in-process-gpu`

## Layout

| Quadrant | Purpose |
|----------|---------|
| Search (top-left) | Google |
| AI Assets (top-right) | Bing Image Creator |
| Reference (bottom-left) | Pinterest |
| Designer (bottom-right) | Built-in design workspace |

Type a URL in any nav bar and press **Enter** or **GO** to navigate that
quadrant. The 🖥️ button in the workspace expands it to fullscreen.

## Workspace tools

- **Shapes / frames / text** — add from the sidebar drawers
- **Brushes** — pencil, circle, spray, pattern; continuous or stamp mode
- **Slice** — select a shape + an image together (drag a box), then ✂️ Slice
  to cut the image into "inside the shape" and "outside" pieces
- **Weld / Unweld** — group or ungroup the current selection
- **Clone stamp / spot healing** — retouch images (Alt+Click sets clone source)
- **Eraser** — erase image pixels directly (Esc exits any tool)
- **Color Key** — click a color on an image to make it transparent
- **Filters** — brightness/contrast/saturation/blur + quick presets
- **Remove BG (✨)** — AI background removal via Gemini (see below)
- **Undo/Redo, layers, lock, snap, import, save PNG**

Right-click any image in another quadrant → **Send Image to Workspace**.
Double-click a masked image to reposition it inside its frame.

## AI Remove BG (optional)

The Gemini API key is read **only in the main process** from `.env`
(never bundled into the renderer). Without a key the button explains what's
missing instead of failing silently.

```
GEMINI_API_KEY=your_key_here
```

## Packaging

```bash
npm run dist
```

Output lands in `release/` (AppImage on Linux, NSIS installer on Windows).
