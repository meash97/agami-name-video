# AGAMI · Name Video Maker

Open **https://meash97.github.io/agami-name-video/**, pick a template, type a name, tap **Make video**.

**Install as an app:** on Android Chrome tap *Install* when the banner appears (or menu → *Add to Home screen*); on iPhone open in Safari → Share → *Add to Home Screen*. Works offline after the first open; updates arrive automatically.

Everything runs in the browser (WebCodecs H.264 + built-in MP4 muxer). No server, no internet needed after loading. Best in Chrome (Android / desktop).

## Templates
- **Birthday · Notebook** — "___ JI, wishing you a happiest birthday…"
- **Anniversary · Sketchbook** — type the whole first line (both names); the hand finishes writing, leaves, and two hearts are drawn after "Happiest Anniversary"

`tools/` holds the build scripts used to prepare a template (clean plate video, handwriting alphabet, per-frame ink colour) and assemble `index.html`.
