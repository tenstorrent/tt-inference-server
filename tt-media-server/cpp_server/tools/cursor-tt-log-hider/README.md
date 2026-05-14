# TT Log Hider (Cursor / VS Code extension)

Hide `TT_LOG_*` lines in C/C++ files while browsing source code.

## Commands

- `TT Log Hider: Hide Logs`
- `TT Log Hider: Show Logs`
- `TT Log Hider: Toggle Hide Logs`

## Settings

- `ttLogHider.logCallPattern` (default: `^\s*TT_LOG_[A-Z]+\s*\(`)
- `ttLogHider.languages` (default: `["c", "cpp", "cuda-cpp"]`)
- `ttLogHider.showGutterHint` (default: `true`)

## Local usage

```bash
npm install
```

Then load the extension folder in Cursor/VS Code extension development mode.
