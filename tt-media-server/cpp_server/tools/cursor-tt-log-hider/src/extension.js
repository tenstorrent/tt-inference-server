const vscode = require("vscode");

const LOG_HIDER_ENABLED = "ttLogHider.enabledEditors";
const LOG_HIDER_GUTTER_ICON = "resources/log-hidden.svg";

let hiddenLogDecoration;
let extensionUri;

function getLogRegex() {
  const pattern = vscode.workspace
    .getConfiguration("ttLogHider")
    .get("logCallPattern", "^\\s*TT_LOG_[A-Z]+\\s*\\(");

  try {
    return new RegExp(pattern);
  } catch {
    return /^\s*TT_LOG_[A-Z]+\s*\(/;
  }
}

function getSupportedLanguages() {
  return vscode.workspace
    .getConfiguration("ttLogHider")
    .get("languages", ["c", "cpp", "cuda-cpp"]);
}

function shouldApply(editor) {
  return getSupportedLanguages().includes(editor.document.languageId);
}

function getEnabledEditorUris(context) {
  const values = context.globalState.get(LOG_HIDER_ENABLED, []);
  return new Set(values);
}

async function setEnabledEditorUris(context, values) {
  await context.globalState.update(LOG_HIDER_ENABLED, [...values.values()]);
}

function buildHiddenLogRanges(editor, logRegex) {
  const options = [];
  const document = editor.document;
  for (let line = 0; line < document.lineCount; line += 1) {
    const lineText = document.lineAt(line).text;
    if (!logRegex.test(lineText)) {
      continue;
    }

    const range = new vscode.Range(
      new vscode.Position(line, 0),
      new vscode.Position(line, lineText.length)
    );
    options.push({
      range,
      hoverMessage: "TT_LOG line hidden by TT Log Hider"
    });
  }
  return options;
}

function getDecoration() {
  if (hiddenLogDecoration) {
    return hiddenLogDecoration;
  }

  const showGutterHint = vscode.workspace
    .getConfiguration("ttLogHider")
    .get("showGutterHint", true);

  hiddenLogDecoration = vscode.window.createTextEditorDecorationType({
    opacity: "0",
    isWholeLine: false,
    ...(showGutterHint
      ? {
          gutterIconPath: vscode.Uri.joinPath(
            extensionUri ?? vscode.Uri.file(""),
            LOG_HIDER_GUTTER_ICON
          ),
          gutterIconSize: "contain"
        }
      : {})
  });

  return hiddenLogDecoration;
}

function applyToEditor(editor) {
  if (!shouldApply(editor)) {
    return;
  }
  const decoration = getDecoration();
  editor.setDecorations(decoration, buildHiddenLogRanges(editor, getLogRegex()));
}

function clearFromEditor(editor) {
  if (!hiddenLogDecoration) {
    return;
  }
  editor.setDecorations(hiddenLogDecoration, []);
}

async function enableForEditor(context, editor) {
  if (!shouldApply(editor)) {
    vscode.window.showInformationMessage(
      `TT Log Hider supports: ${getSupportedLanguages().join(", ")}`
    );
    return;
  }
  const enabled = getEnabledEditorUris(context);
  enabled.add(editor.document.uri.toString());
  await setEnabledEditorUris(context, enabled);
  applyToEditor(editor);
}

async function disableForEditor(context, editor) {
  const enabled = getEnabledEditorUris(context);
  enabled.delete(editor.document.uri.toString());
  await setEnabledEditorUris(context, enabled);
  clearFromEditor(editor);
}

function isEnabledForEditor(context, editor) {
  const enabled = getEnabledEditorUris(context);
  return enabled.has(editor.document.uri.toString());
}

function activate(context) {
  extensionUri = context.extensionUri;

  context.subscriptions.push(
    vscode.commands.registerCommand("ttLogHider.toggle", async () => {
      const editor = vscode.window.activeTextEditor;
      if (!editor) {
        return;
      }
      if (isEnabledForEditor(context, editor)) {
        await disableForEditor(context, editor);
        return;
      }
      await enableForEditor(context, editor);
    })
  );

  context.subscriptions.push(
    vscode.commands.registerCommand("ttLogHider.hide", async () => {
      const editor = vscode.window.activeTextEditor;
      if (!editor) {
        return;
      }
      await enableForEditor(context, editor);
    })
  );

  context.subscriptions.push(
    vscode.commands.registerCommand("ttLogHider.show", async () => {
      const editor = vscode.window.activeTextEditor;
      if (!editor) {
        return;
      }
      await disableForEditor(context, editor);
    })
  );

  context.subscriptions.push(
    vscode.workspace.onDidChangeTextDocument((event) => {
      const editor = vscode.window.visibleTextEditors.find(
        (candidate) =>
          candidate.document.uri.toString() === event.document.uri.toString()
      );
      if (!editor || !isEnabledForEditor(context, editor)) {
        return;
      }
      applyToEditor(editor);
    })
  );

  context.subscriptions.push(
    vscode.window.onDidChangeActiveTextEditor((editor) => {
      if (!editor) {
        return;
      }
      if (isEnabledForEditor(context, editor)) {
        applyToEditor(editor);
      }
    })
  );
}

function deactivate() {
  if (hiddenLogDecoration) {
    hiddenLogDecoration.dispose();
    hiddenLogDecoration = undefined;
  }
}

module.exports = {
  activate,
  deactivate
};
