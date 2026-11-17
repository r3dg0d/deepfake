"""Render the control center with plain Qt (PySide6) when Quickshell itself is not installed.

A minimal stand-in for the Quickshell types the widget uses: ShellRoot, FloatingWindow, Singleton,
Quickshell.env, Process, StdioCollector and IpcHandler. It checks that the QML loads, binds and
lays out without warnings; it does not replace a run under the real `qs`.

    python tests/quickshell_shim/run.py overlays/deepfake-preview/quickshell
"""
import os
import shlex
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QT_QUICK_CONTROLS_STYLE", "Basic")
from PySide6.QtCore import Property, QObject, QProcess, QTimer, QtMsgType, QUrl, Signal, Slot, qInstallMessageHandler
from PySide6.QtGui import QGuiApplication
from PySide6.QtQml import QQmlApplicationEngine, qmlRegisterSingletonInstance, qmlRegisterType

HERE = Path(__file__).parent

class Env(QObject):
    @Slot(str, result="QVariant")
    def env(self, name):
        return os.environ.get(name)

class StdioCollector(QObject):
    streamFinished = Signal()
    textChanged = Signal()
    def __init__(self, parent=None):
        super().__init__(parent); self._text = ""
    @Property(str, notify=textChanged)
    def text(self): return self._text
    def finish(self, text):
        self._text = text; self.textChanged.emit(); self.streamFinished.emit()

class Process(QObject):
    commandChanged = Signal(); runningChanged = Signal(); stdoutChanged = Signal(); stderrChanged = Signal()
    calls = []
    def __init__(self, parent=None):
        super().__init__(parent); self._command = []; self._running = False; self._out = None; self._err = None; self._p = None
    @Property("QVariantList", notify=commandChanged)
    def command(self): return self._command
    @command.setter
    def command(self, v): self._command = list(v); self.commandChanged.emit()
    @Property(QObject, notify=stdoutChanged)
    def stdout(self): return self._out
    @stdout.setter
    def stdout(self, v): self._out = v; self.stdoutChanged.emit()
    @Property(QObject, notify=stderrChanged)
    def stderr(self): return self._err
    @stderr.setter
    def stderr(self, v): self._err = v; self.stderrChanged.emit()
    @Property(bool, notify=runningChanged)
    def running(self): return self._running
    @running.setter
    def running(self, v):
        if v and not self._running:
            self._running = True; self.runningChanged.emit()
            Process.calls.append(list(self._command))
            if os.environ.get("QS_SHIM_LOG"): open(os.environ["QS_SHIM_LOG"], "a").write(shlex.join(map(str, self._command)) + "\n")
            self._p = QProcess(self); self._p.finished.connect(self._done); self._p.errorOccurred.connect(self._error)
            self._p.start(str(self._command[0]), [str(a) for a in self._command[1:]])
    def _error(self, e):
        if self._running and self._p.state() == QProcess.NotRunning: self._done()
    def _done(self, *a):
        if not self._running: return
        out = bytes(self._p.readAllStandardOutput()).decode(errors="replace"); err = bytes(self._p.readAllStandardError()).decode(errors="replace")
        self._running = False; self.runningChanged.emit()
        if self._out: self._out.finish(out)
        if self._err: self._err.finish(err)

class IpcHandler(QObject):
    targetChanged = Signal()
    def __init__(self, parent=None): super().__init__(parent); self._t = ""
    @Property(str, notify=targetChanged)
    def target(self): return self._t
    @target.setter
    def target(self, v): self._t = v

messages = []
def handler(kind, ctx, msg):
    level = {QtMsgType.QtWarningMsg: "WARN", QtMsgType.QtCriticalMsg: "ERROR", QtMsgType.QtFatalMsg: "ERROR"}.get(kind, "INFO")
    messages.append((level, msg)); print(f"{level}: {msg}", flush=True)

def main():
    import shutil
    import tempfile
    src = Path(sys.argv[1]).resolve()
    shell = Path(tempfile.mkdtemp(prefix="qsshell-")) / "shell"
    shutil.copytree(src, shell)
    # Quickshell registers `pragma Singleton` files of the shell directory itself; plain Qt needs a qmldir
    singles = [f.stem for f in shell.glob("*.qml") if "pragma Singleton" in f.read_text()]
    (shell / "qmldir").write_text("".join(f"singleton {n} 1.0 {n}.qml\n" for n in singles))
    qInstallMessageHandler(handler)
    app = QGuiApplication(sys.argv[:1])
    env = Env()
    qmlRegisterSingletonInstance(Env, "Quickshell", 1, 0, "Quickshell", env)
    for name in ("ShellRoot", "FloatingWindow", "Singleton"):
        qmlRegisterType(QUrl.fromLocalFile(str(HERE / f"{name}.qml")), "Quickshell", 1, 0, name)
    qmlRegisterType(Process, "Quickshell.Io", 1, 0, "Process")
    qmlRegisterType(StdioCollector, "Quickshell.Io", 1, 0, "StdioCollector")
    qmlRegisterType(IpcHandler, "Quickshell.Io", 1, 0, "IpcHandler")
    engine = QQmlApplicationEngine()
    engine.quit.connect(app.quit)
    engine.load(QUrl.fromLocalFile(str(shell / "shell.qml")))
    if not engine.rootObjects():
        print("ERROR: failed to load"); sys.exit(2)
    QTimer.singleShot(int(os.environ.get("QS_SHIM_TIMEOUT_MS", "12000")), app.quit)
    app.exec()
    bad = [m for lvl, m in messages if lvl != "INFO"]
    sys.exit(1 if bad else 0)
main()
