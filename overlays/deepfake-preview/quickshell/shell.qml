import QtQuick
import QtQuick.Layouts
import Quickshell
import Quickshell.Io

ShellRoot {
    id: root
    property var session: ({})
    property bool opened: true
    property bool compact: false
    property real now: Date.now() / 1000
    property bool active: !!session.active && now - Number(session.ts || 0) < 15
    property bool stale: !!session.active && !active
    property string stateDir: (Quickshell.env("XDG_STATE_HOME") || Quickshell.env("HOME") + "/.local/state") + "/deepfake"
    property int frameSerial: 0
    property string message: Quickshell.env("DEEPFAKE_WIDGET_ERROR") || ""
    property var fpsHist: []
    property var latHist: []
    function metric(value, suffix, digits) { return value === null || value === undefined ? "—" : Number(value).toFixed(digits === undefined ? 1 : digits) + suffix }
    function dur(s) { s = Math.floor(Number(s) || 0); const h = Math.floor(s / 3600), m = Math.floor(s % 3600 / 60); return (h ? h + "h " : "") + m + "m " + String(s % 60).padStart(2, "0") + "s" }
    function push(arr, v) { const a = arr.slice(-59); a.push(Number(v) || 0); return a }
    FileView {
        id: stateFile
        path: root.stateDir + "/session.json"
        watchChanges: true
        onFileChanged: reload()
        onLoaded: { try { root.session = JSON.parse(text()) } catch (e) { root.session = ({}) } }
    }
    Timer {
        interval: 1000; running: true; repeat: true
        onTriggered: {
            root.now = Date.now() / 1000; stateFile.reload()
            if (root.active) { root.fpsHist = root.push(root.fpsHist, root.session.output_fps); root.latHist = root.push(root.latHist, root.session.latency_ms) }
        }
    }
    Timer { interval: 100; running: root.active && root.opened; repeat: true; onTriggered: root.frameSerial++ }
    IpcHandler {
        target: "deepfake"
        function show(): void { root.opened = true }
        function toggle(): void { root.opened = !root.opened }
    }
    Process { id: stopSession; stdout: StdioCollector { onStreamFinished: root.message = text.trim() } }
    property var vcam: ({ vcam: "", loaded: false })
    readonly property bool vcamOn: root.active && !!root.session.output_device
    property bool starting: false
    function ctl(args) { runProc.command = ["bash", Quickshell.shellPath("run.sh")].concat(args); runProc.running = true; if (args[0] !== "stop") { root.starting = true; startGuard.restart() } }
    Timer { id: startGuard; interval: 25000; onTriggered: root.starting = false }
    onActiveChanged: if (active) starting = false
    Process { id: runProc; stdout: StdioCollector { onStreamFinished: if (text.trim()) root.message = text.trim() } }
    Process { id: vcamProc; command: ["bash", Quickshell.shellPath("run.sh"), "status"]
        stdout: StdioCollector { onStreamFinished: { try { root.vcam = JSON.parse(text) } catch (e) {} } } }
    Timer { interval: 5000; running: root.opened; repeat: true; triggeredOnStart: true; onTriggered: vcamProc.running = true }
    Process { id: snap; stdout: StdioCollector { onStreamFinished: root.message = text.trim() } }

    component Stat: Rectangle {
        property string label; property string value; property color tint: "#d4b8ff"
        Layout.fillWidth: true; implicitHeight: 58; radius: 14; color: "#120f22"; border.color: "#2a2344"
        Column {
            anchors.centerIn: parent; spacing: 5
            Text { anchors.horizontalCenter: parent.horizontalCenter; font.family: "Cellforge"; text: label; color: "#8a84a8"; font.pixelSize: 9; font.letterSpacing: 1.5 }
            Text { anchors.horizontalCenter: parent.horizontalCenter; font.family: "Zerodot"; text: root.active ? value : "—"; color: tint; font.pixelSize: 16 }
        }
    }
    component Spark: Canvas {
        property var data: []; property color stroke: "#b46bff"; property string label
        Layout.fillWidth: true; implicitHeight: 46
        onDataChanged: requestPaint()
        onPaint: {
            const c = getContext("2d"); c.reset()
            const d = data; if (d.length < 2) return
            const max = Math.max.apply(null, d) * 1.15 || 1
            c.beginPath()
            for (let i = 0; i < d.length; ++i) { const x = i / 59 * width, y = height - 4 - d[i] / max * (height - 8); i ? c.lineTo(x, y) : c.moveTo(x, y) }
            c.strokeStyle = stroke; c.lineWidth = 2; c.stroke()
            c.lineTo((d.length - 1) / 59 * width, height); c.lineTo(0, height); c.closePath()
            const g = c.createLinearGradient(0, 0, 0, height); g.addColorStop(0, Qt.rgba(stroke.r, stroke.g, stroke.b, 0.35)); g.addColorStop(1, "transparent")
            c.fillStyle = g; c.fill()
        }
        Text { x: 0; y: 0; font.family: "Cellforge"; text: label; color: "#8a84a8"; font.pixelSize: 9 }
    }

    FloatingWindow {
        id: window
        title: "Galaxy Deepfake"
        visible: root.opened
        implicitWidth: 680; implicitHeight: root.compact ? 470 : 760
        minimumSize: Qt.size(480, 420)
        color: "transparent"
        onClosed: root.opened = false
        Rectangle {
            anchors.fill: parent; radius: 22; border.color: "#3a2f5c"; border.width: 1
            gradient: Gradient {
                GradientStop { position: 0; color: "#120c24" }
                GradientStop { position: 1; color: "#07060d" }
            }
        }
        ColumnLayout {
            anchors.fill: parent; anchors.margins: 20; spacing: 14
            RowLayout {
                Layout.fillWidth: true; spacing: 10
                Text {
                    Layout.fillWidth: true; font.family: "Cellforge"; text: "✦  O R B I T  /  D E E P F A K E"; font.pixelSize: 12; color: "#d4b8ff"
                    MouseArea { anchors.fill: parent; cursorShape: Qt.SizeAllCursor; onPressed: window.startSystemMove() }
                }
                Rectangle {
                    implicitWidth: liveRow.implicitWidth + 18; implicitHeight: 24; radius: 12
                    color: root.active ? "#1d3a2a" : root.stale ? "#3a2d12" : "#1a1630"
                    border.color: root.active ? "#7dffa8" : root.stale ? "#ffd479" : "#3a2f5c"
                    RowLayout {
                        id: liveRow; anchors.centerIn: parent; spacing: 6
                        Rectangle {
                            width: 8; height: 8; radius: 4; color: root.active ? "#7dffa8" : root.stale ? "#ffd479" : "#5d5878"
                            SequentialAnimation on opacity { running: root.active; loops: Animation.Infinite; NumberAnimation { to: 0.25; duration: 700 } NumberAnimation { to: 1; duration: 700 } }
                        }
                        Text { font.family: "Cellforge"; font.pixelSize: 10; color: "#f2f0ff"
                            text: root.active ? String(root.session.state || "live").toUpperCase() : root.stale ? "NO HEARTBEAT" : "OFFLINE" }
                    }
                }
                GalaxyButton { text: root.compact ? "󰊓" : "󰊔"; hint: root.compact ? "Show details" : "Compact view"; onClicked: root.compact = !root.compact }
                GalaxyButton { text: "×"; hint: "Hide widget · webcam keeps running"; onClicked: root.opened = false }
            }
            Rectangle {
                Layout.fillWidth: true; Layout.fillHeight: true
                color: "#0b0914"; radius: 18; border.color: root.active ? "#5b3d99" : "#26223a"; clip: true
                Image {
                    id: preview; anchors.fill: parent; anchors.margins: 6
                    fillMode: Image.PreserveAspectFit; cache: false; asynchronous: false
                    source: root.active ? "file://" + root.stateDir + "/preview.png?frame=" + root.frameSerial : ""
                }
                Column {
                    anchors.centerIn: parent; spacing: 12
                    visible: !root.active || preview.status === Image.Error || preview.status === Image.Null
                    Text { font.family: "JetBrainsMono Nerd Font"; anchors.horizontalCenter: parent.horizontalCenter; text: root.active ? "󰄀" : "󰗟"; font.pixelSize: 54; color: "#b46bff" }
                    Text { font.family: "Cellforge"; anchors.horizontalCenter: parent.horizontalCenter; color: "#9a96b3"; font.pixelSize: 14
                        text: root.active ? "Waiting for preview…" : root.stale ? "Session stopped responding" : "Start deepfake webcam to go live" }
                }
                Row {
                    anchors.left: parent.left; anchors.bottom: parent.bottom; anchors.margins: 14; spacing: 8
                    visible: root.active
                    Repeater {
                        model: [root.session.resolution || "—", "󰆍 " + root.dur(root.session.uptime_s), (root.session.framegen_enabled ? "FG " + String(root.session.framegen_multiplier || "auto") + "×" : "FG off")]
                        Rectangle { required property var modelData; implicitWidth: t.implicitWidth + 16; implicitHeight: 22; radius: 11; color: "#cc07060d"; border.color: "#3a2f5c"
                            Text { id: t; anchors.centerIn: parent; text: modelData; font.family: "Cellforge"; font.pixelSize: 10; color: "#d4b8ff" } }
                    }
                }
            }
            GridLayout {
                Layout.fillWidth: true; columns: 4; columnSpacing: 10; rowSpacing: 10
                Stat { label: "SWAP FPS"; value: root.metric(root.session.alphaface_fps, "") }
                Stat { label: "OUTPUT FPS"; value: root.metric(root.session.output_fps, "") }
                Stat { label: "LATENCY"; value: root.metric(root.session.latency_ms, " ms", 0); tint: Number(root.session.latency_ms) > 120 ? "#ffd479" : "#d4b8ff" }
                Stat { label: "GPU"; value: root.metric(root.session.gpu_util_pct, "%", 0) }
                Stat { visible: !root.compact; label: "VRAM"; value: root.metric(Number(root.session.vram_mb) / 1024, " GB") }
                Stat { visible: !root.compact; label: "FRAMES"; value: root.metric(root.session.generated_frames ?? root.session.frames, "", 0) }
                Stat { visible: !root.compact; label: "DROPPED"; value: root.metric(root.session.dropped_source_frames, "", 0); tint: Number(root.session.dropped_source_frames) > 0 ? "#ff8fb1" : "#d4b8ff" }
                Stat { visible: !root.compact; label: "UPTIME"; value: root.dur(root.session.uptime_s) }
            }
            RowLayout {
                Layout.fillWidth: true; spacing: 12; visible: !root.compact
                Spark { label: "OUTPUT FPS · 60s"; data: root.fpsHist; stroke: "#b46bff" }
                Spark { label: "LATENCY · 60s"; data: root.latHist; stroke: "#6bd0ff" }
            }
            Text {
                Layout.fillWidth: true; visible: !root.compact; font.family: "Cellforge"; textFormat: Text.PlainText; wrapMode: Text.Wrap; color: "#9a96b3"; font.pixelSize: 12
                text: "󰀄 Face: " + (root.session.source_face || "none") + "   󰄀 Input: " + (root.session.input_device || "default") + (root.session.session_id ? "   # " + root.session.session_id : "")
            }
            Text {
                Layout.fillWidth: true; visible: !!root.session.framegen_warning; font.family: "Cellforge"
                text: "⚠ " + (root.session.framegen_warning || ""); textFormat: Text.PlainText; wrapMode: Text.Wrap
                maximumLineCount: 3; elide: Text.ElideRight; color: "#ffd479"; font.pixelSize: 11
            }
            Text { font.family: "Cellforge"; Layout.fillWidth: true; text: root.message || "× hides the panel · Stop ends the camera session"; textFormat: Text.PlainText; wrapMode: Text.Wrap; color: "#8a84a8"; font.pixelSize: 11 }
            RowLayout {
                Layout.fillWidth: true; spacing: 8
                GalaxyButton {
                    text: "󰄄 Snapshot"; enabled: root.active && !snap.running; hint: "Save the current preview to ~/Pictures/deepfake"
                    onClicked: { snap.command = ["bash", "-c", 'd="$HOME/Pictures/deepfake"; mkdir -p "$d"; f="$d/$(date +%Y%m%d-%H%M%S).png"; cp "$1/preview.png" "$f" && echo "Saved $f"', "snap", root.stateDir]; snap.running = true }
                }
                GalaxyButton { text: "󰉋"; hint: "Open state folder"; onClicked: Qt.openUrlExternally("file://" + root.stateDir) }
                Rectangle {   // virtual cam status
                    implicitWidth: vcRow.implicitWidth + 16; implicitHeight: 26; radius: 13
                    color: root.vcamOn ? "#1d3a2a" : "#120f22"; border.color: root.vcamOn ? "#7dffa8" : root.vcam.loaded ? "#3a2f5c" : "#5a3a1a"
                    RowLayout { id: vcRow; anchors.centerIn: parent; spacing: 6
                        Rectangle { width: 7; height: 7; radius: 4; color: root.vcamOn ? "#7dffa8" : root.vcam.loaded ? "#5d5878" : "#ffd479" }
                        Text { font.family: "Cellforge"; font.pixelSize: 10; color: "#d4b8ff"
                            text: root.vcamOn ? "VCAM " + root.session.output_device : root.vcam.loaded ? "VCAM ready " + root.vcam.vcam : "VCAM not loaded" } }
                }
                GalaxyButton {
                    text: root.vcamOn ? "󰗟 Stop virtual cam" : "󰄀 Start virtual cam"
                    enabled: root.vcam.loaded && !runProc.running && !root.starting
                    hint: root.vcam.loaded ? (root.active ? "Restarts the session " + (root.vcamOn ? "without" : "with") + " the virtual cam" : "Start the webcam and feed " + root.vcam.vcam)
                                           : "v4l2loopback isn't loaded — see ~/rice-staging/r3dg0d-virtualcam.nix"
                    onClicked: root.vcamOn ? root.ctl(["restart"]) : root.ctl([root.active ? "restart" : "start", "virtual"])
                }
                GalaxyButton {
                    visible: !root.active
                    text: root.starting ? "󰑐 Starting…" : "󰄀 Start webcam"; enabled: !root.starting && !runProc.running
                    hint: "deepfake webcam (last face, config.toml defaults) · log: ~/.cache/deepfake-widget.log"
                    onClicked: root.ctl(["start"])
                }
                GalaxyButton {
                    visible: root.active
                    text: "󰓛 Stop webcam"; enabled: !stopSession.running
                    onClicked: { stopSession.command = ["python3", Quickshell.shellPath("control.py"), String(root.session.session_id)]; stopSession.running = true }
                }
            }
        }
    }
}
