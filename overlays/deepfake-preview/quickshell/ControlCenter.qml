import QtQuick
import QtQuick.Layouts
import QtQuick.Window
import QtQuick.Controls
import QtQuick.Dialogs
import Quickshell
import Quickshell.Io

// Deepfake control center in the Galaxy theme of the rice (~/.config/quickshell/galaxy).
// Everything shown comes from `deepfake desktop request snapshot`; a metric the backend did not
// report is drawn as "—", never estimated. Controls only call actions the backend implements.
ShellRoot {
    id: root
    property bool opened: true
    property bool expanded: false
    onExpandedChanged: {
        const nativeWindow = window.contentItem.Window.window
        if (nativeWindow) nativeWindow.visibility = expanded ? Window.FullScreen : Window.Windowed
    }
    property var data: ({session: {}, preferences: {}, models: [], characters: [], devices: [], engines: [], gpu: {}})
    property var session: data.session || ({})
    property var prefs: data.preferences || ({})
    property var gpu: data.gpu || ({})
    property var job: data.model_job || ({})
    property var form: ({})
    property bool initialized: false
    property bool connected: false
    property string page: "Dashboard"
    property string message: "Connecting…"
    property bool messageIsError: false
    property int serial: 0
    property var fpsHist: []
    property var latHist: []
    property var pendingStart: null
    property bool pendingRender: false
    property string modelName: ""
    property string modelAction: ""
    property string confirmText: ""
    readonly property bool active: !!session.active
    readonly property bool running: job.state === "running"
    readonly property bool canStartLive: root.form.engine !== "experimental" || (root.form.mode === "head" && root.form.backend === "liveportrait" && (root.data.engines || []).some(e => e.name === "liveportrait" && e.available))
    readonly property string cli: Quickshell.env("DEEPFAKE_CLI") || "deepfake"
    readonly property bool narrow: window.width < 980
    // test hooks (screenshots): DEEPFAKE_UI_PAGE=Models DEEPFAKE_UI_SIZE=760x560
    Component.onCompleted: { const pg = Quickshell.env("DEEPFAKE_UI_PAGE"); if (pg) page = pg }

    function num(v, digits) { return v === null || v === undefined || v === "" || isNaN(Number(v)) ? "—" : Number(v).toFixed(digits === undefined ? 1 : digits) }
    function live(v, digits) { return root.active ? num(v, digits) : "—" }
    function push(arr, v) { const a = arr.slice(-59); a.push(Number(v) || 0); return a }
    function dur(s) { s = Math.floor(Number(s) || 0); const h = Math.floor(s / 3600), m = Math.floor(s % 3600 / 60); return (h ? h + "h " : "") + m + "m " + String(s % 60).padStart(2, "0") + "s" }
    function edit(key, value) { const f = Object.assign({}, form); f[key] = value; form = f }
    function call(action, extra) {
        if (actionProc.running) { setMessage("Please wait for the current operation", false); return }
        actionProc.command = [cli, "desktop", "request", JSON.stringify(Object.assign({protocol_version: 1, action: action}, extra || {}))]
        actionProc.running = true
    }
    function setMessage(text, isError) { message = text; messageIsError = !!isError }
    function save() { call("preferences", {values: form}) }
    function start(virtual, recording) { pendingStart = {virtual: virtual, recording: recording}; save() }
    function path(url) { return decodeURIComponent(String(url).replace(/^file:\/\//, "")) }
    function confirmModel(action, m) {
        modelName = m.name; modelAction = action
        confirmText = action === "remove_model" ? "Remove the downloaded " + m.name + " weights from the model cache?"
            : (action === "reinstall_model" ? "Redownload and re-verify " : "Download ") + m.name + " (" + (m.size_hint || "size unknown") + ")?"
              + "\n\nCode: " + m.code_license + "\nWeights: " + m.weights_license
              + (action === "reinstall_model" ? "\n\nThe current files stay in place until the new ones pass their checksum and GPU load check." : "")
        confirmation.open()
    }

    Process {
        id: actionProc
        stdout: StdioCollector { onStreamFinished: {
            try { const r = JSON.parse(text); root.setMessage(r.error || r.message || "Done", !r.ok)
                if (root.pendingRender && r.ok) { root.pendingRender = false; Qt.callLater(() => root.call("render_video")) }
                else if (root.pendingStart && r.ok) { const p = root.pendingStart; root.pendingStart = null; Qt.callLater(() => root.call("start", p)) }
                else root.pendingStart = null
                if (!r.ok) root.pendingRender = false
            } catch (e) { root.setMessage(text.slice(-500), true); root.pendingStart = null }
        } }
        stderr: StdioCollector { onStreamFinished: if (text.trim()) root.setMessage(text.trim().slice(-800), true) }
    }
    Process {
        id: reader
        command: [root.cli, "desktop", "request", '{"protocol_version":1,"action":"snapshot"}']
        stdout: StdioCollector { onStreamFinished: {
            try { const d = JSON.parse(text); if (!d.ok) { root.setMessage(d.error, true); root.connected = false; return }
                root.data = d; root.connected = true
                if (!root.initialized) { root.form = d.preferences; root.initialized = true; root.setMessage("Ready · processing stays on this computer", false) }
                if (root.active) { root.fpsHist = root.push(root.fpsHist, d.session.output_fps); root.latHist = root.push(root.latHist, d.session.latency_ms) }
            } catch (e) { root.connected = false; root.setMessage("Backend unavailable: is the deepfake package installed for " + root.cli + "?", true) }
        } }
        stderr: StdioCollector { onStreamFinished: if (text.trim() && !root.connected) root.setMessage(text.trim().slice(-800), true) }
    }
    Timer { interval: root.active ? 1000 : 2000; repeat: true; running: root.opened; triggeredOnStart: true; onTriggered: if (!reader.running) reader.running = true }
    // preview refresh ~10 Hz only while visible and live: the widget never competes with inference
    Timer { interval: 100; repeat: true; running: root.opened && root.active && root.page === "Dashboard"; onTriggered: root.serial++ }
    IpcHandler { target: "deepfake"; function show(): void { root.opened = true } function toggle(): void { root.opened = !root.opened } }
    FileDialog { id: imagePicker; nameFilters: ["Reference images (*.png *.jpg *.jpeg *.webp)"]; onAccepted: importPath.text = root.path(selectedFile) }
    FileDialog { id: outputPicker; fileMode: FileDialog.SaveFile; nameFilters: ["MP4 video (*.mp4)"]; onAccepted: { outputPath.text = root.path(selectedFile); root.edit("recording_path", outputPath.text) } }
    FileDialog { id: videoPicker; nameFilters: ["Video (*.mp4 *.mkv *.mov *.avi *.webm)"]; onAccepted: { videoPath.text = root.path(selectedFile); root.edit("input_video", videoPath.text) } }

    component Label: Text { color: Galaxy.ink; font.family: Galaxy.font; font.pixelSize: 13; textFormat: Text.PlainText; wrapMode: Text.Wrap; Layout.fillWidth: true }
    component Hint: Label { color: Galaxy.muted; font.pixelSize: 11 }
    component KV: RowLayout {
        property string k: ""; property string v: "—"; property color tint: Galaxy.ink
        Layout.fillWidth: true; spacing: 8
        Text { text: k; font.family: Galaxy.font; font.pixelSize: 11; color: Galaxy.label; Layout.preferredWidth: 96 }
        Text { text: v; font.family: Galaxy.mono; font.pixelSize: 11; color: v === "—" ? Galaxy.dim : tint; elide: Text.ElideRight; Layout.fillWidth: true }
    }
    component Preview: Rectangle {
        id: pv
        property string kind: "processed"
        color: Galaxy.base; radius: Galaxy.radiusPanel; clip: true
        border.color: root.active ? Galaxy.edgeHot : Galaxy.line
        // Double-buffered: the next frame decodes off-thread into the hidden image and is swapped in
        // only when ready, so the preview never blanks and the UI thread never blocks on PNG decode.
        property int front: 0
        readonly property string url: root.active && pv.visible ? "file://" + root.data.state_dir + "/" + (pv.kind === "original" ? "original.png" : "preview.png") : ""
        readonly property bool ready: imgA.status === Image.Ready || imgB.status === Image.Ready
        property int tick: root.serial
        onTickChanged: {
            if (!url) return
            const back = front === 0 ? imgB : imgA
            if (back.status !== Image.Loading) back.source = url + "?v=" + tick
        }
        onUrlChanged: { imgA.source = url ? url + "?v=a" : ""; imgB.source = ""; front = 0 }
        Image { id: imgA; anchors.fill: parent; anchors.margins: 6; visible: pv.front === 0
            fillMode: Image.PreserveAspectFit; cache: false; asynchronous: true; mirror: !!root.prefs.mirror; smooth: true
            onStatusChanged: if (status === Image.Ready && pv.front !== 0) pv.front = 0 }
        Image { id: imgB; anchors.fill: parent; anchors.margins: 6; visible: pv.front === 1
            fillMode: Image.PreserveAspectFit; cache: false; asynchronous: true; mirror: !!root.prefs.mirror; smooth: true
            onStatusChanged: if (status === Image.Ready && pv.front !== 1) pv.front = 1 }
        Column {
            anchors.centerIn: parent; spacing: 10
            visible: !root.active || !pv.ready
            Text { anchors.horizontalCenter: parent.horizontalCenter; font.family: Galaxy.icons; text: root.active ? "󰄀" : "󰗟"; font.pixelSize: 52; color: Galaxy.purple }
            Text { anchors.horizontalCenter: parent.horizontalCenter; font.family: Galaxy.font; color: Galaxy.muted; font.pixelSize: 13
                text: root.active ? "Waiting for preview…" : "Camera is off · it opens only when you press Start" }
        }
        Row {
            anchors.left: parent.left; anchors.top: parent.top; anchors.margins: 12; spacing: 6
            Rectangle { implicitWidth: tagT.implicitWidth + 14; implicitHeight: 20; radius: 10; color: "#cc07060d"; border.color: Galaxy.edge
                Text { id: tagT; anchors.centerIn: parent; font.family: Galaxy.font; font.pixelSize: 9; font.letterSpacing: 1.2; color: pv.kind === "original" ? Galaxy.muted : Galaxy.lavender
                    text: pv.kind === "original" ? "ORIGINAL" : "PROCESSED · SYNTHETIC MEDIA" } }
        }
        Row {
            anchors.left: parent.left; anchors.bottom: parent.bottom; anchors.margins: 12; spacing: 6
            visible: root.active && pv.kind !== "original"
            Repeater {
                model: [root.session.resolution || "—", root.num(root.session.source_fps, 0) + " fps in", "󰆍 " + root.dur(root.session.uptime_s),
                        root.session.framegen_enabled ? "FG " + String(root.session.framegen_multiplier || "auto") + "×" : "FG off",
                        root.session.output_device ? "󰄀 " + root.session.output_device : "preview only"]
                Rectangle { required property var modelData; implicitWidth: chip.implicitWidth + 16; implicitHeight: 22; radius: 11; color: "#cc07060d"; border.color: Galaxy.edge
                    Text { id: chip; anchors.centerIn: parent; text: modelData; font.family: Galaxy.font; font.pixelSize: 10; color: Galaxy.lavender } }
            }
        }
    }

    Dialog {
        id: confirmation; parent: window.contentItem; anchors.centerIn: parent; modal: true; width: Math.min(560, window.width - 60)
        padding: 22
        header: Text { text: "✦  R E V I E W"; font.family: Galaxy.font; font.pixelSize: 11; color: Galaxy.lavender; leftPadding: 22; topPadding: 18 }
        contentItem: Text { text: root.confirmText; wrapMode: Text.Wrap; color: Galaxy.ink; font.family: Galaxy.font; font.pixelSize: 13 }
        background: Rectangle { color: Galaxy.surface; border.color: Galaxy.edge; radius: Galaxy.radiusPanel }
        footer: RowLayout {
            spacing: 10
            Item { Layout.fillWidth: true }
            GalaxyButton { text: "Cancel"; onClicked: confirmation.reject() }
            GalaxyButton { text: root.modelAction === "remove_model" ? "Remove" : "Proceed"; primary: root.modelAction !== "remove_model"; danger: root.modelAction === "remove_model"
                Layout.rightMargin: 22; Layout.bottomMargin: 18; onClicked: confirmation.accept() }
        }
        onAccepted: root.call(root.modelAction, {model: root.modelName, confirmed: true})
    }

    FloatingWindow {
        id: window; title: "Galaxy Deepfake"; visible: root.opened; color: "transparent"
        readonly property var forcedSize: (Quickshell.env("DEEPFAKE_UI_SIZE") || "").split("x").map(Number)
        implicitWidth: forcedSize.length === 2 ? forcedSize[0] : 1180; implicitHeight: forcedSize.length === 2 ? forcedSize[1] : 820
        minimumSize: Qt.size(760, 560)
        Timer { interval: 2500; running: !!Quickshell.env("DEEPFAKE_UI_SCREENSHOT"); onTriggered: frame.grabToImage(function(result) { result.saveToFile(Quickshell.env("DEEPFAKE_UI_SCREENSHOT")); if (Quickshell.env("DEEPFAKE_UI_SCREENSHOT_QUIT")) Qt.quit() }) }
        onClosed: root.opened = false
        Shortcut { sequence: "Ctrl+Q"; onActivated: { if (root.active) root.call("stop"); root.opened = false } }
        Shortcut { sequence: "Ctrl+S"; onActivated: root.save() }
        Shortcut { sequence: "F11"; onActivated: root.expanded = !root.expanded }
        Shortcut { sequence: "Escape"; onActivated: root.expanded = false }

        Rectangle {
            id: frame
            anchors.fill: parent
            radius: root.expanded ? 0 : Galaxy.radiusWindow
            color: "transparent"
            Starfield { anchors.fill: parent; cornerRadius: frame.radius; seed: 11; paintBase: true }
            Rectangle { anchors.fill: parent; radius: frame.radius; color: "transparent"; border.width: root.expanded ? 0 : 1; border.color: Galaxy.edge }

            RowLayout {
                anchors.fill: parent; anchors.margins: root.expanded ? 0 : 14; spacing: 14

                // ---------------------------------------------------------------- sidebar
                GalaxyPanel {
                    visible: !root.expanded
                    Layout.preferredWidth: root.narrow ? 64 : 208; Layout.fillHeight: true
                    ColumnLayout {
                        anchors.fill: parent; anchors.margins: 12; spacing: 6
                        RowLayout {
                            Layout.fillWidth: true; Layout.topMargin: 6; Layout.bottomMargin: 14; spacing: 10
                            Rectangle { implicitWidth: 34; implicitHeight: 34; radius: 12; color: Galaxy.base; border.color: Galaxy.edge
                                Text { anchors.centerIn: parent; text: "✦"; color: Galaxy.purple; font.pixelSize: 18 } }
                            Column { visible: !root.narrow; Layout.fillWidth: true; spacing: 2
                                Text { text: "DEEPFAKE"; font.family: Galaxy.font; font.pixelSize: 15; color: Galaxy.ink; font.letterSpacing: 2 }
                                Text { text: "local video studio"; font.family: Galaxy.font; font.pixelSize: 10; color: Galaxy.label } }
                        }
                        Repeater {
                            model: [["Dashboard", "󰕮"], ["Camera", "󰄀"], ["Engines", "󰘚"], ["Characters", "󰀉"], ["Models", "󰆼"], ["Recording", "󰑋"], ["Performance", "󰓅"], ["Settings", "󰒓"]]
                            GalaxyNav { required property var modelData; text: modelData[0]; icon: modelData[1]; compact: root.narrow; current: root.page === modelData[0]; onClicked: root.page = modelData[0] }
                        }
                        Item { Layout.fillHeight: true }
                        GalaxyPill { visible: !root.narrow; Layout.alignment: Qt.AlignHCenter
                            state_: !root.connected ? "err" : root.active ? "ok" : root.running ? "busy" : "idle"
                            text: !root.connected ? "BACKEND OFFLINE" : root.active ? String(root.session.state || "live").toUpperCase() : root.running ? "MODEL JOB" : "CAMERA OFF" }
                        Text { visible: !root.narrow; Layout.fillWidth: true; horizontalAlignment: Text.AlignHCenter; text: "disclosed synthetic media"; font.family: Galaxy.font; font.pixelSize: 9; color: Galaxy.dim; Layout.bottomMargin: 6 }
                    }
                }

                // ---------------------------------------------------------------- main column
                ColumnLayout {
                    Layout.fillWidth: true; Layout.fillHeight: true; spacing: 12
                    RowLayout {
                        visible: !root.expanded; Layout.fillWidth: true; spacing: 10
                        Text {
                            Layout.fillWidth: true; font.family: Galaxy.font; font.pixelSize: 12; color: Galaxy.lavender; elide: Text.ElideRight
                            text: "✦  O R B I T  /  " + root.page.toUpperCase().split("").join(" ")
                            MouseArea { anchors.fill: parent; cursorShape: Qt.SizeAllCursor; onPressed: window.startSystemMove() }
                        }
                        GalaxyPill { visible: root.narrow; state_: root.active ? "ok" : "idle"; text: root.active ? "LIVE" : "OFF" }
                        GalaxyButton { text: root.active ? "■  Stop" : "▶  Start webcam"; primary: !root.active; danger: root.active
                            enabled: !actionProc.running && (root.active || (!!root.data.consent && root.canStartLive))
                            hint: !root.data.consent ? "Acknowledge the consent notice in Settings first" : ""
                            onClicked: root.active ? root.call("stop") : root.start(false, false) }
                        GalaxyButton { text: "󰖯"; font.family: Galaxy.icons; hint: "Expand preview (F11)"; onClicked: root.expanded = true }
                        GalaxyButton { text: "×"; hint: "Hide window · a running session keeps running"; onClicked: root.opened = false }
                    }

                    ScrollView {
                        id: scroller
                        Layout.fillWidth: true; Layout.fillHeight: true; clip: true; contentWidth: availableWidth
                        ScrollBar.vertical: ScrollBar { width: 6; contentItem: Rectangle { radius: 3; color: Galaxy.edge } }
                        ColumnLayout {
                            width: scroller.availableWidth; spacing: 12

                            // ---------------- Dashboard
                            ColumnLayout {
                                visible: root.page === "Dashboard" || root.expanded; Layout.fillWidth: true; spacing: 12
                                RowLayout { Layout.fillWidth: true; spacing: 12
                                    Preview { kind: "original"; visible: root.prefs.comparison === "original" || root.prefs.comparison === "side-by-side"
                                        Layout.fillWidth: true; Layout.preferredHeight: root.expanded ? window.height : Math.max(260, window.height * 0.44) }
                                    Preview { visible: root.prefs.comparison !== "original"
                                        Layout.fillWidth: true; Layout.preferredHeight: root.expanded ? window.height : Math.max(260, window.height * 0.44) }
                                }
                                RowLayout { visible: !root.expanded; Layout.fillWidth: true; spacing: 10
                                    GalaxyCombo { Layout.preferredWidth: 180; Layout.fillWidth: false; model: ["processed", "original", "side-by-side"]
                                        currentIndex: Math.max(0, model.indexOf(root.prefs.comparison || "processed")); onActivated: root.call("preferences", {values: {comparison: currentText}}) }
                                    GalaxyToggle { text: "Mirror preview"; checked: !!root.prefs.mirror; onToggled: root.call("preferences", {values: {mirror: checked}}) }
                                    Item { Layout.fillWidth: true }
                                    GalaxyButton { text: "Virtual camera"; enabled: !root.active && root.canStartLive && !!root.data.consent; hint: "Start and feed the v4l2loopback device"; onClicked: root.start(true, false) }
                                    GalaxyButton { text: "Backend log"; onClicked: root.call("log") }
                                }
                                GridLayout { visible: !root.expanded; Layout.fillWidth: true; columns: root.narrow ? 3 : 6; columnSpacing: 10; rowSpacing: 10
                                    GalaxyStat { label: "Camera"; value: root.live(root.session.source_fps); unit: " fps" }
                                    GalaxyStat { label: "Inference"; value: root.live(root.session.native_inference_fps); unit: " fps"; tint: Galaxy.purple }
                                    GalaxyStat { label: "Output"; value: root.live(root.session.output_fps); unit: " fps" }
                                    GalaxyStat { label: "Latency p50"; value: root.live(root.session.latency_ms); unit: " ms" }
                                    GalaxyStat { label: "Latency p95"; value: root.live(root.session.latency_p95_ms); unit: " ms" }
                                    GalaxyStat { label: "Dropped"; value: root.live(root.session.dropped_source_frames, 0); tint: Number(root.session.dropped_source_frames) > 0 ? Galaxy.warn : Galaxy.lavender }
                                }
                                RowLayout { visible: !root.expanded; Layout.fillWidth: true; spacing: 12
                                    GalaxyPanel { Layout.fillWidth: true; implicitHeight: engineCol.implicitHeight + 28
                                        ColumnLayout { id: engineCol; anchors.left: parent.left; anchors.right: parent.right; anchors.top: parent.top; anchors.margins: 14; spacing: 6
                                            GalaxySection { title: "Engine" }
                                            KV { k: "backend"; v: root.active ? String(root.session.backend || root.session.engine || "—") : String(root.form.engine || "—") + " (next start)" }
                                            KV { k: "frame gen"; v: root.active ? (root.session.framegen_enabled ? "RIFE " + (root.session.framegen_variant || "") + " · " + root.session.framegen_multiplier + "× · " + root.num(root.session.framegen_ms_per_frame) + " ms/frame" : "off" + (root.session.framegen_warning ? " · " + root.session.framegen_warning : "")) : String(root.form.frame_gen || "—") }
                                            KV { k: "source"; v: root.active ? String(root.session.input_device || "camera " + (root.form.camera || "0")) : "camera " + String(root.form.camera || "0") }
                                            KV { k: "preset"; v: String(root.form.preset || "—") }
                                        }
                                    }
                                    GalaxyPanel { Layout.fillWidth: true; implicitHeight: gpuCol.implicitHeight + 28
                                        ColumnLayout { id: gpuCol; anchors.left: parent.left; anchors.right: parent.right; anchors.top: parent.top; anchors.margins: 14; spacing: 6
                                            GalaxySection { title: "GPU" }
                                            KV { k: "device"; v: String(root.gpu.name || "—") }
                                            KV { k: "VRAM"; v: root.gpu.mem_used_mib !== undefined ? root.gpu.mem_used_mib + " / " + root.gpu.mem_total_mib + " MiB" : "—" }
                                            KV { k: "util · temp"; v: root.gpu.util_pct !== undefined ? root.gpu.util_pct + " % · " + root.gpu.temp_c + " °C · " + root.gpu.power_w + " W" : "—" }
                                            KV { k: "session VRAM"; v: root.active && root.session.vram_mb ? root.num(root.session.vram_mb, 0) + " MiB" : "—" }
                                        }
                                    }
                                }
                                RowLayout { visible: !root.expanded && root.active; Layout.fillWidth: true; spacing: 12
                                    GalaxySpark { label: "OUTPUT FPS"; data: root.fpsHist; current: root.num(root.session.output_fps) }
                                    GalaxySpark { label: "LATENCY MS"; data: root.latHist; stroke: Galaxy.magenta; current: root.num(root.session.latency_ms) }
                                }
                            }

                            // ---------------- Camera
                            GalaxyPanel {
                                visible: root.page === "Camera" && !root.expanded; Layout.fillWidth: true; implicitHeight: camCol.implicitHeight + 32
                                ColumnLayout { id: camCol; anchors.left: parent.left; anchors.right: parent.right; anchors.top: parent.top; anchors.margins: 16; spacing: 10
                                    GalaxySection { title: "Capture" }
                                    Hint { text: "The camera opens only when you press Start. Stop the session before changing capture or output settings." }
                                    Repeater { model: root.data.devices || []
                                        KV { required property var modelData; k: modelData.path; v: modelData.name + " · " + modelData.kind; tint: modelData.kind === "loopback" ? Galaxy.purple : Galaxy.ink } }
                                    Label { text: "Camera index or device" }
                                    GalaxyField { text: root.form.camera || "0"; enabled: !root.active; onEditingFinished: root.edit("camera", text) }
                                    RowLayout { Layout.fillWidth: true; spacing: 10
                                        GalaxyField { text: String(root.form.width || 640); placeholderText: "Width"; enabled: !root.active; validator: IntValidator { bottom: 64; top: 4096 } onEditingFinished: root.edit("width", Number(text)) }
                                        GalaxyField { text: String(root.form.height || 480); placeholderText: "Height"; enabled: !root.active; validator: IntValidator { bottom: 64; top: 4096 } onEditingFinished: root.edit("height", Number(text)) }
                                        GalaxyField { text: String(root.form.fps || 30); placeholderText: "FPS"; enabled: !root.active; validator: IntValidator { bottom: 1; top: 120 } onEditingFinished: root.edit("fps", Number(text)) }
                                    }
                                    Label { text: "Virtual camera device (empty = auto-detect)" }
                                    GalaxyField { text: root.form.output_device || ""; placeholderText: "/dev/video10"; enabled: !root.active; onEditingFinished: root.edit("output_device", text) }
                                    Hint { text: "Virtual output uses v4l2loopback. Pick that device in OBS or your browser." }
                                    GalaxyButton { text: "Save capture preferences"; primary: true; enabled: !root.active; onClicked: root.save() }
                                }
                            }

                            // ---------------- Engines
                            GalaxyPanel {
                                visible: (root.page === "Engines") && !root.expanded; Layout.fillWidth: true; implicitHeight: engCol.implicitHeight + 32
                                ColumnLayout { id: engCol; anchors.left: parent.left; anchors.right: parent.right; anchors.top: parent.top; anchors.margins: 16; spacing: 10
                                    GalaxySection { title: "Face engine" }
                                    RowLayout { Layout.fillWidth: true; spacing: 10
                                        GalaxyButton { text: "AlphaFace · normal"; Layout.fillWidth: true; selected: root.form.engine === "alphaface"; enabled: !root.active; onClicked: root.edit("engine", "alphaface") }
                                        GalaxyButton { text: "Experimental · neural character"; Layout.fillWidth: true; selected: root.form.engine === "experimental"; enabled: !root.active; onClicked: root.edit("engine", "experimental") }
                                    }
                                    RowLayout { visible: root.form.engine === "experimental"; Layout.fillWidth: true; spacing: 10
                                        GalaxyCombo { model: ["head", "fullbody"]; enabled: !root.active; currentIndex: Math.max(0, model.indexOf(root.form.mode)); onActivated: { root.edit("mode", currentText); root.edit("backend", currentText === "fullbody" ? "editalive" : "liveportrait") } }
                                        GalaxyCombo { model: root.form.mode === "fullbody" ? ["editalive"] : ["liveportrait", "personalive"]; enabled: !root.active; currentIndex: Math.max(0, model.indexOf(root.form.backend)); onActivated: root.edit("backend", currentText) }
                                    }
                                    Hint { visible: root.form.engine === "experimental"; text: "LivePortrait: experimental single-subject head animation. PersonaLive / EditaLive: offline adapters only, separate setup required." }
                                    Label { text: "Reference image" }
                                    GalaxyField { text: root.form.reference || ""; enabled: !root.active; placeholderText: "Choose a character or enter a local reference path"; onEditingFinished: root.edit("reference", text) }
                                    GridLayout { Layout.fillWidth: true; columns: root.narrow ? 1 : 3; columnSpacing: 10; rowSpacing: 6
                                        Hint { text: "Quality preset" } Hint { text: "Frame generation (RIFE)" } Item { visible: !root.narrow; implicitHeight: 1 }
                                        GalaxyCombo { model: ["latency", "balanced", "quality"]; enabled: !root.active; currentIndex: Math.max(0, model.indexOf(root.form.preset)); onActivated: root.edit("preset", currentText) }
                                        GalaxyCombo { model: ["auto", "off", "2x", "3x", "4x"]; enabled: !root.active; currentIndex: Math.max(0, model.indexOf(root.form.frame_gen)); onActivated: root.edit("frame_gen", currentText) }
                                        GalaxyButton { text: "Save"; primary: true; enabled: !root.active; onClicked: root.save() }
                                    }
                                    Hint { text: root.active ? "Engine and quality changes apply on the next start (stop the session first)." : "auto enables interpolation only when the camera rate is below the target and the RIFE weights verified." }
                                    Repeater { model: root.data.engines || []
                                        KV { required property var modelData; k: modelData.name; v: (modelData.status || "available") + (modelData.validation ? " · " + modelData.validation : ""); tint: modelData.available ? Galaxy.ok : Galaxy.muted } }
                                }
                            }

                            // ---------------- Characters
                            GalaxyPanel {
                                visible: root.page === "Characters" && !root.expanded; Layout.fillWidth: true; implicitHeight: chCol.implicitHeight + 32
                                ColumnLayout { id: chCol; anchors.left: parent.left; anchors.right: parent.right; anchors.top: parent.top; anchors.margins: 16; spacing: 10
                                    GalaxySection { title: "Character library" }
                                    Hint { text: "Images and provenance stay in your private local library. Nothing is uploaded." }
                                    RowLayout { Layout.fillWidth: true; spacing: 10; GalaxyField { id: importPath; placeholderText: "Reference image path" } GalaxyButton { text: "Browse…"; onClicked: imagePicker.open() } }
                                    GalaxyField { id: characterName; placeholderText: "Character name" }
                                    GalaxyField { id: provenance; placeholderText: "Consent / provenance: self, synthetic character, or permission details" }
                                    GalaxyButton { text: "Import character"; primary: true; onClicked: root.call("import_character", {path: importPath.text, name: characterName.text, provenance: provenance.text}) }
                                    Repeater { model: root.data.characters || []
                                        Rectangle { id: characterCard; required property var modelData; Layout.fillWidth: true; implicitHeight: 120; radius: Galaxy.radiusTile; color: Galaxy.tile
                                            border.color: root.form.reference === modelData.path ? Galaxy.purple : Galaxy.line
                                            RowLayout { anchors.fill: parent; anchors.margins: 12; spacing: 14
                                                Image { Layout.preferredWidth: 96; Layout.fillHeight: true; fillMode: Image.PreserveAspectFit; asynchronous: true; sourceSize.width: 192; source: "file://" + characterCard.modelData.path }
                                                ColumnLayout { Layout.fillWidth: true; Label { text: characterCard.modelData.name } Hint { text: characterCard.modelData.provenance } Hint { text: characterCard.modelData.width + "×" + characterCard.modelData.height } }
                                                GalaxyButton { text: "Use"; selected: root.form.reference === characterCard.modelData.path; onClicked: { root.edit("reference", characterCard.modelData.path); root.save() } }
                                                GalaxyButton { text: "Remove"; danger: true; onClicked: root.call("remove_character", {id: characterCard.modelData.id}) }
                                            }
                                        }
                                    }
                                }
                            }

                            // ---------------- Models
                            ColumnLayout {
                                visible: root.page === "Models" && !root.expanded; Layout.fillWidth: true; spacing: 10
                                GalaxyPanel { Layout.fillWidth: true; implicitHeight: jobCol.implicitHeight + 28
                                    ColumnLayout { id: jobCol; anchors.left: parent.left; anchors.right: parent.right; anchors.top: parent.top; anchors.margins: 14; spacing: 8
                                        RowLayout { Layout.fillWidth: true
                                            GalaxySection { title: "Model operations" }
                                            GalaxyPill { state_: root.running ? "busy" : root.job.state === "error" ? "err" : root.job.state === "complete" ? "ok" : "idle"
                                                text: (root.job.state || "idle").toUpperCase() + (root.job.action ? " · " + String(root.job.action).replace("_model", "") + " " + root.job.model : "") }
                                        }
                                        GalaxyProgress { visible: root.running; indeterminate: !root.job.total_bytes; value: root.job.total_bytes ? root.job.bytes / root.job.total_bytes : 0 }
                                        Hint { visible: root.running && !!root.job.file; text: (root.job.file || "") + (root.job.total_bytes ? " · " + (root.job.bytes / 1048576).toFixed(0) + " / " + (root.job.total_bytes / 1048576).toFixed(0) + " MiB" : "") }
                                        Hint { visible: root.job.state === "error"; color: Galaxy.err; text: String(root.job.error || "") }
                                        Hint { text: "Downloads are staged and checked (size, SHA-256, format, GPU load) before they replace anything. Weights are never committed or uploaded." }
                                        GalaxyButton { text: "Cancel operation"; visible: root.running; danger: true; onClicked: root.call("cancel_model") }
                                    }
                                }
                                Repeater { model: root.data.models || []
                                    Rectangle { id: modelCard; required property var modelData; Layout.fillWidth: true; implicitHeight: modelContents.implicitHeight + 26; radius: Galaxy.radiusTile; color: Galaxy.tile
                                        readonly property bool core: !modelData.experimental
                                        readonly property string integ: modelData.integrity || (modelData.installed ? "unverified" : "not installed")
                                        border.color: modelData.installed && integ === "verified" ? Galaxy.edge : Galaxy.line
                                        ColumnLayout { id: modelContents; anchors.left: parent.left; anchors.right: parent.right; anchors.top: parent.top; anchors.margins: 13; spacing: 6
                                            RowLayout { Layout.fillWidth: true; spacing: 8
                                                Text { text: modelCard.modelData.name; font.family: Galaxy.font; font.pixelSize: 14; color: Galaxy.ink }
                                                GalaxyPill { state_: !modelCard.modelData.installed ? "idle" : modelCard.integ === "verified" ? (modelCard.modelData.load_check_ok === false ? "err" : "ok") : "warn"
                                                    text: !modelCard.modelData.installed ? "AVAILABLE" : modelCard.integ === "verified" ? (modelCard.modelData.load_check_ok === true ? "VERIFIED · GPU LOAD OK" : modelCard.modelData.load_check_ok === false ? "LOAD CHECK FAILED" : "VERIFIED") : "UNVERIFIED" }
                                                Text { visible: !!modelCard.modelData.experimental; text: "experimental"; font.family: Galaxy.font; font.pixelSize: 10; color: Galaxy.label }
                                                Item { Layout.fillWidth: true }
                                                Text { text: modelCard.modelData.size_hint || ""; font.family: Galaxy.mono; font.pixelSize: 10; color: Galaxy.muted }
                                            }
                                            Hint { text: "Code: " + modelCard.modelData.code_license }
                                            Hint { text: "Weights: " + modelCard.modelData.weights_license; color: /NON-COMMERCIAL|UNDOCUMENTED|NC/.test(String(modelCard.modelData.weights_license)) ? Galaxy.warn : Galaxy.muted }
                                            Hint { visible: !!(modelCard.modelData.blocked || modelCard.modelData.notes); text: modelCard.modelData.blocked || modelCard.modelData.notes || ""; color: modelCard.modelData.blocked ? Galaxy.err : Galaxy.dim }
                                            RowLayout { spacing: 8
                                                GalaxyButton { text: "Install"; visible: !modelCard.modelData.installed; primary: true; enabled: !modelCard.modelData.blocked && !root.running; onClicked: root.confirmModel("install_model", modelCard.modelData) }
                                                GalaxyButton { text: "Verify"; enabled: modelCard.modelData.installed && !root.running; hint: modelCard.core ? "Re-hash every file and load it on the GPU" : "Re-hash every file"; onClicked: root.call("verify_model", {model: modelCard.modelData.name}) }
                                                GalaxyButton { text: "Repair"; visible: modelCard.core; enabled: !root.running; hint: "Staged redownload; old files stay until the new ones pass"; onClicked: root.confirmModel("reinstall_model", modelCard.modelData) }
                                                GalaxyButton { text: "Remove"; danger: true; enabled: modelCard.modelData.installed && !root.running; onClicked: root.confirmModel("remove_model", modelCard.modelData) }
                                            }
                                        }
                                    }
                                }
                            }

                            // ---------------- Recording
                            GalaxyPanel {
                                visible: root.page === "Recording" && !root.expanded; Layout.fillWidth: true; implicitHeight: recCol.implicitHeight + 32
                                ColumnLayout { id: recCol; anchors.left: parent.left; anchors.right: parent.right; anchors.top: parent.top; anchors.margins: 16; spacing: 10
                                    GalaxySection { title: "Recording & offline render" }
                                    Label { text: "Offline video input (optional)" }
                                    RowLayout { Layout.fillWidth: true; spacing: 10; GalaxyField { id: videoPath; text: root.form.input_video || ""; placeholderText: "Input video path"; onEditingFinished: root.edit("input_video", text) } GalaxyButton { text: "Browse…"; onClicked: videoPicker.open() } }
                                    Label { text: "Output file" }
                                    RowLayout { Layout.fillWidth: true; spacing: 10; GalaxyField { id: outputPath; text: root.form.recording_path || ""; placeholderText: "/path/to/new-recording.mp4"; onEditingFinished: root.edit("recording_path", text) } GalaxyButton { text: "Browse…"; onClicked: outputPicker.open() } }
                                    GalaxyCombo { Layout.preferredWidth: 220; Layout.fillWidth: false; model: ["auto", "libx264", "h264_nvenc"]; currentIndex: Math.max(0, model.indexOf(root.form.encoder)); onActivated: root.edit("encoder", currentText) }
                                    Hint { text: "Recording starts a new webcam session; Stop finalises the file. Live audio is not captured. Existing files are never overwritten. The disclosure stays on every recording." }
                                    RowLayout { spacing: 10
                                        GalaxyButton { text: "● Start recording"; primary: true; enabled: !root.active && root.canStartLive && !!root.form.recording_path && !!root.data.consent; onClicked: root.start(false, true) }
                                        GalaxyButton { text: "Stop and save"; enabled: root.active; onClicked: root.call("stop") }
                                        GalaxyButton { text: "Render video with selected engine"; enabled: !root.active && !!root.form.input_video && !!root.form.recording_path && !!root.data.consent; onClicked: { root.pendingRender = true; root.save() } }
                                    }
                                    Hint { text: root.session.output_video ? "Recording: " + root.session.output_video : "No recording reported" }
                                }
                            }

                            // ---------------- Performance
                            ColumnLayout {
                                visible: root.page === "Performance" && !root.expanded; Layout.fillWidth: true; spacing: 12
                                GridLayout { Layout.fillWidth: true; columns: root.narrow ? 2 : 4; columnSpacing: 10; rowSpacing: 10
                                    GalaxyStat { label: "GPU util"; value: root.num(root.gpu.util_pct, 0); unit: " %" }
                                    GalaxyStat { label: "VRAM used"; value: root.num(root.gpu.mem_used_mib, 0); unit: " MiB" }
                                    GalaxyStat { label: "GPU temp"; value: root.num(root.gpu.temp_c, 0); unit: " °C" }
                                    GalaxyStat { label: "GPU power"; value: root.num(root.gpu.power_w, 0); unit: " W" }
                                    GalaxyStat { label: "Queue depth"; value: root.live(root.session.queue_depth, 0) }
                                    GalaxyStat { label: "Generated frames"; value: root.live(root.session.generated_frames, 0) }
                                    GalaxyStat { label: "FG ms / frame"; value: root.live(root.session.framegen_ms_per_frame); unit: " ms" }
                                    GalaxyStat { label: "Dropped source"; value: root.live(root.session.dropped_source_frames, 0) }
                                }
                                GalaxyPanel { Layout.fillWidth: true; implicitHeight: 170
                                    ColumnLayout { anchors.fill: parent; anchors.margins: 14; spacing: 10
                                        GalaxySpark { label: "OUTPUT FPS · LAST 60 S"; data: root.fpsHist; current: root.live(root.session.output_fps); Layout.fillHeight: true }
                                        GalaxySpark { label: "LATENCY MS"; data: root.latHist; stroke: Galaxy.magenta; current: root.live(root.session.latency_ms); Layout.fillHeight: true }
                                    }
                                }
                                Hint { text: "Interpolated frames never count as native inference. Metrics the backend did not report show —." }
                            }

                            // ---------------- Settings
                            GalaxyPanel {
                                visible: root.page === "Settings" && !root.expanded; Layout.fillWidth: true; implicitHeight: setCol.implicitHeight + 32
                                ColumnLayout { id: setCol; anchors.left: parent.left; anchors.right: parent.right; anchors.top: parent.top; anchors.margins: 16; spacing: 10
                                    GalaxySection { title: "Consent & disclosure" }
                                    Label { text: root.data.consent_text || "Loading notice…"; color: Galaxy.muted }
                                    GalaxyButton { text: root.data.consent ? "Consent notice acknowledged" : "I own or have permission to use my references and footage"; primary: !root.data.consent; enabled: !root.data.consent; onClicked: root.call("consent", {accepted: true}) }
                                    GalaxySection { title: "Shortcuts"; Layout.topMargin: 10 }
                                    Hint { text: "Ctrl+S saves defaults · F11 expands the preview · Esc leaves it · Ctrl+Q stops the session and closes the window. Closing with × only hides it." }
                                    Hint { text: "The disclosure watermark stays enabled on processed output." }
                                }
                            }
                            Hint { visible: !root.data.consent && root.page !== "Settings" && !root.expanded; text: "Before starting, open Settings and acknowledge the consent and synthetic-media notice."; color: Galaxy.lavender }
                        }
                    }
                    Rectangle { visible: !root.expanded; Layout.fillWidth: true; implicitHeight: msg.implicitHeight + 16; radius: Galaxy.radiusControl; color: root.messageIsError ? Galaxy.errBg : Galaxy.surface; border.color: root.messageIsError ? Galaxy.err : Galaxy.line
                        Text { id: msg; anchors.fill: parent; anchors.margins: 8; text: root.message; color: root.messageIsError ? Galaxy.err : Galaxy.muted; font.family: Galaxy.font; font.pixelSize: 11; wrapMode: Text.Wrap; maximumLineCount: 4; elide: Text.ElideRight }
                    }
                }
            }
            Rectangle {
                visible: !root.expanded
                anchors.right: parent.right; anchors.bottom: parent.bottom; anchors.margins: 4; width: 18; height: 18; color: "transparent"
                Text { anchors.centerIn: parent; text: "◢"; color: grip.containsMouse ? Galaxy.lavender : Galaxy.edge; font.pixelSize: 13 }
                MouseArea { id: grip; anchors.fill: parent; hoverEnabled: true; cursorShape: Qt.SizeFDiagCursor; onPressed: { const w = window.contentItem.Window.window; if (w) w.startSystemResize(Qt.RightEdge | Qt.BottomEdge) } }
            }
        }
    }
}
