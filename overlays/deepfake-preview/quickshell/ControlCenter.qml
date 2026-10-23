import QtQuick
import QtQuick.Layouts
import QtQuick.Window
import QtQuick.Controls
import QtQuick.Dialogs
import Quickshell
import Quickshell.Io

ShellRoot {
    id: root
    property bool opened: true
    property bool expanded: false
    onExpandedChanged: {
        const nativeWindow = window.contentItem.Window.window
        if (nativeWindow) nativeWindow.visibility = expanded ? Window.FullScreen : Window.Windowed
    }
    property var data: ({session: {}, preferences: {}, models: [], characters: [], devices: [], engines: []})
    property var session: data.session || ({})
    property var prefs: data.preferences || ({})
    property var form: ({})
    property bool initialized: false
    property string page: "Dashboard"
    property string message: "Connecting…"
    property int serial: 0
    property var history: []
    property var pendingStart: null
    property bool pendingRender: false
    property string modelName: ""
    property string modelAction: ""
    property string confirmText: ""
    readonly property bool active: !!session.active
    readonly property bool canStartLive: root.form.engine !== "experimental" || (root.form.mode === "head" && root.form.backend === "liveportrait" && (root.data.engines || []).some(e => e.name === "liveportrait" && e.available))
    readonly property string cli: Quickshell.env("DEEPFAKE_CLI") || "deepfake"
    readonly property color bg: prefs.theme === "light" ? "#f2f4f8" : "#101318"
    readonly property color panel: prefs.theme === "light" ? "#ffffff" : "#1b212a"
    readonly property color ink: prefs.theme === "light" ? "#172030" : "#edf1f7"
    readonly property color muted: prefs.theme === "light" ? "#536176" : "#a5b0c1"
    readonly property color line: prefs.theme === "light" ? "#d4dbe5" : "#354052"
    readonly property color accent: prefs.theme === "matrix" ? "#68ed92" : "#91b6ff"
    function metric(v, suffix) { return v === null || v === undefined ? "—" : Number(v).toFixed(1) + (suffix || "") }
    function edit(key, value) { const f = Object.assign({}, form); f[key] = value; form = f }
    function call(action, extra) {
        if (actionProc.running) { message = "Please wait for the current operation"; return }
        actionProc.command = [cli, "desktop", "request", JSON.stringify(Object.assign({protocol_version: 1, action: action}, extra || {}))]
        actionProc.running = true
    }
    function save() { call("preferences", {values: form}) }
    function start(virtual, recording) { pendingStart = {virtual: virtual, recording: recording} save() }
    function path(url) { return decodeURIComponent(String(url).replace(/^file:\/\//, "")) }
    Process {
        id: actionProc
        stdout: StdioCollector { onStreamFinished: {
            try { const r = JSON.parse(text); root.message = r.error || r.message || "Done";
                if (root.pendingRender && r.ok) { root.pendingRender = false; Qt.callLater(() => root.call("render_video")) }
                else if (root.pendingStart && r.ok) { const p = root.pendingStart; root.pendingStart = null; Qt.callLater(() => root.call("start", p)) }
                else root.pendingStart = null
                if (!r.ok) root.pendingRender = false
            } catch(e) { root.message = text.slice(-500); root.pendingStart = null }
        } }
        stderr: StdioCollector { onStreamFinished: if (text.trim()) root.message = text.trim().slice(-800) }
    }
    Process {
        id: reader
        command: [root.cli, "desktop", "request", '{"protocol_version":1,"action":"snapshot"}']
        stdout: StdioCollector { onStreamFinished: {
            try { const d = JSON.parse(text); if (!d.ok) { root.message = d.error; return } root.data = d
                if (!root.initialized) { root.form = d.preferences; root.initialized = true; root.message = "Ready · processing stays on this computer" }
                if (root.active && d.session.native_inference_fps !== undefined) { const h = root.history.slice(-89); h.push(d.session.native_inference_fps); root.history = h }
            } catch(e) { root.message = "Backend unavailable. Install this checkout's Python package before launching." }
        } }
        stderr: StdioCollector { onStreamFinished: if (text.trim()) root.message = text.trim().slice(-800) }
    }
    Timer { interval: 1500; repeat: true; running: root.opened; triggeredOnStart: true; onTriggered: if (!reader.running) reader.running = true }
    Timer { interval: 100; repeat: true; running: root.opened && root.active; onTriggered: root.serial++ }
    IpcHandler { target: "deepfake"; function show(): void { root.opened = true } function toggle(): void { root.opened = !root.opened } }
    FileDialog { id: imagePicker; nameFilters: ["Reference images (*.png *.jpg *.jpeg *.webp)"]; onAccepted: importPath.text = root.path(selectedFile) }
    FileDialog { id: outputPicker; fileMode: FileDialog.SaveFile; nameFilters: ["MP4 video (*.mp4)"]; onAccepted: { outputPath.text = root.path(selectedFile); root.edit("recording_path", outputPath.text) } }
    FileDialog { id: videoPicker; nameFilters: ["Video (*.mp4 *.mkv *.mov *.avi *.webm)"]; onAccepted: { videoPath.text = root.path(selectedFile); root.edit("input_video", videoPath.text) } }
    Dialog {
        id: confirmation; parent: window.contentItem; anchors.centerIn: parent; modal: true; width: Math.min(620, window.width - 60)
        title: "Review model operation"; standardButtons: Dialog.Ok | Dialog.Cancel
        contentItem: Text { text: root.confirmText; wrapMode: Text.Wrap; color: root.ink }
        background: Rectangle { color: root.panel; border.color: root.line; radius: 8 }
        onAccepted: root.call(root.modelAction, {model: root.modelName, confirmed: true})
    }
    component Label: Text { color: root.ink; font.pixelSize: 14; textFormat: Text.PlainText; wrapMode: Text.Wrap; Layout.fillWidth: true }
    component Hint: Label { color: root.muted; font.pixelSize: 12 }
    component Action: Button {
        padding: 12; palette.buttonText: root.ink
        background: Rectangle { radius: 7; color: parent.down ? root.line : root.panel; border.color: parent.activeFocus ? root.accent : root.line }
        contentItem: Text { text: parent.text; color: parent.enabled ? root.ink : root.muted; horizontalAlignment: Text.AlignHCenter; verticalAlignment: Text.AlignVCenter; font.pixelSize: 13 }
    }
    component Field: TextField {
        color: root.ink; selectByMouse: true; padding: 11; placeholderTextColor: root.muted; Layout.fillWidth: true
        background: Rectangle { radius: 6; color: root.bg; border.color: parent.activeFocus ? root.accent : root.line }
    }
    component Choice: ComboBox { Layout.fillWidth: true; palette.button: root.panel; palette.buttonText: root.ink; palette.text: root.ink; palette.base: root.panel; palette.highlight: root.accent }
    component Card: Rectangle { color: root.panel; radius: 8; border.color: root.line; Layout.fillWidth: true }
    component Stat: Card {
        property string label: ""; property var value; property string suffix: ""
        implicitHeight: 82
        Column { anchors.fill: parent; anchors.margins: 14; spacing: 8
            Text { text: parent.parent.label; color: root.muted; font.pixelSize: 12 }
            Text { text: root.active ? root.metric(parent.parent.value, parent.parent.suffix) : "—"; color: root.ink; font.pixelSize: 22 }
        }
    }
    component Preview: Rectangle {
        property string kind: "processed"
        color: "#090c10"; radius: 8; clip: true
        Image { anchors.fill: parent; fillMode: Image.PreserveAspectFit; cache: false; asynchronous: true; mirror: !!root.prefs.mirror
            source: root.active ? "file://" + root.data.state_dir + "/" + (parent.kind === "original" ? "original.png" : "preview.png") + "?v=" + root.serial : "" }
        Text { anchors.centerIn: parent; visible: !root.active; text: "Camera is off"; color: "#9aa7ba"; font.pixelSize: 18 }
        Text { x: 12; y: 12; text: parent.kind === "original" ? "ORIGINAL" : "PROCESSED · SYNTHETIC MEDIA"; color: "white"; font.pixelSize: 10; style: Text.Outline; styleColor: "black" }
    }
    FloatingWindow {
        id: window; title: "Deepfake"; visible: root.opened; color: root.bg
        implicitWidth: 1180; implicitHeight: 820; minimumSize: Qt.size(780, 580)
        Timer { interval: 2500; running: !!Quickshell.env("DEEPFAKE_UI_SCREENSHOT"); onTriggered: appContent.grabToImage(function(result) { result.saveToFile(Quickshell.env("DEEPFAKE_UI_SCREENSHOT")) }) }
        onClosed: { if (root.active) root.call("stop"); root.opened = false }
        Shortcut { sequence: "Ctrl+Q"; onActivated: { if (root.active) root.call("stop"); root.opened = false } }
        Shortcut { sequence: "Ctrl+S"; onActivated: root.save() }
        Shortcut { sequence: "F11"; onActivated: root.expanded = !root.expanded }
        Shortcut { sequence: "Escape"; onActivated: root.expanded = false }
        RowLayout {
            id: appContent
            anchors.fill: parent; spacing: 0
            Rectangle {
                visible: !root.expanded; Layout.preferredWidth: window.width < 1000 ? 150 : 190; Layout.fillHeight: true; color: root.panel
                ColumnLayout { anchors.fill: parent; anchors.margins: 16; spacing: 8
                    Text { text: "DEEPFAKE"; color: root.ink; font.pixelSize: 20; font.bold: true; Layout.topMargin: 12 }
                    Hint { text: "LOCAL VIDEO STUDIO"; font.pixelSize: 10; Layout.bottomMargin: 25 }
                    Repeater { model: ["Dashboard", "Camera", "Engines", "Characters", "Models", "Recording", "Performance", "Settings"]
                        Action { required property string modelData; text: (root.page === modelData ? "›  " : "") + modelData; Layout.fillWidth: true; onClicked: root.page = modelData }
                    }
                    Item { Layout.fillHeight: true }
                    Hint { text: root.active ? "● " + (root.session.state || "Running") : "○ Camera off"; color: root.active ? root.accent : root.muted }
                    Hint { text: "Disclosed synthetic media" }
                }
            }
            ColumnLayout {
                Layout.fillWidth: true; Layout.fillHeight: true; Layout.margins: 24; spacing: 14
                RowLayout { visible: !root.expanded; Layout.fillWidth: true
                    Label { text: root.page; font.pixelSize: 26; font.weight: Font.DemiBold }
                    Action { text: root.active ? "Stop session" : "Start webcam"; enabled: !actionProc.running && (root.active || (!!root.data.consent && root.canStartLive)); onClicked: root.active ? root.call("stop") : root.start(false, false) }
                }
                ScrollView {
                    Layout.fillWidth: true; Layout.fillHeight: true; clip: true; contentWidth: availableWidth
                    ColumnLayout {
                        width: parent.width; spacing: 16
                        ColumnLayout {
                            visible: root.page === "Dashboard" || root.expanded; Layout.fillWidth: true; spacing: 14
                            RowLayout { Layout.fillWidth: true
                                Preview { kind: "original"; visible: root.prefs.comparison === "original" || root.prefs.comparison === "side-by-side"; Layout.fillWidth: true; Layout.preferredHeight: Math.max(250, window.height * 0.4) }
                                Preview { visible: root.prefs.comparison !== "original"; Layout.fillWidth: true; Layout.preferredHeight: Math.max(250, window.height * 0.4) }
                            }
                            RowLayout { Layout.fillWidth: true
                                Choice { model: ["processed", "original", "side-by-side"]; currentIndex: Math.max(0, model.indexOf(root.prefs.comparison || "processed")); onActivated: root.call("preferences", {values: {comparison: currentText}}) }
                                Action { text: "Mirror"; onClicked: root.call("preferences", {values: {mirror: !root.prefs.mirror}}) }
                                Action { text: root.expanded ? "Exit preview (Esc)" : "Expand preview (F11)"; onClicked: root.expanded = !root.expanded }
                            }
                            RowLayout { visible: !root.expanded; Layout.fillWidth: true
                                Stat { label: "Camera"; value: root.session.source_fps; suffix: " fps" }
                                Stat { label: "Native inference"; value: root.session.native_inference_fps; suffix: " fps" }
                                Stat { label: "Output"; value: root.session.output_fps; suffix: " fps" }
                                Stat { label: "Median latency"; value: root.session.latency_ms; suffix: " ms" }
                            }
                            Hint { text: "Backend: " + (root.session.backend || "auto") + " · Resolution: " + (root.session.resolution || "—") }
                            RowLayout { visible: !root.expanded
                                Action { text: "Record webcam"; enabled: !root.active && root.canStartLive && !!root.form.recording_path && !!root.data.consent; onClicked: root.start(false, true) }
                                Action { text: "Start virtual camera"; enabled: !root.active && root.canStartLive && !!root.data.consent; onClicked: root.start(true, false) }
                                Action { text: "Backend log"; onClicked: root.call("log") }
                            }
                        }
                        ColumnLayout {
                            visible: root.page === "Camera" && !root.expanded; Layout.fillWidth: true; spacing: 12
                            Hint { text: "Camera opens only when you press Start. Stop before changing capture or output settings." }
                            Repeater { model: root.data.devices || []; Label { required property var modelData; text: modelData.path + " · " + modelData.name + " (" + modelData.kind + ")" } }
                            Label { text: "Camera index or device" }
                            Field { text: root.form.camera || "0"; onEditingFinished: root.edit("camera", text) }
                            RowLayout { Layout.fillWidth: true
                                Field { text: String(root.form.width || 640); placeholderText: "Width"; validator: IntValidator { bottom: 64; top: 4096 } onEditingFinished: root.edit("width", Number(text)) }
                                Field { text: String(root.form.height || 480); placeholderText: "Height"; validator: IntValidator { bottom: 64; top: 4096 } onEditingFinished: root.edit("height", Number(text)) }
                                Field { text: String(root.form.fps || 30); placeholderText: "FPS"; validator: IntValidator { bottom: 1; top: 120 } onEditingFinished: root.edit("fps", Number(text)) }
                            }
                            Label { text: "Virtual camera device (empty = auto-detect)" }
                            Field { text: root.form.output_device || ""; placeholderText: "/dev/video10"; onEditingFinished: root.edit("output_device", text) }
                            Hint { text: "Virtual output uses v4l2loopback. Select that device in OBS or your browser. Errors appear in Backend log." }
                            Action { text: "Save capture preferences"; onClicked: root.save() }
                        }
                        ColumnLayout {
                            visible: (root.page === "Engines" || root.page === "Dashboard") && !root.expanded; Layout.fillWidth: true; spacing: 12
                            RowLayout { Layout.fillWidth: true
                                Action { text: (root.form.engine === "alphaface" ? "● " : "") + "AlphaFace · Normal"; Layout.fillWidth: true; enabled: !root.active; onClicked: root.edit("engine", "alphaface") }
                                Action { text: (root.form.engine === "experimental" ? "● " : "") + "Experimental · Neural Character"; Layout.fillWidth: true; enabled: !root.active; onClicked: root.edit("engine", "experimental") }
                            }
                            RowLayout { visible: root.form.engine === "experimental"; Layout.fillWidth: true
                                Choice { model: ["head", "fullbody"]; enabled: !root.active; currentIndex: Math.max(0, model.indexOf(root.form.mode)); onActivated: { root.edit("mode", currentText); root.edit("backend", currentText === "fullbody" ? "editalive" : "liveportrait") } }
                                Choice { model: root.form.mode === "fullbody" ? ["editalive"] : ["liveportrait", "personalive"]; enabled: !root.active; currentIndex: Math.max(0, model.indexOf(root.form.backend)); onActivated: root.edit("backend", currentText) }
                            }
                            Hint { visible: root.form.engine === "experimental"; text: "LivePortrait: experimental single-subject head animation. PersonaLive / EditaLive: offline adapters only, separate setup required. Hybrid is unavailable." }
                            Label { text: "Reference image" }
                            Field { text: root.form.reference || ""; placeholderText: "Choose a character or enter a local reference path"; onEditingFinished: root.edit("reference", text) }
                            RowLayout { Layout.fillWidth: true
                                Choice { model: ["latency", "balanced", "quality"]; currentIndex: Math.max(0, model.indexOf(root.form.preset)); onActivated: root.edit("preset", currentText) }
                                Choice { model: ["auto", "off", "2x", "3x", "4x"]; currentIndex: Math.max(0, model.indexOf(root.form.frame_gen)); onActivated: root.edit("frame_gen", currentText) }
                                Action { text: "Save"; onClicked: root.save() }
                            }
                            Repeater { model: root.page === "Engines" ? (root.data.engines || []) : []
                                Hint { required property var modelData; text: modelData.name + " · " + (modelData.status || "normal face engine") + " · " + (modelData.validation || "") }
                            }
                        }
                        ColumnLayout {
                            visible: root.page === "Characters" && !root.expanded; Layout.fillWidth: true; spacing: 12
                            Hint { text: "Images and provenance stay in your private local library. No automatic uploads." }
                            RowLayout { Layout.fillWidth: true; Field { id: importPath; placeholderText: "Reference image path" } Action { text: "Browse…"; onClicked: imagePicker.open() } }
                            Field { id: characterName; placeholderText: "Character name" }
                            Field { id: provenance; placeholderText: "Consent/provenance: self, synthetic character, or permission details" }
                            Action { text: "Import character"; onClicked: root.call("import_character", {path: importPath.text, name: characterName.text, provenance: provenance.text}) }
                            Repeater { model: root.data.characters || []
                                Card { id: characterCard; required property var modelData; implicitHeight: 130
                                    RowLayout { anchors.fill: parent; anchors.margins: 12; spacing: 15
                                        Image { Layout.preferredWidth: 100; Layout.fillHeight: true; fillMode: Image.PreserveAspectFit; source: "file://" + characterCard.modelData.path }
                                        ColumnLayout { Layout.fillWidth: true; Label { text: characterCard.modelData.name } Hint { text: characterCard.modelData.provenance } Hint { text: characterCard.modelData.width + "×" + characterCard.modelData.height } }
                                        Action { text: "Use"; onClicked: { root.edit("reference", characterCard.modelData.path); root.save() } }
                                        Action { text: "Remove"; onClicked: root.call("remove_character", {id: characterCard.modelData.id}) }
                                    }
                                }
                            }
                        }
                        ColumnLayout {
                            visible: root.page === "Models" && !root.expanded; Layout.fillWidth: true; spacing: 14
                            Hint { text: "Review size and terms before downloading. Weights and worker environments are installed separately." }
                            Label { text: "Model operation: " + ((root.data.model_job || {}).state || "idle") + " " + ((root.data.model_job || {}).error || "") }
                            ProgressBar { Layout.fillWidth: true; visible: (root.data.model_job || {}).state === "running"; indeterminate: !(root.data.model_job || {}).total_bytes; value: (root.data.model_job || {}).total_bytes ? root.data.model_job.bytes / root.data.model_job.total_bytes : 0 }
                            Action { text: "Cancel model operation"; visible: (root.data.model_job || {}).state === "running"; onClicked: root.call("cancel_model") }
                            Repeater { model: root.data.models || []
                                Card { id: modelCard; required property var modelData; implicitHeight: modelContents.implicitHeight + 28
                                    ColumnLayout { id: modelContents; anchors.left: parent.left; anchors.right: parent.right; anchors.margins: 14; y: 14; spacing: 8
                                        Label { text: modelCard.modelData.name + (modelCard.modelData.installed ? " · Installed" : " · Available"); font.bold: true }
                                        Hint { text: (modelCard.modelData.size_hint || "Size unknown") + " · Code: " + modelCard.modelData.code_license + " · Weights: " + modelCard.modelData.weights_license }
                                        Hint { text: modelCard.modelData.blocked || modelCard.modelData.notes || "" }
                                        RowLayout {
                                            Action { text: "Install"; enabled: !modelCard.modelData.blocked; onClicked: { root.modelName = modelCard.modelData.name; root.modelAction = "install_model"; root.confirmText = "Download " + modelCard.modelData.size_hint + " for " + root.modelName + "?\n\nCode: " + modelCard.modelData.code_license + "\nWeights: " + modelCard.modelData.weights_license; confirmation.open() } }
                                            Action { text: "Verify"; enabled: modelCard.modelData.installed; onClicked: root.call("verify_model", {model: modelCard.modelData.name}) }
                                            Action { text: "Remove"; enabled: modelCard.modelData.installed; onClicked: { root.modelName = modelCard.modelData.name; root.modelAction = "remove_model"; root.confirmText = "Remove downloaded " + root.modelName + " weights from the model cache?"; confirmation.open() } }
                                        }
                                    }
                                }
                            }
                        }
                        ColumnLayout {
                            visible: root.page === "Recording" && !root.expanded; Layout.fillWidth: true; spacing: 14
                            Label { text: "Offline video input (optional)" }
                            RowLayout { Layout.fillWidth: true; Field { id: videoPath; text: root.form.input_video || ""; placeholderText: "Input video path"; onEditingFinished: root.edit("input_video", text) } Action { text: "Browse…"; onClicked: videoPicker.open() } }
                            Label { text: "Output file" }
                            RowLayout { Layout.fillWidth: true; Field { id: outputPath; text: root.form.recording_path || ""; placeholderText: "/path/to/new-recording.mp4"; onEditingFinished: root.edit("recording_path", text) } Action { text: "Browse…"; onClicked: outputPicker.open() } }
                            Choice { model: ["auto", "libx264", "h264_nvenc"]; currentIndex: Math.max(0, model.indexOf(root.form.encoder)); onActivated: root.edit("encoder", currentText) }
                            Hint { text: "Recording starts a new webcam session with the selected capture/output settings. Stop finalizes the file. Live audio is not captured. Existing files are protected." }
                            RowLayout { Action { text: "Start recording"; enabled: !root.active && root.canStartLive && !!root.form.recording_path && !!root.data.consent; onClicked: root.start(false, true) } Action { text: "Stop and save"; enabled: root.active; onClicked: root.call("stop") } }
                            Action { text: "Render video with selected engine"; enabled: !root.active && !!root.form.input_video && !!root.form.recording_path && !!root.data.consent; onClicked: { root.pendingRender = true; root.save() } }
                            Hint { text: root.session.output_video ? "Recording: " + root.session.output_video : "No recording reported" }
                        }
                        ColumnLayout {
                            visible: root.page === "Performance" && !root.expanded; Layout.fillWidth: true; spacing: 14
                            RowLayout { Layout.fillWidth: true; Stat { label: "GPU"; value: root.session.gpu_util_pct; suffix: "%" } Stat { label: "VRAM"; value: root.session.vram_mb; suffix: " MiB" } Stat { label: "Queue"; value: root.session.queue_depth } }
                            RowLayout { Layout.fillWidth: true; Stat { label: "P95 latency"; value: root.session.latency_p95_ms; suffix: " ms" } Stat { label: "Generated frames"; value: root.session.generated_frames } Stat { label: "Dropped source"; value: root.session.dropped_source_frames } }
                            Card { implicitHeight: 180
                                Canvas { anchors.fill: parent; anchors.margins: 20; property var points: root.history; onPointsChanged: requestPaint()
                                    onPaint: { const c = getContext("2d"); c.reset(); if (points.length < 2) return; const max = Math.max(1, ...points); c.strokeStyle = root.accent; c.lineWidth = 2; c.beginPath(); for (let i = 0; i < points.length; ++i) { const x = i / 89 * width, y = height - points[i] / max * height; if (i === 0) c.moveTo(x, y); else c.lineTo(x, y) } c.stroke() }
                                }
                            }
                            Hint { text: "Native inference FPS · recent samples. Interpolated frames never count as native inference. Unavailable metrics display —." }
                        }
                        ColumnLayout {
                            visible: root.page === "Settings" && !root.expanded; Layout.fillWidth: true; spacing: 14
                            Label { text: "Theme" }
                            Choice { model: ["dark", "light", "matrix"]; currentIndex: Math.max(0, model.indexOf(root.prefs.theme)); onActivated: root.call("preferences", {values: {theme: currentText}}) }
                            Hint { text: "Ctrl+S saves defaults · F11 expands preview · Esc exits expanded preview · Ctrl+Q stops the session and closes the window." }
                            Hint { text: "Engine, character, camera, quality, frame generation, encoder and output preferences are saved through their pages. Disclosure stays enabled." }
                            Label { text: root.data.consent_text || "Loading notice…" }
                            Action { text: root.data.consent ? "Consent notice acknowledged" : "I own or have permission to use my references and footage"; enabled: !root.data.consent; onClicked: root.call("consent", {accepted: true}) }
                        }
                        Hint { visible: !root.data.consent && root.page !== "Settings"; text: "Before starting, open Settings and acknowledge the consent and synthetic-media notice."; color: root.accent }
                    }
                }
                Rectangle { Layout.fillWidth: true; height: 1; color: root.line }
                Text { visible: !root.expanded; text: root.message; color: root.muted; font.pixelSize: 12; wrapMode: Text.Wrap; Layout.fillWidth: true; maximumLineCount: 5; elide: Text.ElideRight }
            }
        }
    }
}
