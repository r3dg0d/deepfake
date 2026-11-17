import QtQuick
import QtQuick.Layouts
import QtQuick.Window
import QtQuick.Controls
import QtQuick.Dialogs
import Quickshell
import Quickshell.Io

// Deepfake control center in the Galaxy theme of the rice (~/.config/quickshell/galaxy).
//
// This file is the shell: backend state, the request queue, readiness rules, dialogs, toasts, the
// window frame and the sidebar. Each page lives in its own <Name>Page.qml and receives `app` (this
// object). Everything shown comes from `deepfake desktop request snapshot`; a metric the backend did
// not report is drawn as "—", never estimated. Controls only call actions the backend implements.
ShellRoot {
    id: root

    // ------------------------------------------------------------------ window state
    property bool opened: true
    property bool expanded: false
    onExpandedChanged: {
        const nativeWindow = window.contentItem.Window.window
        if (nativeWindow) nativeWindow.visibility = expanded ? Window.FullScreen : Window.Windowed
    }
    readonly property bool narrow: window.width < 1020
    readonly property real windowHeight: window.height
    readonly property var pages: [["Studio", "󰕮"], ["Characters", "󰀉"], ["Models", "󰆼"], ["Render", "󰑋"], ["Performance", "󰓅"], ["Settings", "󰒓"]]
    property string page: "Studio"

    // ------------------------------------------------------------------ backend state
    property var data: ({session: {}, preferences: {}, models: [], characters: [], devices: [], engines: [], gpu: {}})
    property var session: data.session || ({})
    property var prefs: data.preferences || ({})
    property var gpu: data.gpu || ({})
    property var job: data.model_job || ({})
    // Lists that drive Repeaters are replaced only when their content changes, so delegates (and what the
    // user is typing in them) survive the snapshot polling.
    property var modelNames: []
    property var characters: []
    property var devices: []
    property var form: ({})    // editable copy of the preferences
    property var dirty: ({})   // edited keys not yet saved
    property bool initialized: false
    property bool connected: false
    property int serial: 0
    property var fpsHist: []
    property var latHist: []
    property var logs: ({backend: "", worker: "", model: ""})
    readonly property bool active: !!session.active
    readonly property bool jobRunning: job.state === "running"
    readonly property bool busy: actionProc.running || queue.length > 0
    readonly property string cli: Quickshell.env("DEEPFAKE_CLI") || "deepfake"

    // ------------------------------------------------------------------ helpers
    function num(v, digits) { return v === null || v === undefined || v === "" || isNaN(Number(v)) ? "—" : Number(v).toFixed(digits === undefined ? 1 : digits) }
    function live(v, digits) { return root.active ? num(v, digits) : "—" }
    function push(arr, v) { const a = arr.slice(-59); a.push(Number(v) || 0); return a }
    function dur(s) { s = Math.floor(Number(s) || 0); const h = Math.floor(s / 3600), m = Math.floor(s % 3600 / 60); return (h ? h + "h " : "") + m + "m " + String(s % 60).padStart(2, "0") + "s" }
    function gib(bytes) { return (Number(bytes || 0) / 1073741824).toFixed(Number(bytes || 0) < 10737418240 ? 2 : 1) + " GiB" }
    function path(url) { return decodeURIComponent(String(url).replace(/^file:\/\//, "")) }
    function basename(p) { return String(p || "").split("/").pop() }
    function model(name) { return (data.models || []).find(m => m.name === name) || ({}) }
    function engine(name) { return (data.engines || []).find(e => e.name === name) || ({}) }
    function stable(prop, value) { if (JSON.stringify(root[prop]) !== JSON.stringify(value)) root[prop] = value }
    function go(name) { page = name; expanded = false }

    // ------------------------------------------------------------------ preferences: edit, autosave
    function edit(key, value) {
        if (form[key] === value) return
        const f = Object.assign({}, form); f[key] = value; form = f
        const d = Object.assign({}, dirty); d[key] = value; dirty = d
        saveTimer.restart()
    }
    function flush() {
        saveTimer.stop()
        if (Object.keys(dirty).length === 0) return
        const values = dirty; dirty = ({})
        enqueue("preferences", {values: values}, {quiet: true})
    }
    Timer { id: saveTimer; interval: 350; onTriggered: root.flush() }

    // ------------------------------------------------------------------ request queue (one backend call at a time)
    property var queue: []
    property var current: null
    function enqueue(action, extra, options) {
        queue = queue.concat([{action: action, extra: extra || {}, options: options || {}}])
        pump()
    }
    function pump() {
        if (actionProc.running || queue.length === 0) return
        current = queue[0]; queue = queue.slice(1)
        actionProc.command = [cli, "desktop", "request", JSON.stringify(Object.assign({protocol_version: 1, action: current.action}, current.extra))]
        actionProc.running = true
    }
    function finished(text) {
        const request = current || ({action: "", options: {}})
        current = null
        let reply = null
        try { reply = JSON.parse(text) } catch (e) { reply = {ok: false, error: text.trim().slice(-500) || "The backend returned nothing"} }
        if (!reply.ok) {
            queue = []  // later requests were queued on the assumption that this one worked
            notify(reply.error || "Request failed", "error")
        } else {
            if (reply.log !== undefined) { const l = Object.assign({}, logs); l[reply.which || "backend"] = reply.log; logs = l }
            if (reply.preferences && Object.keys(dirty).length === 0) form = reply.preferences
            if (!request.options.quiet && reply.message) notify(reply.message, "ok")
        }
        if (!reader.running) reader.running = true  // show the effect now, not on the next poll
        Qt.callLater(root.pump)
    }
    Process {
        id: actionProc
        stdout: StdioCollector { onStreamFinished: root.finished(text) }
        stderr: StdioCollector { onStreamFinished: if (text.trim()) root.notify(text.trim().slice(-600), "error") }
        onRunningChanged: if (!running) Qt.callLater(root.pump)
    }

    // ------------------------------------------------------------------ snapshot polling
    Process {
        id: reader
        command: [root.cli, "desktop", "request", '{"protocol_version":1,"action":"snapshot"}']
        stdout: StdioCollector { onStreamFinished: root.snapshot(text) }
        stderr: StdioCollector { onStreamFinished: if (text.trim() && !root.connected) root.offline(text.trim().slice(-400)) }
    }
    property string offlineReason: ""
    function offline(reason) { connected = false; offlineReason = reason }
    function snapshot(text) {
        let d = null
        try { d = JSON.parse(text) } catch (e) { offline("Backend unavailable: is the deepfake package installed for " + cli + "?"); return }
        if (!d.ok) { offline(d.error || "Backend error"); return }
        const before = job.state, beforeLabel = jobLabel(job)
        data = d; connected = true; offlineReason = ""
        stable("modelNames", (d.models || []).map(m => m.name))
        stable("characters", d.characters || [])
        stable("devices", d.devices || [])
        if (!initialized) { form = d.preferences; initialized = true }
        if (active) { fpsHist = push(fpsHist, d.session.output_fps); latHist = push(latHist, d.session.latency_ms) }
        // report what finished while the window was open, once
        const j = d.model_job || ({})
        if (before === "running" && j.state !== "running") {
            if (j.state === "complete") notify(beforeLabel + " finished", "ok")
            else if (j.state === "error" || j.state === "interrupted") notify(beforeLabel + " failed: " + (j.error || j.state), "error")
        }
        const s = d.session || ({})
        if (s.state === "error" && s.error && seenError !== s.session_id) { seenError = s.session_id; notify(String(s.error).slice(0, 600), "error") }
    }
    property string seenError: ""
    function jobLabel(j) { return j && j.action ? String(j.action).replace("_model", "") + " " + j.model : "Model operation" }
    Timer { interval: root.active || root.jobRunning ? 1000 : 2000; repeat: true; running: root.opened; triggeredOnStart: true; onTriggered: if (!reader.running) reader.running = true }
    // preview refresh ~10 Hz only while visible and live: the widget never competes with inference
    Timer { interval: 100; repeat: true; running: root.opened && root.active && root.page === "Studio"; onTriggered: root.serial++ }
    // `open` is what launchers call: `qs ipc call deepfake show` never reaches show(), because the qs 0.3 CLI parses
    // "show" as its own `ipc show` subcommand (lists targets, exits 0). show() stays for older callers.
    IpcHandler { target: "deepfake"; function open(): void { root.opened = true } function show(): void { root.opened = true } function toggle(): void { root.opened = !root.opened } }

    // ------------------------------------------------------------------ engines and readiness
    readonly property var engineChoices: [
        {key: "alphaface", title: "Face swap", sub: "AlphaFace · live", engine: "alphaface", mode: "head", backend: "liveportrait"},
        {key: "liveportrait", title: "Head", sub: "LivePortrait · live · rotoscoped", engine: "experimental", mode: "head", backend: "liveportrait"},
        {key: "personalive", title: "Portrait", sub: "PersonaLive · file render", engine: "experimental", mode: "head", backend: "personalive"},
        {key: "editalive", title: "Full body", sub: "EditaLive · file render", engine: "experimental", mode: "fullbody", backend: "editalive"}]
    readonly property string engineKey: form.engine === "experimental" ? (form.mode === "fullbody" ? "editalive" : (form.backend === "personalive" ? "personalive" : "liveportrait")) : "alphaface"
    readonly property var engineChoice: engineChoices.find(c => c.key === engineKey) || engineChoices[0]
    readonly property bool offlineEngine: engineKey === "personalive" || engineKey === "editalive"
    readonly property bool headEngine: engineKey === "liveportrait"
    function chooseEngine(key) {
        const c = engineChoices.find(x => x.key === key)
        if (!c) return
        edit("engine", c.engine)
        if (c.engine === "experimental") { edit("mode", c.mode); edit("backend", c.backend) }
    }
    // {state: ok|warn|err|idle, text} for an engine option
    function engineState(key) {
        if (key === "alphaface") {
            if (model("alphaface").installed) return {state: "ok", text: "ready"}
            if (model("inswapper").installed) return {state: "warn", text: "inswapper fallback"}
            return {state: "err", text: "weights missing"}
        }
        const e = engine(key)
        if (!e.name) return {state: "idle", text: ""}
        if (e.disabled_reason) return {state: "err", text: "disabled"}
        if (!e.installed) return {state: "err", text: model(key).present_bytes > 0 ? "download incomplete" : "weights missing"}
        if (!e.configured) return {state: "warn", text: "needs setup"}
        if ((e.missing_dependencies || []).length) return {state: "warn", text: "missing packages"}
        return {state: "ok", text: e.live ? "ready" : "ready to render"}
    }
    // What has to be true before the selected engine can run. `blocking` items disable Start / Render.
    readonly property var checklist: {
        const items = []
        const consent = !!data.consent
        items.push({ok: consent, blocking: true, title: consent ? "Consent and disclosure notice acknowledged" : "Acknowledge the consent and disclosure notice",
                    detail: consent ? "" : "Required once before any camera or render starts.", label: "Review", run: () => root.go("Settings")})
        const hasRef = !!form.reference
        items.push({ok: hasRef, blocking: true, title: hasRef ? "Character: " + root.characterName(form.reference) : "Choose a character",
                    detail: hasRef ? "" : "A reference image you own or have permission to use.", label: "Choose", run: () => root.go("Characters")})
        const key = engineKey, st = engineState(key), e = engine(key)
        if (key === "alphaface") {
            items.push({ok: st.state !== "err", blocking: true, title: st.state === "err" ? "AlphaFace weights are not installed" : "AlphaFace weights installed",
                        detail: st.state === "warn" ? "Running on the non-commercial inswapper fallback." : "", label: "Open Models", run: () => root.go("Models")})
        } else {
            const m = model(key)
            items.push({ok: !!e.installed, blocking: true,
                        title: e.installed ? engineChoice.sub.split(" · ")[0] + " weights installed" : engineChoice.sub.split(" · ")[0] + " weights are not installed",
                        detail: e.installed ? "" : (m.present_bytes > 0 ? gib(m.present_bytes) + " of " + m.size_hint + " downloaded. Resume in Models." : (m.size_hint || "") + " download."),
                        label: "Open Models", run: () => root.go("Models")})
            items.push({ok: !!e.configured, blocking: true, title: e.configured ? "Source checkout and worker Python configured" : "Set up the source checkout" + (e.live ? "" : " and worker Python"),
                        detail: e.configured ? "" : "Pinned upstream revision " + String(e.revision || "").slice(0, 8) + ".", label: "Set up", run: () => root.go("Models")})
            if ((e.missing_dependencies || []).length)
                items.push({ok: false, blocking: true, title: "Python packages missing: " + e.missing_dependencies.join(", "), detail: "pip install -e '.[experimental-head]' plus a CUDA torch build.", label: "", run: null})
            if (e.disabled_reason)
                items.push({ok: false, blocking: true, title: "This backend is disabled", detail: e.disabled_reason, label: "Enable…", run: () => root.askEnable(key)})
        }
        if (model("yunet").name && !model("yunet").installed)
            items.push({ok: false, blocking: false, title: "Face detector (YuNet) is not installed", detail: "Needed on OpenCV 5, which has no Haar cascades.", label: "Open Models", run: () => root.go("Models")})
        if (headEngine && !data.matte_installed) {
            const needs = (form.background || "keep") !== "keep" || form.rotoscope === "on"
            items.push({ok: false, blocking: needs, title: needs ? "Background replacement needs the matte model" : "Rotoscoping is off: the matte model is not installed",
                        detail: "PP-HumanSeg, 6 MB, Apache-2.0. Without it the head is pasted as a soft rectangle.", label: "Install", run: () => root.confirmModel("install_model", root.model("pphumanseg"))})
        }
        return items
    }
    readonly property var blockers: checklist.filter(c => !c.ok && c.blocking)
    readonly property bool ready: connected && blockers.length === 0
    readonly property bool canStartLive: ready && !offlineEngine && !active
    function characterName(reference) {
        const c = characters.find(x => x.path === reference)
        return c ? c.name : basename(reference)
    }

    // ------------------------------------------------------------------ actions
    function start(virtual, recording) { flush(); enqueue("start", {virtual: !!virtual, recording: !!recording}) }
    function stop() { enqueue("stop") }
    function render() { flush(); enqueue("render_video") }
    function loadLog(which) { enqueue("log", {which: which}, {quiet: true}) }
    function confirmModel(action, m) {
        const size = m.size_hint || "size unknown"
        const licenses = "\n\nCode: " + m.code_license + "\nWeights: " + m.weights_license
        const texts = {
            remove_model: "Remove the downloaded " + m.name + " weights (" + size + ") from the model cache?",
            install_model: "Download " + m.name + " (" + size + ")?" + licenses + (m.present_bytes > 0 ? "\n\n" + gib(m.present_bytes) + " is already on disk; only the rest is downloaded." : ""),
            repair_model: "Re-check every " + m.name + " file and download only the missing or damaged ones (" + size + " in total)?" + licenses,
            reinstall_model: "Download all of " + m.name + " again (" + size + ")?" + licenses + "\n\nInstalled files stay in place until their replacements passed the checksum."}
        ask({title: action.replace("_model", "") + " " + m.name, text: texts[action], danger: action === "remove_model",
             proceed: {remove_model: "Remove", install_model: "Download", repair_model: "Repair", reinstall_model: "Download again"}[action],
             run: () => root.enqueue(action, {model: m.name, confirmed: true})})
    }
    function askEnable(key) {
        const e = engine(key), mem = e.memory || ({})
        ask({title: "enable " + key, proceed: "Enable",
             text: "This backend was disabled:\n" + (e.disabled_reason || "") +
                   "\n\nIt will run under the memory guard: the worker is stopped if it uses more than " + mem.limit_gb + " GiB of the " + mem.total_gb +
                   " GiB of host RAM, or if the host runs low (" + (mem.method || "watchdog") + "). A stopped worker ends the render; the desktop keeps running.",
             run: () => root.enqueue("enable_backend", {backend: key, confirmed: true})})
    }

    // ------------------------------------------------------------------ toasts
    property var toasts: []
    property int toastSeq: 0
    function notify(text, kind) {
        if (!text) return
        const id = ++toastSeq
        toasts = toasts.filter(t => t.text !== String(text)).concat([{id: id, text: String(text), kind: kind || "info"}]).slice(-4)
    }
    function dismiss(id) { toasts = toasts.filter(t => t.id !== id) }

    // ------------------------------------------------------------------ confirmation dialog
    property var question: ({title: "", text: "", proceed: "Proceed", danger: false, run: null})
    function ask(q) { question = Object.assign({title: "", text: "", proceed: "Proceed", danger: false, run: null}, q); confirmation.open() }
    Dialog {
        id: confirmation; parent: window.contentItem; anchors.centerIn: parent; modal: true; width: Math.min(560, window.width - 60)
        padding: 22
        header: Text { text: "✦  " + String(root.question.title).toUpperCase().split("").join(" "); font.family: Galaxy.font; font.pixelSize: 11; color: Galaxy.lavender; leftPadding: 22; topPadding: 18; elide: Text.ElideRight }
        contentItem: Text { text: root.question.text; wrapMode: Text.Wrap; color: Galaxy.ink; font.family: Galaxy.font; font.pixelSize: 13; textFormat: Text.PlainText }
        background: Rectangle { color: Galaxy.surface; border.color: Galaxy.edge; radius: Galaxy.radiusPanel }
        footer: RowLayout {
            spacing: 10
            Item { Layout.fillWidth: true }
            GalaxyButton { text: "Cancel"; Layout.bottomMargin: 18; onClicked: confirmation.reject() }
            GalaxyButton { text: root.question.proceed; primary: !root.question.danger; danger: !!root.question.danger
                Layout.rightMargin: 22; Layout.bottomMargin: 18; onClicked: confirmation.accept() }
        }
        onAccepted: if (root.question.run) root.question.run()
    }

    // ------------------------------------------------------------------ file picker (one dialog, parented to the window)
    function pick(kind, done) {
        const home = "file://" + Quickshell.env("HOME")
        const spec = {
            image: {title: "Choose an image", folder: home + "/Pictures", filters: ["Images (*.png *.jpg *.jpeg *.webp)", "All files (*)"], save: false},
            video: {title: "Choose an input video", folder: home + "/Videos", filters: ["Video (*.mp4 *.mkv *.mov *.avi *.webm)", "All files (*)"], save: false},
            save: {title: "Save video as", folder: home + "/Videos", filters: ["MP4 video (*.mp4)"], save: true}}[kind]
        picker.done = done
        picker.title = spec.title; picker.nameFilters = spec.filters; picker.currentFolder = spec.folder
        picker.fileMode = spec.save ? FileDialog.SaveFile : FileDialog.OpenFile
        picker.defaultSuffix = spec.save ? "mp4" : ""
        picker.open()
    }

    FloatingWindow {
        id: window; title: "Galaxy Deepfake"; visible: root.opened; color: "transparent"
        // test hooks (screenshots): DEEPFAKE_UI_PAGE=Models DEEPFAKE_UI_SIZE=760x560 DEEPFAKE_UI_SCREENSHOT=/tmp/x.png
        readonly property var forcedSize: (Quickshell.env("DEEPFAKE_UI_SIZE") || "").split("x").map(Number)
        implicitWidth: forcedSize.length === 2 ? forcedSize[0] : 1240; implicitHeight: forcedSize.length === 2 ? forcedSize[1] : 840
        minimumSize: Qt.size(760, 560)
        Component.onCompleted: {
            const renamed = {Dashboard: "Studio", Engines: "Studio", Camera: "Settings", Recording: "Render"}
            const requested = Quickshell.env("DEEPFAKE_UI_PAGE")
            if (requested) root.page = renamed[requested] || requested
        }
        Timer { interval: 2500; running: !!Quickshell.env("DEEPFAKE_UI_SCREENSHOT"); onTriggered: frame.grabToImage(function(result) { result.saveToFile(Quickshell.env("DEEPFAKE_UI_SCREENSHOT")); if (Quickshell.env("DEEPFAKE_UI_SCREENSHOT_QUIT")) Qt.quit() }) }
        onClosed: root.opened = false
        // The file dialog lives inside the window with an explicit parentWindow: declared on ShellRoot (no window)
        // it never opens under Quickshell (visible stays false).
        FileDialog { id: picker; property var done: null; parentWindow: window.contentItem.Window.window
            onAccepted: if (done) done(root.path(selectedFile)) }
        Timer { interval: 800; running: !!Quickshell.env("DEEPFAKE_UI_TEST_DIALOG")
            onTriggered: { root.pick("image", null); Qt.callLater(() => console.log("DIALOG visible=" + picker.visible)) } }
        Shortcut { sequence: "Ctrl+Q"; onActivated: { if (root.active) root.stop(); root.opened = false } }
        Shortcut { sequence: "Ctrl+S"; onActivated: root.flush() }
        Shortcut { sequence: "F11"; onActivated: { root.page = "Studio"; root.expanded = !root.expanded } }
        Shortcut { sequence: "Escape"; onActivated: root.expanded = false }
        Repeater { model: root.pages.length; Item { required property int index; Shortcut { sequence: "Ctrl+" + (index + 1); onActivated: root.go(root.pages[index][0]) } } }

        Rectangle {
            id: frame
            anchors.fill: parent
            radius: root.expanded ? 0 : Galaxy.radiusWindow
            color: "transparent"
            Nebula { anchors.fill: parent; radius: frame.radius; seed: 11; rim: !root.expanded; hot: root.active }

            RowLayout {
                anchors.fill: parent; anchors.margins: root.expanded ? 0 : 14; spacing: 14

                // ---------------------------------------------------------------- sidebar
                GalaxyPanel {
                    visible: !root.expanded
                    Layout.preferredWidth: root.narrow ? 64 : 196; Layout.fillHeight: true
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
                            model: root.pages
                            GalaxyNav { required property var modelData; text: modelData[0]; icon: modelData[1]; compact: root.narrow
                                current: root.page === modelData[0]; onClicked: root.go(modelData[0])
                                // a dot on Models while a download or check runs, on Render while a file renders
                                Rectangle { id: busyDot
                                    visible: (parent.modelData[0] === "Models" && root.jobRunning) || (parent.modelData[0] === "Render" && root.active && root.session.mode === "video")
                                    anchors.right: parent.right; anchors.verticalCenter: parent.verticalCenter; anchors.rightMargin: root.narrow ? 6 : 12
                                    width: 7; height: 7; radius: 4; color: Galaxy.purple
                                    SequentialAnimation on opacity { running: busyDot.visible; loops: Animation.Infinite; NumberAnimation { to: 0.3; duration: 800 } NumberAnimation { to: 1; duration: 800 } } }
                            }
                        }
                        Item { Layout.fillHeight: true }
                        GalaxyPill { visible: !root.narrow; Layout.alignment: Qt.AlignHCenter
                            state_: !root.connected ? "err" : root.active ? "ok" : root.jobRunning ? "busy" : "idle"
                            text: !root.connected ? "BACKEND OFFLINE" : root.active ? String(root.session.state || "live").toUpperCase() : root.jobRunning ? "MODEL JOB" : "CAMERA OFF" }
                        Text { visible: !root.narrow; Layout.fillWidth: true; horizontalAlignment: Text.AlignHCenter; font.family: Galaxy.font; font.pixelSize: 9; color: Galaxy.dim; Layout.bottomMargin: 6
                            text: "on-device · nothing leaves this PC" + (root.data.version ? "\nv" + root.data.version : "") }
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
                        GalaxyButton {
                            text: root.active ? "■  Stop" : root.offlineEngine ? "󰑋  Render…" : "▶  Start"
                            primary: !root.active; danger: root.active
                            enabled: root.active || (root.offlineEngine ? root.connected : root.ready) && !root.busy
                            hint: root.active ? "Stop the session and finalise any recording"
                                : root.offlineEngine ? root.engineChoice.sub.split(" · ")[0] + " renders video files: open Render"
                                : root.blockers.length ? root.blockers[0].title : "Start the webcam with " + root.engineChoice.sub.split(" · ")[0]
                            onClicked: root.active ? root.stop() : root.offlineEngine ? root.go("Render") : root.start(false, false)
                        }
                        GalaxyButton { text: "×"; hint: "Hide window · a running session keeps running"; onClicked: root.opened = false }
                    }
                    // backend unreachable: say so where every page sees it
                    Rectangle { visible: !root.connected && !root.expanded && root.offlineReason.length > 0; Layout.fillWidth: true; implicitHeight: offlineText.implicitHeight + 18
                        radius: Galaxy.radiusControl; color: Galaxy.errBg; border.color: Galaxy.err
                        Text { id: offlineText; anchors.fill: parent; anchors.margins: 9; text: root.offlineReason; color: Galaxy.err; font.family: Galaxy.font; font.pixelSize: 12; wrapMode: Text.Wrap } }

                    ScrollView {
                        id: scroller
                        Layout.fillWidth: true; Layout.fillHeight: true; clip: true; contentWidth: availableWidth
                        ScrollBar.vertical: ScrollBar { width: 6; contentItem: Rectangle { radius: 3; color: Galaxy.edge } }
                        ColumnLayout {
                            width: scroller.availableWidth; spacing: 12
                            // Pages stay instantiated so what was typed survives switching between them.
                            StudioPage { app: root; visible: root.page === "Studio" || root.expanded }
                            CharactersPage { app: root; visible: root.page === "Characters" && !root.expanded }
                            ModelsPage { app: root; visible: root.page === "Models" && !root.expanded }
                            RenderPage { app: root; visible: root.page === "Render" && !root.expanded }
                            PerformancePage { app: root; visible: root.page === "Performance" && !root.expanded }
                            SettingsPage { app: root; visible: root.page === "Settings" && !root.expanded }
                        }
                    }
                }
            }

            // ---------------------------------------------------------------- toasts
            Column {
                visible: !root.expanded
                anchors.right: parent.right; anchors.bottom: parent.bottom; anchors.margins: 26; spacing: 8
                width: Math.min(420, parent.width - 52)
                Repeater {
                    model: root.toasts
                    Rectangle {
                        id: toast
                        required property var modelData
                        readonly property bool bad: modelData.kind === "error"
                        width: parent.width; implicitHeight: toastText.implicitHeight + 22; radius: Galaxy.radiusControl
                        color: bad ? Galaxy.errBg : Galaxy.raised
                        border.color: bad ? Galaxy.err : modelData.kind === "ok" ? Galaxy.edgeHot : Galaxy.edge
                        RowLayout { anchors.fill: parent; anchors.margins: 11; spacing: 10
                            Text { text: toast.bad ? "!" : "✦"; color: toast.bad ? Galaxy.err : Galaxy.purple; font.pixelSize: 13; Layout.alignment: Qt.AlignTop }
                            Text { id: toastText; Layout.fillWidth: true; text: toast.modelData.text; wrapMode: Text.Wrap; maximumLineCount: 8; elide: Text.ElideRight
                                textFormat: Text.PlainText; color: toast.bad ? Galaxy.err : Galaxy.ink; font.family: Galaxy.font; font.pixelSize: 12 }
                            Text { text: "×"; color: Galaxy.muted; font.pixelSize: 14; Layout.alignment: Qt.AlignTop
                                MouseArea { anchors.fill: parent; anchors.margins: -6; cursorShape: Qt.PointingHandCursor; onClicked: root.dismiss(toast.modelData.id) } }
                        }
                        // errors stay until dismissed; everything else leaves on its own
                        Timer { interval: 5000; running: !toast.bad; onTriggered: root.dismiss(toast.modelData.id) }
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
