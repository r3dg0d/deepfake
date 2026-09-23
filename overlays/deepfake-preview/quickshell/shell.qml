import QtQuick
import QtQuick.Layouts
import Quickshell
import Quickshell.Wayland
import Quickshell.Io

Scope {
    id: root

    property var session: ({
        active: false,
        mode: "webcam",
        state: "init",
        source_face: "",
        input_device: "",
        output_device: "",
        resolution: "",
        source_fps: 0,
        alphaface_fps: 0,
        framegen_enabled: false,
        framegen_multiplier: 0,
        output_fps: 0,
        latency_ms: 0,
        vram_mb: 0,
        vram_total_mb: 0,
        gpu_util_pct: 0,
        progress_pct: 0,
        eta_s: 0,
        framegen_warning: "",
        uptime_s: 0
    })
    property var preview: ({ active: false, fps: 0, frames: 0, width: 0, height: 0, uptime_s: 0 })
    readonly property string stateDir: Quickshell.env("HOME") + "/.local/state/deepfake"
    readonly property string framePath: stateDir + "/preview.png"
    readonly property string iconDir: Qt.resolvedUrl("./icons/")
    property string frameUrl: ""
    property bool showMetrics: true

    function writeControl(obj) {
        // FileView can't write easily; use a tiny shell helper
        Quickshell.execDetached([
            "bash", "-lc",
            "printf '%s\\n' '" + JSON.stringify(obj).replace(/'/g, "'\\''") + "' > '" + root.stateDir + "/control.json'"
        ])
    }

    function statusLabel() {
        var st = String(root.session.state || "")
        if (st === "running" && root.session.mode === "virtualcam")
            return "VIRTUAL CAMERA LIVE"
        if (st === "running")
            return "LIVE"
        if (st === "processing")
            return "PROCESSING"
        if (st === "starting")
            return "STARTING"
        if (root.session.framegen_warning)
            return "DEGRADED"
        return st.toUpperCase() || "IDLE"
    }

    function modeLabel() {
        var m = String(root.session.mode || "")
        if (m === "virtualcam") return "Virtual Camera"
        if (m === "video") return "Video Processing"
        if (m === "webcam") return "Webcam"
        return m
    }

    FileView {
        id: sessionFile
        path: root.stateDir + "/session.json"
        watchChanges: true
        onFileChanged: sessionFile.reload()
        onLoaded: {
            try {
                root.session = JSON.parse(sessionFile.text())
            } catch (e) {
                root.session = ({ active: false })
            }
        }
    }

    FileView {
        id: previewFile
        path: root.stateDir + "/preview.json"
        watchChanges: true
        onFileChanged: previewFile.reload()
        onLoaded: {
            try {
                root.preview = JSON.parse(previewFile.text())
                if (root.preview && root.preview.active)
                    root.frameUrl = "file://" + root.framePath + "?t=" + (root.preview.frames || Date.now())
            } catch (e) {
                root.preview = ({ active: false })
            }
        }
    }

    PanelWindow {
        id: win
        visible: !!(root.session && root.session.active)
        screen: Quickshell.screens.length > 0 ? Quickshell.screens[0] : null
        WlrLayershell.layer: WlrLayer.Overlay
        WlrLayershell.keyboardFocus: WlrKeyboardFocus.None
        exclusiveZone: 0
        color: "transparent"
        anchors { bottom: true; right: true }
        margins { bottom: 16; right: 16 }
        implicitWidth: 380
        implicitHeight: card.implicitHeight

        Rectangle {
            id: card
            anchors.fill: parent
            radius: 12
            color: "#f0080c08"
            border.color: "#00ff66"
            border.width: 1
            implicitHeight: col.implicitHeight + 24

            ColumnLayout {
                id: col
                anchors.fill: parent
                anchors.margins: 12
                spacing: 8

                RowLayout {
                    Layout.fillWidth: true
                    spacing: 8
                    Text {
                        text: "DEEPFAKE"
                        color: "#00ff66"
                        font.pixelSize: 12
                        font.bold: true
                        font.letterSpacing: 1.6
                        font.family: "monospace"
                        Layout.fillWidth: true
                    }
                    Rectangle {
                        radius: 6
                        color: "#122612"
                        border.color: "#00ff66"
                        border.width: 1
                        implicitHeight: statusTxt.implicitHeight + 8
                        implicitWidth: statusTxt.implicitWidth + 14
                        Text {
                            id: statusTxt
                            anchors.centerIn: parent
                            text: "● " + root.statusLabel()
                            color: "#b8ffb8"
                            font.pixelSize: 11
                            font.bold: true
                            font.family: "monospace"
                        }
                    }
                    Rectangle {
                        width: 28; height: 28; radius: 6
                        color: closeMa.containsMouse ? "#2a1515" : "#141414"
                        border.color: closeMa.containsMouse ? "#ff5555" : "#335533"
                        border.width: 1
                        Image {
                            anchors.centerIn: parent
                            source: root.iconDir + "close.svg"
                            sourceSize.width: 14; sourceSize.height: 14
                            width: 14; height: 14
                        }
                        MouseArea {
                            id: closeMa
                            anchors.fill: parent
                            hoverEnabled: true
                            cursorShape: Qt.PointingHandCursor
                            onClicked: root.writeControl({ cmd: "stop" })
                        }
                    }
                }

                Text {
                    text: root.modeLabel()
                    color: "#7dff9a"
                    font.pixelSize: 13
                    font.bold: true
                    font.family: "monospace"
                    Layout.fillWidth: true
                }

                Rectangle {
                    Layout.fillWidth: true
                    Layout.preferredHeight: 160
                    radius: 8
                    color: "#050805"
                    border.color: "#1a4a1a"
                    border.width: 1
                    clip: true
                    visible: root.session.mode !== "video" || !!(root.preview && root.preview.active)
                    Image {
                        anchors.fill: parent
                        anchors.margins: 4
                        fillMode: Image.PreserveAspectFit
                        asynchronous: true
                        cache: false
                        source: root.frameUrl
                    }
                }

                // Video progress
                ColumnLayout {
                    visible: root.session.mode === "video"
                    Layout.fillWidth: true
                    spacing: 4
                    Text {
                        text: (root.session.input_video || "") + " → " + (root.session.output_video || "")
                        color: "#7dff9a"
                        font.pixelSize: 10
                        font.family: "monospace"
                        elide: Text.ElideMiddle
                        Layout.fillWidth: true
                    }
                    Rectangle {
                        Layout.fillWidth: true
                        height: 8
                        radius: 4
                        color: "#122612"
                        Rectangle {
                            width: parent.width * Math.min(1, Math.max(0, (root.session.progress_pct || 0) / 100))
                            height: parent.height
                            radius: 4
                            color: "#00ff66"
                        }
                    }
                    Text {
                        text: Number(root.session.progress_pct || 0).toFixed(0) + "% · ETA " + Number(root.session.eta_s || 0).toFixed(0) + "s"
                        color: "#3d7a4a"
                        font.pixelSize: 10
                        font.family: "monospace"
                    }
                }

                GridLayout {
                    Layout.fillWidth: true
                    columns: 2
                    columnSpacing: 12
                    rowSpacing: 3
                    visible: root.showMetrics

                    Text { text: "FACE"; color: "#3d7a4a"; font.pixelSize: 10; font.family: "monospace" }
                    Text { text: String(root.session.source_face || "—"); color: "#7dff9a"; font.pixelSize: 11; font.bold: true; font.family: "monospace"; Layout.fillWidth: true; elide: Text.ElideMiddle }

                    Text { text: "INPUT"; color: "#3d7a4a"; font.pixelSize: 10; font.family: "monospace" }
                    Text { text: String(root.session.input_device || root.session.input_video || "—"); color: "#7dff9a"; font.pixelSize: 11; font.bold: true; font.family: "monospace"; Layout.fillWidth: true; elide: Text.ElideMiddle }

                    Text { text: "OUTPUT"; color: "#3d7a4a"; font.pixelSize: 10; font.family: "monospace"; visible: !!root.session.output_device }
                    Text { text: String(root.session.output_device || ""); color: "#7dff9a"; font.pixelSize: 11; font.bold: true; font.family: "monospace"; Layout.fillWidth: true; visible: !!root.session.output_device }

                    Text { text: "RES"; color: "#3d7a4a"; font.pixelSize: 10; font.family: "monospace" }
                    Text { text: String(root.session.resolution || ((root.preview.width||0) + "×" + (root.preview.height||0))); color: "#7dff9a"; font.pixelSize: 11; font.bold: true; font.family: "monospace"; Layout.fillWidth: true }

                    Text { text: "SOURCE"; color: "#3d7a4a"; font.pixelSize: 10; font.family: "monospace" }
                    Text { text: Number(root.session.source_fps || 0).toFixed(1) + " FPS"; color: "#7dff9a"; font.pixelSize: 11; font.bold: true; font.family: "monospace"; Layout.fillWidth: true }

                    Text { text: "ALPHAFACE"; color: "#3d7a4a"; font.pixelSize: 10; font.family: "monospace" }
                    Text { text: Number(root.session.alphaface_fps || 0).toFixed(1) + " FPS"; color: "#7dff9a"; font.pixelSize: 11; font.bold: true; font.family: "monospace"; Layout.fillWidth: true }

                    Text { text: "FRAMEGEN"; color: "#3d7a4a"; font.pixelSize: 10; font.family: "monospace" }
                    Text {
                        text: root.session.framegen_enabled
                              ? (String(root.session.framegen_multiplier || "?") + "×")
                              : (root.session.framegen_warning ? "OFF (fallback)" : "OFF")
                        color: root.session.framegen_enabled ? "#7dff9a" : "#ffaa55"
                        font.pixelSize: 11; font.bold: true; font.family: "monospace"; Layout.fillWidth: true
                    }

                    Text { text: "OUTPUT FPS"; color: "#3d7a4a"; font.pixelSize: 10; font.family: "monospace" }
                    Text { text: Number(root.session.output_fps || root.preview.fps || 0).toFixed(1) + " FPS"; color: "#7dff9a"; font.pixelSize: 11; font.bold: true; font.family: "monospace"; Layout.fillWidth: true }

                    Text { text: "LATENCY"; color: "#3d7a4a"; font.pixelSize: 10; font.family: "monospace" }
                    Text { text: Number(root.session.latency_ms || 0).toFixed(0) + " ms"; color: "#7dff9a"; font.pixelSize: 11; font.bold: true; font.family: "monospace"; Layout.fillWidth: true }

                    Text { text: "VRAM"; color: "#3d7a4a"; font.pixelSize: 10; font.family: "monospace" }
                    Text {
                        text: root.session.vram_mb
                              ? (Number(root.session.vram_mb).toFixed(1) + " / " + Number(root.session.vram_total_mb || 0).toFixed(0) + " GB").replace(" GB", " MB")
                              : "—"
                        color: "#7dff9a"; font.pixelSize: 11; font.bold: true; font.family: "monospace"; Layout.fillWidth: true
                    }
                }

                // Controls
                RowLayout {
                    Layout.fillWidth: true
                    spacing: 6
                    Repeater {
                        model: [
                            { label: "FG auto", cmd: { cmd: "set_frame_gen", value: "auto" } },
                            { label: "2×", cmd: { cmd: "set_frame_gen", value: "2x" } },
                            { label: "3×", cmd: { cmd: "set_frame_gen", value: "3x" } },
                            { label: "FG off", cmd: { cmd: "set_frame_gen", value: "off" } }
                        ]
                        delegate: Rectangle {
                            radius: 6
                            color: ma.containsMouse ? "#1a3a1a" : "#101810"
                            border.color: "#335533"
                            border.width: 1
                            implicitHeight: 26
                            implicitWidth: lab.implicitWidth + 14
                            Text {
                                id: lab
                                anchors.centerIn: parent
                                text: modelData.label
                                color: "#7dff9a"
                                font.pixelSize: 10
                                font.family: "monospace"
                            }
                            MouseArea {
                                id: ma
                                anchors.fill: parent
                                hoverEnabled: true
                                cursorShape: Qt.PointingHandCursor
                                onClicked: root.writeControl(modelData.cmd)
                            }
                        }
                    }
                }

                Text {
                    visible: !!root.session.framegen_warning
                    text: "⚠ " + String(root.session.framegen_warning || "")
                    color: "#ffaa55"
                    font.pixelSize: 10
                    font.family: "monospace"
                    wrapMode: Text.Wrap
                    Layout.fillWidth: true
                }

                Text {
                    text: "session " + String(root.session.session_id || "") + " · Ctrl+C cleans up"
                    color: "#2a5a35"
                    font.pixelSize: 9
                    font.family: "monospace"
                    Layout.fillWidth: true
                }
            }
        }

        Timer {
            interval: 500
            running: win.visible
            repeat: true
            onTriggered: {
                sessionFile.reload()
                previewFile.reload()
            }
        }
    }
}
