import QtQuick
import QtQuick.Layouts
import Quickshell
import Quickshell.Wayland
import Quickshell.Io

Scope {
    id: root

    property var stats: ({ active: false, fps: 0, frames: 0, width: 0, height: 0, uptime_s: 0, ts: 0 })
    readonly property string stateDir: Quickshell.env("HOME") + "/.local/state/deepfake"
    readonly property string framePath: stateDir + "/preview.png"
    readonly property string iconDir: Qt.resolvedUrl("./icons/")
    property string frameUrl: ""

    FileView {
        id: statsFile
        path: root.stateDir + "/preview.json"
        watchChanges: true
        onFileChanged: statsFile.reload()
        onLoaded: {
            try {
                root.stats = JSON.parse(statsFile.text())
                if (root.stats && root.stats.active)
                    root.frameUrl = "file://" + root.framePath + "?t=" + (root.stats.frames || Date.now())
            } catch (e) {
                root.stats = ({ active: false })
            }
        }
    }

    PanelWindow {
        id: win
        visible: !!(root.stats && root.stats.active)
        screen: Quickshell.screens.length > 0 ? Quickshell.screens[0] : null
        WlrLayershell.layer: WlrLayer.Overlay
        WlrLayershell.keyboardFocus: WlrKeyboardFocus.None
        exclusiveZone: 0
        color: "transparent"
        anchors { bottom: true; right: true }
        margins { bottom: 16; right: 16 }
        implicitWidth: 360
        implicitHeight: card.implicitHeight

        Rectangle {
            id: card
            anchors.fill: parent
            radius: 12
            color: "#f0080c08"
            border.color: "#00ff66"
            border.width: 1
            implicitHeight: col.implicitHeight + 24

            Rectangle {
                anchors.fill: parent
                anchors.margins: 1
                radius: 11
                color: "transparent"
                border.color: "#2200ff66"
                border.width: 1
            }

            ColumnLayout {
                id: col
                anchors.fill: parent
                anchors.margins: 12
                spacing: 10

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
                        implicitHeight: fpsTxt.implicitHeight + 8
                        implicitWidth: fpsTxt.implicitWidth + 14
                        Text {
                            id: fpsTxt
                            anchors.centerIn: parent
                            text: Number(root.stats.fps || 0).toFixed(1) + " FPS"
                            color: "#b8ffb8"
                            font.pixelSize: 12
                            font.bold: true
                            font.family: "monospace"
                        }
                    }
                    Rectangle {
                        width: 28
                        height: 28
                        radius: 6
                        color: closeMa.containsMouse ? "#2a1515" : "#141414"
                        border.color: closeMa.containsMouse ? "#ff5555" : "#335533"
                        border.width: 1
                        Image {
                            anchors.centerIn: parent
                            source: root.iconDir + "close.svg"
                            sourceSize.width: 14
                            sourceSize.height: 14
                            width: 14
                            height: 14
                        }
                        MouseArea {
                            id: closeMa
                            anchors.fill: parent
                            hoverEnabled: true
                            cursorShape: Qt.PointingHandCursor
                            onClicked: {
                                root.stats = ({ active: false })
                                Quickshell.execDetached(["rm", "-f", root.stateDir + "/preview.json"])
                            }
                        }
                    }
                }

                Rectangle {
                    Layout.fillWidth: true
                    Layout.preferredHeight: 200
                    radius: 8
                    color: "#050805"
                    border.color: "#1a4a1a"
                    border.width: 1
                    clip: true
                    Image {
                        anchors.fill: parent
                        anchors.margins: 4
                        fillMode: Image.PreserveAspectFit
                        asynchronous: true
                        cache: false
                        source: root.frameUrl
                    }
                }

                GridLayout {
                    Layout.fillWidth: true
                    columns: 2
                    columnSpacing: 12
                    rowSpacing: 4
                    Text { text: "FRAMES"; color: "#3d7a4a"; font.pixelSize: 10; font.family: "monospace" }
                    Text { text: String(root.stats.frames || 0); color: "#7dff9a"; font.pixelSize: 11; font.bold: true; font.family: "monospace"; Layout.fillWidth: true }
                    Text { text: "SIZE"; color: "#3d7a4a"; font.pixelSize: 10; font.family: "monospace" }
                    Text { text: (root.stats.width || 0) + "×" + (root.stats.height || 0); color: "#7dff9a"; font.pixelSize: 11; font.bold: true; font.family: "monospace"; Layout.fillWidth: true }
                    Text { text: "UPTIME"; color: "#3d7a4a"; font.pixelSize: 10; font.family: "monospace" }
                    Text { text: Number(root.stats.uptime_s || 0).toFixed(1) + "s"; color: "#7dff9a"; font.pixelSize: 11; font.bold: true; font.family: "monospace"; Layout.fillWidth: true }
                }

                Text {
                    text: "matrix overlay · ~/.local/state/deepfake/"
                    color: "#2a5a35"
                    font.pixelSize: 10
                    font.family: "monospace"
                    Layout.fillWidth: true
                }
            }
        }

        Timer {
            interval: 1200
            running: win.visible
            repeat: true
            onTriggered: statsFile.reload()
        }
    }
}
