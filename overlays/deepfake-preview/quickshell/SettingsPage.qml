import QtQuick
import QtQuick.Layouts

ColumnLayout {
    id: page
    required property var app
    Layout.fillWidth: true
    spacing: 12

    Card { title: "Consent & disclosure"
        aside: GalaxyPill { state_: app.data.consent ? "ok" : "warn"; text: app.data.consent ? "ACKNOWLEDGED" : "REQUIRED" }
        TextLabel { text: app.data.consent_text || "Loading notice…"; color: Galaxy.muted; font.pixelSize: 12 }
        GalaxyButton { visible: !app.data.consent; text: "I own or have permission to use my references and footage"; primary: true
            onClicked: app.enqueue("consent", {accepted: true}) }
        TextHint { text: "The synthetic-media watermark stays enabled on processed output." }
    }

    Card { title: "Capture"
        TextHint { text: "The camera opens only when you press Start. Changes apply to the next session." }
        Repeater { model: app.devices
            KeyValue { required property var modelData; keyWidth: 110; k: modelData.path; v: modelData.name + " · " + modelData.kind; tint: modelData.kind === "loopback" ? Galaxy.purple : Galaxy.ink } }
        TextHint { visible: app.devices.length === 0; text: "No V4L2 devices found." }
        GridLayout { Layout.fillWidth: true; columns: app.narrow ? 2 : 4; columnSpacing: 10; rowSpacing: 6
            TextHint { text: "Camera index or device" } TextHint { text: "Width" } TextHint { text: "Height" } TextHint { text: "FPS" }
            GalaxyField { text: app.form.camera || "0"; enabled: !app.active; onEditingFinished: app.edit("camera", text) }
            GalaxyField { text: String(app.form.width || 640); enabled: !app.active; validator: IntValidator { bottom: 64; top: 4096 } onEditingFinished: app.edit("width", Number(text)) }
            GalaxyField { text: String(app.form.height || 480); enabled: !app.active; validator: IntValidator { bottom: 64; top: 4096 } onEditingFinished: app.edit("height", Number(text)) }
            GalaxyField { text: String(app.form.fps || 30); enabled: !app.active; validator: IntValidator { bottom: 1; top: 240 } onEditingFinished: app.edit("fps", Number(text)) }
        }
        TextHint { text: "Virtual camera device (empty = auto-detect the v4l2loopback device)" }
        GalaxyField { text: app.form.output_device || ""; placeholderText: "/dev/video10"; enabled: !app.active; onEditingFinished: app.edit("output_device", text) }
    }

    Card { title: "Shortcuts"
        GridLayout { Layout.fillWidth: true; columns: app.narrow ? 1 : 2; columnSpacing: 20; rowSpacing: 4
            KeyValue { k: "Ctrl+1 … 6"; v: "switch page" }
            KeyValue { k: "F11 / Esc"; v: "fullscreen preview / leave" }
            KeyValue { k: "Ctrl+S"; v: "save settings now (they also save as you edit)" }
            KeyValue { k: "Ctrl+Q"; v: "stop the session and close" }
        }
        TextHint { text: "Closing with × only hides the window; a running session keeps running." }
    }

    Card { title: "Diagnostics"
        aside: GalaxyButton { text: "Load backend log"; implicitHeight: 28; font.pixelSize: 11; onClicked: app.loadLog("backend") }
        KeyValue { k: "version"; v: String(app.data.version || "—") }
        KeyValue { k: "backend"; v: app.connected ? app.cli + " · protocol " + app.data.protocol_version : "offline" }
        KeyValue { k: "state directory"; v: String(app.data.state_dir || "—") }
        LogView { text: app.logs.backend; emptyText: "Startup, CUDA and device errors of the last session appear here." }
    }
}
