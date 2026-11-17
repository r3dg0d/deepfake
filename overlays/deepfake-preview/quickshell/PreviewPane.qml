import QtQuick

// Live preview of the processed (or original) frames the session writes to the state directory.
// Double-buffered: the next frame decodes off-thread into the hidden image and is swapped in only when
// ready, so the preview never blanks and the UI thread never blocks on PNG decode.
Rectangle {
    id: pv
    required property var app
    property string kind: "processed"
    color: Galaxy.base
    radius: Galaxy.radiusPanel
    clip: true
    border.color: app.active ? Galaxy.edgeHot : Galaxy.line
    property int front: 0
    readonly property string url: app.active && pv.visible ? "file://" + app.data.state_dir + "/" + (pv.kind === "original" ? "original.png" : "preview.png") : ""
    readonly property bool ready: imgA.status === Image.Ready || imgB.status === Image.Ready
    property int tick: app.serial
    onTickChanged: {
        if (!url) return
        const back = front === 0 ? imgB : imgA
        if (back.status !== Image.Loading) back.source = url + "?v=" + tick
    }
    onUrlChanged: { imgA.source = url ? url + "?v=a" : ""; imgB.source = ""; front = 0 }
    Image { id: imgA; anchors.fill: parent; anchors.margins: 6; visible: pv.front === 0
        fillMode: Image.PreserveAspectFit; cache: false; asynchronous: true; mirror: !!app.form.mirror; smooth: true
        onStatusChanged: if (status === Image.Ready && pv.front !== 0) pv.front = 0 }
    Image { id: imgB; anchors.fill: parent; anchors.margins: 6; visible: pv.front === 1
        fillMode: Image.PreserveAspectFit; cache: false; asynchronous: true; mirror: !!app.form.mirror; smooth: true
        onStatusChanged: if (status === Image.Ready && pv.front !== 1) pv.front = 1 }
    Column {
        anchors.centerIn: parent; spacing: 10; width: parent.width - 40
        visible: !app.active || !pv.ready
        Text { anchors.horizontalCenter: parent.horizontalCenter; font.family: Galaxy.icons; text: app.active ? "󰄀" : "󰗟"; font.pixelSize: 52; color: Galaxy.purple }
        Text { width: parent.width; horizontalAlignment: Text.AlignHCenter; wrapMode: Text.Wrap; font.family: Galaxy.font; color: Galaxy.muted; font.pixelSize: 13
            text: app.active ? (app.session.loading_stage || "Waiting for the first frame…") : "Camera is off · it opens only when you press Start" }
    }
    Row {
        anchors.left: parent.left; anchors.top: parent.top; anchors.margins: 12; spacing: 6
        Rectangle { implicitWidth: tag.implicitWidth + 14; implicitHeight: 20; radius: 10; color: "#cc07060d"; border.color: Galaxy.edge
            Text { id: tag; anchors.centerIn: parent; font.family: Galaxy.font; font.pixelSize: 9; font.letterSpacing: 1.2
                color: pv.kind === "original" ? Galaxy.muted : Galaxy.lavender
                text: pv.kind === "original" ? "ORIGINAL" : "PROCESSED · SYNTHETIC MEDIA" } }
    }
    Flow {
        anchors.left: parent.left; anchors.right: parent.right; anchors.bottom: parent.bottom; anchors.margins: 12; spacing: 6
        visible: app.active && pv.kind !== "original"
        Repeater {
            model: [app.session.resolution || "—", app.num(app.session.source_fps, 0) + " fps in", "󰥔 " + app.dur(app.session.uptime_s),
                    app.session.framegen_enabled ? "FG " + String(app.session.framegen_multiplier || "auto") + "×" : "FG off",
                    app.session.rotoscope ? "matte " + (app.session.background && app.session.background !== "keep" ? "· " + app.session.background : "on") : "",
                    app.session.output_device ? "󰄀 " + app.session.output_device : "preview only"].filter(t => t.length > 0)
            Rectangle { required property var modelData; implicitWidth: chip.implicitWidth + 16; implicitHeight: 22; radius: 11; color: "#cc07060d"; border.color: Galaxy.edge
                Text { id: chip; anchors.centerIn: parent; text: modelData; font.family: Galaxy.font; font.pixelSize: 10; color: Galaxy.lavender } }
        }
    }
}
