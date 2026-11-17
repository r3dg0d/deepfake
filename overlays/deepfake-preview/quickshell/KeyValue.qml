import QtQuick
import QtQuick.Layouts

// One "label  value" line. A value the backend did not report is passed as "—" and drawn dim.
RowLayout {
    property string k: ""
    property string v: "—"
    property color tint: Galaxy.ink
    property int keyWidth: 104
    Layout.fillWidth: true
    spacing: 8
    Text { text: k; font.family: Galaxy.font; font.pixelSize: 11; color: Galaxy.label; Layout.preferredWidth: keyWidth; elide: Text.ElideRight }
    Text { text: v; font.family: Galaxy.mono; font.pixelSize: 11; color: v === "—" ? Galaxy.dim : tint; elide: Text.ElideMiddle; Layout.fillWidth: true }
}
