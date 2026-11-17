import QtQuick
import QtQuick.Layouts

// Segmented choice. `model` holds strings or {value, label, hint} objects; `picked(value)` fires on a click.
Rectangle {
    id: seg
    property var model: []
    property string value: ""
    property bool stretch: false
    signal picked(string value)
    implicitHeight: 36
    implicitWidth: row.implicitWidth + 6
    radius: Galaxy.radiusControl
    color: Galaxy.base
    border.color: Galaxy.line
    opacity: enabled ? 1 : 0.55
    RowLayout {
        id: row
        anchors.fill: parent; anchors.margins: 3; spacing: 2
        Repeater {
            model: seg.model
            Rectangle {
                id: cell
                required property var modelData
                readonly property string v: typeof modelData === "string" ? modelData : modelData.value
                readonly property string l: typeof modelData === "string" ? modelData : modelData.label
                readonly property bool on: v === seg.value
                Layout.fillHeight: true
                Layout.fillWidth: seg.stretch
                implicitWidth: Math.max(46, text.implicitWidth + 22)
                radius: 8
                color: on ? Galaxy.raised : area.containsMouse ? Galaxy.surface : "transparent"
                border.color: on ? Galaxy.purple : "transparent"
                Behavior on color { ColorAnimation { duration: Galaxy.fast } }
                Text { id: text; anchors.centerIn: parent; text: cell.l; font.family: Galaxy.font; font.pixelSize: 12
                    color: cell.on ? Galaxy.ink : Galaxy.muted; elide: Text.ElideRight; width: Math.min(implicitWidth, cell.width - 8); horizontalAlignment: Text.AlignHCenter }
                MouseArea { id: area; anchors.fill: parent; hoverEnabled: true; cursorShape: Qt.PointingHandCursor; onClicked: if (!cell.on) seg.picked(cell.v) }
            }
        }
    }
}
