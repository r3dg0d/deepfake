import QtQuick
import QtQuick.Layouts

// A selectable option with a title, a one-line description and a status on the right.
// state_: "ok" | "warn" | "err" | "idle"
Rectangle {
    id: row
    property string title: ""
    property string sub: ""
    property string status: ""
    property string state_: "idle"
    property bool selected: false
    signal clicked()
    readonly property color tone: state_ === "ok" ? Galaxy.ok : state_ === "warn" ? Galaxy.warn : state_ === "err" ? Galaxy.err : Galaxy.dim
    Layout.fillWidth: true
    implicitHeight: 52
    radius: Galaxy.radiusControl
    color: selected ? Galaxy.raised : area.containsMouse ? Galaxy.surface : Galaxy.base
    border.color: selected ? Galaxy.purple : area.containsMouse ? Galaxy.edge : Galaxy.line
    opacity: enabled ? 1 : 0.6
    Behavior on color { ColorAnimation { duration: Galaxy.fast } }
    Behavior on border.color { ColorAnimation { duration: Galaxy.fast } }
    RowLayout {
        anchors.fill: parent; anchors.leftMargin: 12; anchors.rightMargin: 12; spacing: 10
        Rectangle { implicitWidth: 14; implicitHeight: 14; radius: 7; color: "transparent"; border.color: row.selected ? Galaxy.purple : Galaxy.edge; border.width: 2
            Rectangle { anchors.centerIn: parent; width: 6; height: 6; radius: 3; color: Galaxy.purple; visible: row.selected } }
        ColumnLayout { Layout.fillWidth: true; spacing: 2
            Text { Layout.fillWidth: true; text: row.title; font.family: Galaxy.font; font.pixelSize: 13; color: Galaxy.ink; elide: Text.ElideRight }
            Text { Layout.fillWidth: true; text: row.sub; font.family: Galaxy.font; font.pixelSize: 10; color: Galaxy.label; elide: Text.ElideRight }
        }
        Rectangle { visible: row.status.length > 0; implicitWidth: 7; implicitHeight: 7; radius: 4; color: row.tone }
        Text { visible: row.status.length > 0; text: row.status; font.family: Galaxy.font; font.pixelSize: 10; color: row.tone
            Layout.maximumWidth: 120; elide: Text.ElideRight }
    }
    MouseArea { id: area; anchors.fill: parent; hoverEnabled: true; cursorShape: Qt.PointingHandCursor; onClicked: row.clicked() }
}
