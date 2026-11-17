import QtQuick
import QtQuick.Layouts

// One line of a readiness checklist: what is needed, whether it is met, and the button that fixes it.
RowLayout {
    id: check
    property bool ok: false
    property bool blocking: true
    property string title: ""
    property string detail: ""
    property string actionLabel: ""
    signal action()
    Layout.fillWidth: true
    spacing: 10
    Rectangle {
        implicitWidth: 20; implicitHeight: 20; radius: 10
        Layout.alignment: Qt.AlignTop
        color: check.ok ? Galaxy.okBg : check.blocking ? Galaxy.errBg : Galaxy.warnBg
        border.color: check.ok ? Galaxy.ok : check.blocking ? Galaxy.err : Galaxy.warn
        Text { anchors.centerIn: parent; text: check.ok ? "✓" : check.blocking ? "!" : "i"; font.pixelSize: 11; font.bold: true
            color: check.ok ? Galaxy.ok : check.blocking ? Galaxy.err : Galaxy.warn }
    }
    ColumnLayout { Layout.fillWidth: true; spacing: 1
        Text { Layout.fillWidth: true; text: check.title; font.family: Galaxy.font; font.pixelSize: 12; color: check.ok ? Galaxy.muted : Galaxy.ink; wrapMode: Text.Wrap }
        Text { Layout.fillWidth: true; visible: text.length > 0; text: check.detail; font.family: Galaxy.font; font.pixelSize: 10; color: Galaxy.label; wrapMode: Text.Wrap }
    }
    GalaxyButton { visible: !check.ok && check.actionLabel.length > 0; text: check.actionLabel; implicitHeight: 30; font.pixelSize: 12; onClicked: check.action() }
}
