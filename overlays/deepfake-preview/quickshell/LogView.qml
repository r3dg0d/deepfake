import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

// Read-only log tail in the mono face; stays scrolled to the newest line.
Rectangle {
    id: view
    property string text: ""
    property string emptyText: "Nothing logged yet"
    Layout.fillWidth: true
    implicitHeight: 220
    radius: Galaxy.radiusControl
    color: Galaxy.void_
    border.color: Galaxy.line
    clip: true
    ScrollView {
        id: scroll
        anchors.fill: parent; anchors.margins: 10
        contentWidth: availableWidth
        TextEdit {
            width: scroll.availableWidth
            readOnly: true; selectByMouse: true; wrapMode: TextEdit.WrapAnywhere; textFormat: TextEdit.PlainText
            text: view.text.length > 0 ? view.text : view.emptyText
            color: view.text.length > 0 ? Galaxy.lavender : Galaxy.dim
            font.family: Galaxy.mono; font.pixelSize: 11
            selectionColor: Galaxy.violet
            onTextChanged: Qt.callLater(() => { scroll.ScrollBar.vertical.position = Math.max(0, 1 - scroll.ScrollBar.vertical.size) })
        }
    }
}
