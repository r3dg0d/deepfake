import QtQuick
import QtQuick.Window
Window {
    id: w
    property real implicitWidth: 800
    property real implicitHeight: 600
    property size minimumSize: Qt.size(0, 0)
    signal closed()
    width: implicitWidth; height: implicitHeight
    minimumWidth: minimumSize.width; minimumHeight: minimumSize.height
    onClosing: closed()
}
