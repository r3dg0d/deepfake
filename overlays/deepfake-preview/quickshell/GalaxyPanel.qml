import QtQuick

// Card on the Nebula backdrop: solid, slightly translucent raised fill + thin luminous border, so the window's
// nebula reads faintly through. Solid colour only (gradients on rounded rects seam on the GPU renderer).
Item {
    id: panel
    property bool hot: false
    property real radius: Galaxy.radiusPanel
    default property alias content: holder.data
    Rectangle {
        anchors.fill: parent; radius: panel.radius
        color: Galaxy.card
        border.width: 1
        border.color: panel.hot ? Galaxy.edgeHot : Galaxy.edge
        Behavior on border.color { ColorAnimation { duration: Galaxy.slow } }
        // top hairline highlight: the "lit edge" of the rice
        Rectangle { x: panel.radius; width: parent.width - 2 * panel.radius; y: 0; height: 1; color: Galaxy.rimLight; opacity: panel.hot ? 0.9 : 0.5 }
    }
    Item { id: holder; anchors.fill: parent }
}
