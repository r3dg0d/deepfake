import QtQuick

// Glass card: rice panel gradient, thin luminous border. Children go in the default property.
Rectangle {
    property bool hot: false
    radius: Galaxy.radiusPanel
    border.width: 1
    border.color: hot ? Galaxy.edgeHot : Galaxy.edge
    gradient: Gradient {
        GradientStop { position: 0; color: Galaxy.glassTop }
        GradientStop { position: 0.45; color: Galaxy.glassMid }
        GradientStop { position: 1; color: Galaxy.glassBottom }
    }
    Behavior on border.color { ColorAnimation { duration: Galaxy.slow } }
}
