import QtQuick
import QtQuick.Layouts

// Titled glass card that sizes itself to its content. Children go into a column:
//   Card { title: "Engine"; TextHint { text: "…" } GalaxyButton { … } }
// `aside` is an optional item shown at the right of the title (a pill, a small button).
GalaxyPanel {
    id: card
    property string title: ""
    property int padding: 16
    property alias spacing: body.spacing
    property alias aside: asideSlot.data
    default property alias cardContent: body.data
    Layout.fillWidth: true
    implicitHeight: column.implicitHeight + padding * 2
    content: ColumnLayout {
        id: column
        anchors.left: parent.left; anchors.right: parent.right; anchors.top: parent.top
        anchors.margins: card.padding
        spacing: 12
        RowLayout {
            visible: card.title.length > 0 || asideSlot.children.length > 0
            Layout.fillWidth: true; spacing: 8
            GalaxySection { title: card.title }
            RowLayout { id: asideSlot; spacing: 8 }
        }
        ColumnLayout { id: body; Layout.fillWidth: true; spacing: 10 }
    }
}
