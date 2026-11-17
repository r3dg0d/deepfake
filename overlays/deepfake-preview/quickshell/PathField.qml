import QtQuick
import QtQuick.Layouts

// A path text field with a Browse… button. `committed(path)` fires when the text is edited or a file is picked.
RowLayout {
    id: field
    property alias text: input.text
    property alias placeholderText: input.placeholderText
    property string browseLabel: "Browse…"
    property bool browsable: true
    signal committed(string path)
    signal browse()
    Layout.fillWidth: true
    spacing: 8
    GalaxyField { id: input; onEditingFinished: field.committed(text) }
    GalaxyButton { visible: field.browsable; text: field.browseLabel; onClicked: field.browse() }
}
