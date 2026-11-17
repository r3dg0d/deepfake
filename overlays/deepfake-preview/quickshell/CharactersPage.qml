import QtQuick
import QtQuick.Layouts

// Private, local library of reference images.
ColumnLayout {
    id: page
    required property var app
    Layout.fillWidth: true
    spacing: 12

    Card { title: "Add a character"
        TextHint { text: "Images and their provenance stay in your private local library. Nothing is uploaded." }
        PathField { id: importPath; placeholderText: "Reference image"; onBrowse: app.pick("image", p => importPath.text = p) }
        GridLayout { Layout.fillWidth: true; columns: app.narrow ? 1 : 2; columnSpacing: 10; rowSpacing: 10
            GalaxyField { id: characterName; placeholderText: "Name" }
            GalaxyField { id: provenance; placeholderText: "Consent / provenance: self, synthetic character, or permission details" }
        }
        RowLayout { spacing: 10
            GalaxyButton { text: "Import"; primary: true; enabled: importPath.text.length > 0 && characterName.text.length > 0 && provenance.text.length > 0 && !app.busy
                hint: "Needs an image, a name and where the likeness comes from"
                onClicked: { app.enqueue("import_character", {path: importPath.text, name: characterName.text, provenance: provenance.text}); importPath.text = ""; characterName.text = ""; provenance.text = "" } }
            TextHint { text: "Use only likenesses you own or have permission to use." }
        }
    }

    Card { title: "Library"
        aside: TextHint { Layout.fillWidth: false; text: app.characters.length + (app.characters.length === 1 ? " character" : " characters") }
        TextHint { visible: app.characters.length === 0; text: "No characters yet. Import a reference image above; it becomes selectable here and in the Studio." }
        GridLayout { Layout.fillWidth: true; visible: app.characters.length > 0
            columns: Math.max(1, Math.floor(width / 250)); columnSpacing: 10; rowSpacing: 10
            Repeater { model: app.characters
                Rectangle { id: tile
                    required property var modelData
                    readonly property bool current: app.form.reference === modelData.path
                    Layout.fillWidth: true; implicitHeight: 108; radius: Galaxy.radiusTile; color: Galaxy.tile
                    border.color: current ? Galaxy.purple : Galaxy.line
                    RowLayout { anchors.fill: parent; anchors.margins: 10; spacing: 12
                        Rectangle { Layout.preferredWidth: 86; Layout.fillHeight: true; radius: Galaxy.radiusControl; color: Galaxy.base; clip: true
                            Image { anchors.fill: parent; fillMode: Image.PreserveAspectCrop; asynchronous: true; sourceSize.width: 172; sourceSize.height: 172; source: "file://" + tile.modelData.path } }
                        ColumnLayout { Layout.fillWidth: true; spacing: 3
                            TextLabel { text: tile.modelData.name; elide: Text.ElideRight; wrapMode: Text.NoWrap }
                            TextHint { text: tile.modelData.provenance; maximumLineCount: 2; elide: Text.ElideRight }
                            TextHint { text: tile.modelData.width + "×" + tile.modelData.height; color: Galaxy.dim }
                            RowLayout { spacing: 6
                                GalaxyButton { text: tile.current ? "In use" : "Use"; implicitHeight: 28; font.pixelSize: 11; selected: tile.current; primary: !tile.current
                                    enabled: !tile.current && !app.active; onClicked: app.edit("reference", tile.modelData.path) }
                                GalaxyButton { text: "Remove"; implicitHeight: 28; font.pixelSize: 11; danger: true
                                    onClicked: app.ask({title: "remove character", text: "Remove " + tile.modelData.name + " from the library?", proceed: "Remove", danger: true,
                                                        run: () => { if (tile.current) app.edit("reference", ""); app.enqueue("remove_character", {id: tile.modelData.id}) }}) }
                            }
                        }
                    }
                }
            }
        }
    }
}
