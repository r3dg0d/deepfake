import QtQuick
import QtQuick.Layouts

// Studio: the live preview on the left, everything that shapes the next session in the inspector on the right.
ColumnLayout {
    id: page
    required property var app
    readonly property var session: app.session
    readonly property bool wide: !app.narrow && !app.expanded
    readonly property bool rendering: app.active && session.mode === "video"
    Layout.fillWidth: true
    spacing: 12

    GridLayout {
        Layout.fillWidth: true
        columns: page.wide ? 2 : 1
        columnSpacing: 12; rowSpacing: 12

        // ------------------------------------------------------------ preview column
        ColumnLayout {
            Layout.fillWidth: true; Layout.alignment: Qt.AlignTop; spacing: 12
            RowLayout { Layout.fillWidth: true; spacing: 12
                PreviewPane { app: page.app; kind: "original"; visible: app.form.comparison === "original" || app.form.comparison === "side-by-side"
                    Layout.fillWidth: true; Layout.preferredHeight: app.expanded ? app.windowHeight : Math.max(280, app.windowHeight * 0.46) }
                PreviewPane { app: page.app; visible: app.form.comparison !== "original"
                    Layout.fillWidth: true; Layout.preferredHeight: app.expanded ? app.windowHeight : Math.max(280, app.windowHeight * 0.46) }
            }
            RowLayout { visible: !app.expanded; Layout.fillWidth: true; spacing: 10
                Segmented { model: [{value: "processed", label: "Processed"}, {value: "original", label: "Original"}, {value: "side-by-side", label: "Side by side"}]
                    value: app.form.comparison || "processed"; onPicked: v => app.edit("comparison", v) }
                GalaxyToggle { text: "Mirror"; checked: !!app.form.mirror; onToggled: app.edit("mirror", checked) }
                Item { Layout.fillWidth: true }
                GalaxyButton { text: "󰊓"; font.family: Galaxy.icons; hint: "Fullscreen preview (F11)"; onClicked: app.expanded = true }
            }

            // a session that ended with an error: say what happened and where the details are
            Card { visible: !app.expanded && !app.active && session.state === "error" && !!session.error; title: "Last session failed"; hot: true
                TextLabel { text: String(session.error || "").slice(0, 700); color: Galaxy.err; font.pixelSize: 12 }
                RowLayout { spacing: 8
                    GalaxyButton { text: "Backend log"; onClicked: { app.loadLog("backend"); app.go("Settings") } }
                    GalaxyButton { visible: !!session.worker_log; text: "Worker log"; onClicked: { app.loadLog("worker"); app.go("Render") } }
                }
            }

            // a file render in progress shows its stage here too
            Card { visible: !app.expanded && page.rendering; title: "Rendering"; hot: true
                aside: GalaxyButton { text: "Details"; implicitHeight: 28; font.pixelSize: 11; onClicked: app.go("Render") }
                TextLabel { text: session.loading_stage || "Processing…" }
                GalaxyProgress { indeterminate: !session.model_bytes_total || session.loading_stage !== "Verifying checkpoints"
                    value: session.model_bytes_total ? session.model_bytes_verified / session.model_bytes_total : 0 }
            }

            GridLayout { visible: !app.expanded; Layout.fillWidth: true; columns: app.narrow ? 3 : 6; columnSpacing: 10; rowSpacing: 10
                GalaxyStat { label: "Camera"; value: app.live(session.source_fps); unit: " fps" }
                GalaxyStat { label: "Inference"; value: app.live(session.native_inference_fps); unit: " fps"; tint: Galaxy.purple }
                GalaxyStat { label: "Output"; value: app.live(session.output_fps); unit: " fps" }
                GalaxyStat { label: "Latency p50"; value: app.live(session.latency_ms); unit: " ms" }
                GalaxyStat { label: "Latency p95"; value: app.live(session.latency_p95_ms); unit: " ms" }
                GalaxyStat { label: "Dropped"; value: app.live(session.dropped_source_frames, 0); tint: Number(session.dropped_source_frames) > 0 ? Galaxy.warn : Galaxy.lavender }
            }
            RowLayout { visible: !app.expanded && app.active && !page.rendering; Layout.fillWidth: true; spacing: 12
                GalaxySpark { label: "OUTPUT FPS"; data: app.fpsHist; current: app.num(session.output_fps) }
                GalaxySpark { label: "LATENCY MS"; data: app.latHist; stroke: Galaxy.magenta; current: app.num(session.latency_ms) }
            }

            // what still stands between the user and a running session, each with the button that fixes it
            Card { visible: !app.expanded && !app.active; title: app.ready ? "Ready" : "Before you start"
                aside: GalaxyPill { state_: app.ready ? "ok" : "warn"; text: app.ready ? (app.offlineEngine ? "READY TO RENDER" : "READY") : app.blockers.length + " TO DO" }
                Repeater { model: app.checklist
                    CheckRow { required property var modelData; ok: modelData.ok; blocking: modelData.blocking; title: modelData.title; detail: modelData.detail
                        actionLabel: modelData.label; onAction: if (modelData.run) modelData.run() }
                }
                TextHint { visible: app.offlineEngine; text: app.engineChoice.sub.split(" · ")[0] + " does not run on a live camera. It renders a video file on the Render page." }
            }
        }

        // ------------------------------------------------------------ inspector
        ColumnLayout {
            visible: !app.expanded
            Layout.alignment: Qt.AlignTop
            Layout.fillWidth: !page.wide
            Layout.preferredWidth: page.wide ? 330 : -1
            Layout.maximumWidth: page.wide ? 330 : Number.POSITIVE_INFINITY
            spacing: 12

            Card { title: "Engine"
                Repeater { model: app.engineChoices
                    ChoiceRow { required property var modelData
                        readonly property var st: app.engineState(modelData.key)
                        title: modelData.title; sub: modelData.sub; status: st.text; state_: st.state
                        selected: app.engineKey === modelData.key; enabled: !app.active; onClicked: app.chooseEngine(modelData.key) }
                }
                TextHint { visible: app.active; text: "Stop the session to change the engine." }
            }

            Card { title: "Character"
                RowLayout { Layout.fillWidth: true; spacing: 12
                    Rectangle { implicitWidth: 64; implicitHeight: 64; radius: Galaxy.radiusControl; color: Galaxy.base; border.color: Galaxy.line; clip: true
                        Image { anchors.fill: parent; anchors.margins: 2; visible: !!app.form.reference; fillMode: Image.PreserveAspectCrop; asynchronous: true
                            sourceSize.width: 128; sourceSize.height: 128; source: app.form.reference ? "file://" + app.form.reference : "" }
                        Text { anchors.centerIn: parent; visible: !app.form.reference; text: "󰀉"; font.family: Galaxy.icons; font.pixelSize: 26; color: Galaxy.dim }
                    }
                    ColumnLayout { Layout.fillWidth: true; spacing: 3
                        TextLabel { text: app.form.reference ? app.characterName(app.form.reference) : "No character selected"; elide: Text.ElideRight; wrapMode: Text.NoWrap }
                        TextHint { text: app.form.reference ? app.form.reference : "Import or pick a reference image"; elide: Text.ElideMiddle; wrapMode: Text.NoWrap }
                    }
                    GalaxyButton { text: app.form.reference ? "Change" : "Choose"; primary: !app.form.reference; onClicked: app.go("Characters") }
                }
            }

            // rotoscoping and background replacement exist for the LivePortrait head engine only
            Card { visible: app.headEngine; title: "Matte"
                aside: GalaxyPill { state_: !app.data.matte_installed ? "warn" : session.rotoscope && app.active ? "ok" : "idle"
                    text: !app.data.matte_installed ? "MODEL MISSING" : app.active ? (session.rotoscope ? "ON · " + app.num(session.matte_ms) + " MS" : "OFF") : "HEAD ENGINE" }
                TextHint { text: "Rotoscope the generated head" }
                Segmented { Layout.fillWidth: true; stretch: true; enabled: !app.active
                    model: [{value: "auto", label: "Auto"}, {value: "on", label: "On"}, {value: "off", label: "Off"}]
                    value: app.form.rotoscope || "auto"; onPicked: v => { app.edit("rotoscope", v); if (v === "off") app.edit("background", "keep") } }
                TextHint { text: "Background" }
                GridLayout { Layout.fillWidth: true; columns: 4; columnSpacing: 6; rowSpacing: 6
                    Repeater { model: [["keep", "Keep"], ["blur", "Blur"], ["green", "Green"], ["blue", "Blue"], ["black", "Black"], ["white", "White"], ["#", "Colour"], ["image", "Image"]]
                        GalaxyButton { required property var modelData
                            readonly property string current: app.form.background || "keep"
                            Layout.fillWidth: true; implicitHeight: 30; font.pixelSize: 11; text: modelData[1]
                            enabled: !app.active && app.form.rotoscope !== "off"
                            selected: modelData[0] === "#" ? current.startsWith("#") : current === modelData[0]
                            onClicked: app.edit("background", modelData[0] === "#" ? (current.startsWith("#") ? current : "#101020") : modelData[0]) }
                    }
                }
                GalaxyField { visible: String(app.form.background || "").startsWith("#"); enabled: !app.active; text: app.form.background || ""
                    placeholderText: "#rrggbb"; validator: RegularExpressionValidator { regularExpression: /#[0-9a-fA-F]{6}/ }
                    onEditingFinished: app.edit("background", text.toLowerCase()) }
                PathField { visible: app.form.background === "image"; enabled: !app.active; text: app.form.background_image || ""; placeholderText: "Background image"
                    onCommitted: p => app.edit("background_image", p); onBrowse: app.pick("image", p => app.edit("background_image", p)) }
                Segmented { visible: (app.form.background || "keep") !== "keep"; Layout.fillWidth: true; stretch: true; enabled: !app.active
                    model: [{value: "person", label: "Keep my body"}, {value: "head", label: "Head only"}]
                    value: app.form.matte_scope || "person"; onPicked: v => app.edit("matte_scope", v) }
                TextHint { text: app.form.rotoscope === "off" ? "Off: the whole square crop, including the reference image's background, is blended over the camera."
                    : (app.form.background || "keep") === "keep" ? "The character is cut out of its crop; the camera background stays."
                    : app.form.matte_scope === "head" ? "Only the generated head is kept in front of the new background."
                    : "The generated head and your body below the chin are kept in front of the new background." }
                GalaxyButton { visible: !app.data.matte_installed; text: "Install matte model (6 MB)"; primary: true; enabled: !app.jobRunning
                    onClicked: app.confirmModel("install_model", app.model("pphumanseg")) }
            }

            Card { title: "Quality"
                TextHint { text: "Preset" }
                Segmented { Layout.fillWidth: true; stretch: true; enabled: !app.active
                    model: [{value: "latency", label: "Latency"}, {value: "balanced", label: "Balanced"}, {value: "quality", label: "Quality"}]
                    value: app.form.preset || "balanced"; onPicked: v => app.edit("preset", v) }
                TextHint { visible: !app.offlineEngine; text: "Frame generation (RIFE)" }
                Segmented { visible: !app.offlineEngine; Layout.fillWidth: true; stretch: true; enabled: !app.active
                    model: [{value: "auto", label: "Auto"}, {value: "off", label: "Off"}, "2x", "3x", "4x"]
                    value: app.form.frame_gen || "auto"; onPicked: v => app.edit("frame_gen", v) }
                TextHint { visible: app.active && !!session.framegen_warning; color: Galaxy.warn; text: "Frame generation fell back to native output: " + (session.framegen_warning || "") }
            }

            Card { title: "Output"; visible: !app.offlineEngine
                GalaxyButton { Layout.fillWidth: true; implicitHeight: 42; text: app.active ? "■  Stop" : "▶  Start webcam"; primary: !app.active; danger: app.active
                    enabled: app.active || app.canStartLive && !app.busy; onClicked: app.active ? app.stop() : app.start(false, false) }
                GalaxyButton { Layout.fillWidth: true; visible: !app.active; text: "󰄀  Start with virtual camera"; enabled: app.canStartLive && !app.busy
                    hint: "Also feeds the v4l2loopback device for OBS, browsers and calls"; onClicked: app.start(true, false) }
                TextHint { text: app.active ? (session.output_device ? "Feeding " + session.output_device + "." : "Preview only. Stop to switch outputs.")
                    : "The camera opens only when you press Start. Recording to a file starts on the Render page." }
            }

            Card { title: "GPU"
                KeyValue { k: "device"; v: String(app.gpu.name || "—") }
                KeyValue { k: "VRAM"; v: app.gpu.mem_used_mib !== undefined ? app.gpu.mem_used_mib + " / " + app.gpu.mem_total_mib + " MiB" : "—" }
                KeyValue { k: "load"; v: app.gpu.util_pct !== undefined ? app.gpu.util_pct + " % · " + app.gpu.temp_c + " °C · " + app.gpu.power_w + " W" : "—" }
                KeyValue { k: "session VRAM"; v: app.active && session.vram_mb ? app.num(session.vram_mb, 0) + " MiB" : "—" }
            }
        }
    }
}
