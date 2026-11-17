import QtQuick
import QtQuick.Layouts

// Files: render an input video with the selected engine, or record the webcam session to a file.
ColumnLayout {
    id: page
    required property var app
    readonly property var session: app.session
    readonly property bool rendering: app.active && session.mode === "video"
    readonly property string engineName: app.engineChoice.sub.split(" · ")[0]
    readonly property var state_: app.engineState(app.engineKey)
    Layout.fillWidth: true
    spacing: 12

    Card { title: "Render a video"
        aside: GalaxyPill { state_: page.rendering ? "busy" : page.state_.state === "ok" ? "ok" : page.state_.state === "idle" ? "idle" : "warn"
            text: (page.rendering ? "RENDERING · " : "") + page.engineName.toUpperCase() + (page.rendering || !page.state_.text ? "" : " · " + page.state_.text.toUpperCase()) }
        TextHint { text: "Input video" }
        PathField { id: videoPath; text: app.form.input_video || ""; placeholderText: "Video to transform"; enabled: !app.active
            onCommitted: p => app.edit("input_video", p); onBrowse: app.pick("video", p => { videoPath.text = p; app.edit("input_video", p) }) }
        TextHint { text: "Output file (a new file; existing files are never overwritten)" }
        PathField { id: outputPath; text: app.form.recording_path || ""; placeholderText: "/path/to/new-video.mp4"; enabled: !app.active
            onCommitted: p => app.edit("recording_path", p); onBrowse: app.pick("save", p => { outputPath.text = p; app.edit("recording_path", p) }) }
        GridLayout { Layout.fillWidth: true; columns: app.narrow ? 1 : 2; columnSpacing: 12; rowSpacing: 8
            ColumnLayout { spacing: 6
                TextHint { text: "Encoder" }
                Segmented { enabled: !app.active; model: [{value: "auto", label: "Auto"}, {value: "libx264", label: "x264 (CPU)"}, {value: "h264_nvenc", label: "NVENC"}]
                    value: app.form.encoder || "auto"; onPicked: v => app.edit("encoder", v) }
            }
            ColumnLayout { visible: app.engineKey === "editalive"; Layout.fillWidth: true; spacing: 6
                TextHint { text: "Appearance prompt (optional)" }
                GalaxyField { enabled: !app.active; text: app.form.prompt || ""; placeholderText: "Transform the person to match the reference character…"; onEditingFinished: app.edit("prompt", text) }
            }
        }
        RowLayout { spacing: 10
            GalaxyButton { text: page.rendering ? "■  Stop render" : "󰑋  Render with " + page.engineName; primary: !page.rendering; danger: page.rendering
                enabled: page.rendering || (!app.active && app.ready && !!app.form.input_video && !!app.form.recording_path && !app.busy)
                hint: app.blockers.length ? app.blockers[0].title : !app.form.input_video ? "Choose an input video" : !app.form.recording_path ? "Choose an output file" : ""
                onClicked: page.rendering ? app.stop() : app.render() }
            GalaxyButton { text: "Change engine"; onClicked: app.go("Studio") }
        }
        Repeater { model: app.blockers
            CheckRow { required property var modelData; ok: false; blocking: true; title: modelData.title; detail: modelData.detail
                actionLabel: modelData.label; onAction: if (modelData.run) modelData.run() } }
        TextHint { text: app.offlineEngine
            ? page.engineName + " runs as an isolated worker under the memory guard. Frame generation is off for file renders; the disclosure watermark is added on export."
            : "The file is processed frame by frame with the live engine at its own frame rate; audio is copied from the input." }
    }

    Card { title: page.rendering ? "Progress" : "Last render"; visible: page.rendering || (session.mode === "video" && (session.state === "error" || session.state === "stopped") && !!session.backend)
        hot: page.rendering
        TextLabel { text: page.rendering ? (session.loading_stage || "Processing…") : session.state === "error" ? "Failed: " + String(session.error || "").slice(0, 500) : (session.loading_stage || "Finished")
            color: session.state === "error" && !page.rendering ? Galaxy.err : Galaxy.ink }
        GalaxyProgress { visible: page.rendering
            indeterminate: !(session.loading_stage === "Verifying checkpoints" && session.model_bytes_total) && !session.progress_pct
            value: session.loading_stage === "Verifying checkpoints" && session.model_bytes_total ? session.model_bytes_verified / session.model_bytes_total : (session.progress_pct || 0) / 100 }
        GridLayout { Layout.fillWidth: true; columns: app.narrow ? 1 : 2; columnSpacing: 20; rowSpacing: 4
            KeyValue { k: "engine"; v: String(session.backend || session.engine || "—") }
            KeyValue { k: "elapsed"; v: session.uptime_s !== undefined ? app.dur(session.uptime_s) : "—" }
            KeyValue { k: "worker RAM"; v: session.worker_rss_gb !== undefined ? session.worker_rss_gb + " GiB of " + session.memory_limit_gb + " GiB ceiling" : "—" }
            KeyValue { k: "host RAM free"; v: session.host_available_gb !== undefined ? session.host_available_gb + " GiB" : "—" }
            KeyValue { k: "memory guard"; v: String(session.memory_guard || "—") }
            KeyValue { k: "output"; v: String(session.output_video || app.form.recording_path || "—") }
        }
    }

    Card { title: "Record the webcam"; visible: !app.offlineEngine
        TextHint { text: "Starts a new webcam session that also writes the output file chosen above. Stop finalises the file. Live audio is not captured; the disclosure stays on every recording." }
        RowLayout { spacing: 10
            GalaxyButton { text: "●  Start recording"; primary: true; enabled: app.canStartLive && !!app.form.recording_path && !app.busy
                hint: !app.form.recording_path ? "Choose an output file first" : ""; onClicked: app.start(false, true) }
            GalaxyButton { text: "Stop and save"; enabled: app.active && !page.rendering; onClicked: app.stop() }
        }
        TextHint { visible: app.active && !!session.output_video; text: "Recording to " + (session.output_video || "") }
    }

    Card { title: "Worker log"; visible: app.offlineEngine || !!session.worker_log
        aside: GalaxyButton { text: "Refresh"; implicitHeight: 28; font.pixelSize: 11; onClicked: app.loadLog("worker") }
        LogView { text: app.logs.worker; emptyText: "The isolated worker's output appears here during and after a render." }
    }
    // follow the worker while it runs
    Timer { interval: 3000; repeat: true; running: page.rendering && page.visible && !!session.worker_log; triggeredOnStart: true; onTriggered: if (!app.busy) app.loadLog("worker") }
}
