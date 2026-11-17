import QtQuick
import QtQuick.Layouts

ColumnLayout {
    id: page
    required property var app
    readonly property var session: app.session
    Layout.fillWidth: true
    spacing: 12

    GridLayout { Layout.fillWidth: true; columns: app.narrow ? 2 : 4; columnSpacing: 10; rowSpacing: 10
        GalaxyStat { label: "Camera"; value: app.live(session.source_fps); unit: " fps" }
        GalaxyStat { label: "Native inference"; value: app.live(session.native_inference_fps); unit: " fps"; tint: Galaxy.purple }
        GalaxyStat { label: "Output"; value: app.live(session.output_fps); unit: " fps" }
        GalaxyStat { label: "Dropped source"; value: app.live(session.dropped_source_frames, 0) }
        GalaxyStat { label: "Latency p50"; value: app.live(session.latency_ms); unit: " ms" }
        GalaxyStat { label: "Latency p95"; value: app.live(session.latency_p95_ms); unit: " ms" }
        GalaxyStat { label: "Matte"; value: app.active && session.rotoscope ? app.num(session.matte_ms) : "—"; unit: " ms" }
        GalaxyStat { label: "FG ms / frame"; value: app.live(session.framegen_ms_per_frame); unit: " ms" }
        GalaxyStat { label: "GPU util"; value: app.num(app.gpu.util_pct, 0); unit: " %" }
        GalaxyStat { label: "VRAM used"; value: app.num(app.gpu.mem_used_mib, 0); unit: " MiB" }
        GalaxyStat { label: "GPU temp"; value: app.num(app.gpu.temp_c, 0); unit: " °C" }
        GalaxyStat { label: "GPU power"; value: app.num(app.gpu.power_w, 0); unit: " W" }
        GalaxyStat { label: "Queue depth"; value: app.live(session.queue_depth, 0) }
        GalaxyStat { label: "Generated frames"; value: app.live(session.generated_frames, 0) }
        GalaxyStat { label: "Session VRAM"; value: app.active && session.vram_mb ? app.num(session.vram_mb, 0) : "—"; unit: " MiB" }
        GalaxyStat { label: "Uptime"; value: app.active ? app.dur(session.uptime_s) : "—" }
    }
    Card { title: "Last 60 seconds"
        GalaxySpark { label: "OUTPUT FPS"; data: app.fpsHist; current: app.live(session.output_fps); implicitHeight: 70 }
        GalaxySpark { label: "LATENCY MS"; data: app.latHist; stroke: Galaxy.magenta; current: app.live(session.latency_ms); implicitHeight: 70 }
        TextHint { visible: !app.active; text: "Graphs fill while a session runs." }
    }
    TextHint { text: "Interpolated frames never count as native inference. The matte time is the compositing stage on the inference thread; camera-side segmentation overlaps the GPU. Metrics the backend did not report show —." }
}
