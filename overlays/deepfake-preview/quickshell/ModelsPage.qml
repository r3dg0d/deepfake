import QtQuick
import QtQuick.Layouts

// Model manager: download, verify, repair, reinstall and remove weights, and set up the isolated
// worker environments of the experimental engines.
ColumnLayout {
    id: page
    required property var app
    readonly property var job: app.job
    Layout.fillWidth: true
    spacing: 12

    Card { title: "Model operations"
        hot: app.jobRunning
        aside: GalaxyPill {
            state_: app.jobRunning ? "busy" : job.state === "error" || job.state === "interrupted" ? "err" : job.state === "complete" ? "ok" : "idle"
            text: (job.state || "idle").toUpperCase() + (job.action ? " · " + app.jobLabel(job) : "") }
        GalaxyProgress { visible: app.jobRunning; indeterminate: !job.total_bytes; value: job.total_bytes ? job.bytes / job.total_bytes : 0 }
        RowLayout { visible: app.jobRunning; Layout.fillWidth: true; spacing: 10
            TextHint { text: (job.stage || "Working") + (job.total_bytes ? " · " + app.gib(job.bytes) + " of " + app.gib(job.total_bytes) + " (" + Math.floor(100 * job.bytes / job.total_bytes) + " %)" : "") }
            GalaxyButton { text: "Cancel"; danger: true; implicitHeight: 30; font.pixelSize: 12; hint: "Partial downloads are kept and resumed next time"; onClicked: app.enqueue("cancel_model") }
        }
        TextHint { visible: job.state === "error" || job.state === "interrupted"; color: Galaxy.err; text: String(job.error || "") }
        TextHint { visible: job.state === "complete" && !!job.result && typeof job.result === "string"; text: String(job.result || "") }
        TextHint { text: "Every file is checked against a pinned SHA-256 before it is used. Downloads resume after an interruption. Weights are never committed or uploaded." }
    }

    GalaxySection { title: "Core"; Layout.topMargin: 4 }
    Repeater { model: app.modelNames.filter(n => !app.model(n).experimental)
        ModelCard { required property string modelData; name: modelData } }

    GalaxySection { title: "Experimental engines"; Layout.topMargin: 4 }
    Repeater { model: app.modelNames.filter(n => !!app.model(n).experimental)
        ModelCard { required property string modelData; name: modelData } }

    component ModelCard: Rectangle {
        id: card
        property string name: ""
        readonly property var m: app.model(name)
        readonly property var e: app.engine(name)          // experimental engines also have a worker setup
        readonly property bool isEngine: e.engine === "experimental"
        readonly property bool installed: !!m.installed
        readonly property bool partial: !installed && m.present_bytes > 0
        readonly property string integrity: m.integrity || (installed ? "unverified" : "not installed")
        readonly property bool mine: app.jobRunning && app.job.model === name
        Layout.fillWidth: true
        implicitHeight: contents.implicitHeight + 28
        radius: Galaxy.radiusTile
        color: Galaxy.tile
        border.color: mine ? Galaxy.edgeHot : installed && integrity === "verified" ? Galaxy.edge : Galaxy.line
        ColumnLayout { id: contents; anchors.left: parent.left; anchors.right: parent.right; anchors.top: parent.top; anchors.margins: 14; spacing: 8
            RowLayout { Layout.fillWidth: true; spacing: 8
                Text { text: card.name; font.family: Galaxy.font; font.pixelSize: 14; color: Galaxy.ink }
                GalaxyPill {
                    state_: card.mine ? "busy" : m.blocked ? "err" : !card.installed ? (card.partial ? "warn" : "idle")
                          : card.integrity === "verified" ? (m.load_check_ok === false ? "err" : "ok") : "warn"
                    text: card.mine ? String(app.job.action).replace("_model", "").toUpperCase() + "…" : m.blocked ? "UNAVAILABLE"
                        : !card.installed ? (card.partial ? "PARTIAL · " + app.gib(m.present_bytes) : "NOT INSTALLED")
                        : card.integrity === "verified" ? (m.load_check_ok === true ? "VERIFIED · GPU LOAD OK" : m.load_check_ok === false ? "LOAD CHECK FAILED" : "VERIFIED") : "UNVERIFIED" }
                Item { Layout.fillWidth: true }
                Text { text: (m.file_count ? m.file_count + " files · " : "") + (m.size_hint || ""); font.family: Galaxy.mono; font.pixelSize: 10; color: Galaxy.muted }
            }
            TextHint { visible: !!(m.blocked || m.notes); text: m.blocked || m.notes || ""; color: m.blocked ? Galaxy.err : Galaxy.muted }
            TextHint { text: "Code: " + (m.code_license || "—") + "\nWeights: " + (m.weights_license || "—")
                color: /NON-COMMERCIAL|UNDOCUMENTED|research-use|OpenRAIL/.test(String(m.weights_license)) ? Galaxy.warn : Galaxy.dim }
            RowLayout { spacing: 8
                GalaxyButton { visible: !card.installed; primary: true; text: card.partial ? "Resume download" : "Install"
                    enabled: !m.blocked && !app.jobRunning; onClicked: app.confirmModel("install_model", m) }
                GalaxyButton { visible: card.installed; text: "Verify"; enabled: !app.jobRunning
                    hint: m.experimental ? "Re-hash every file against its pinned SHA-256" : "Re-hash every file and load it on the GPU"
                    onClicked: app.enqueue("verify_model", {model: card.name}) }
                GalaxyButton { visible: card.installed; text: "Repair"; enabled: !app.jobRunning
                    hint: "Re-check every file and download only what is missing or damaged"; onClicked: app.confirmModel("repair_model", m) }
                GalaxyButton { visible: card.installed || card.partial; text: "Reinstall"; enabled: !app.jobRunning && !m.blocked
                    hint: "Download everything again; installed files stay until their replacements verified"; onClicked: app.confirmModel("reinstall_model", m) }
                GalaxyButton { visible: card.installed || card.partial; text: "Remove"; danger: true; enabled: !app.jobRunning; onClicked: app.confirmModel("remove_model", m) }
            }

            // ---- worker setup of an experimental engine
            Rectangle { visible: card.isEngine; Layout.fillWidth: true; implicitHeight: 1; color: Galaxy.line }
            ColumnLayout { visible: card.isEngine; Layout.fillWidth: true; spacing: 8
                RowLayout { Layout.fillWidth: true; spacing: 8
                    Text { text: e.live ? "ENGINE SETUP" : "WORKER SETUP"; font.family: Galaxy.font; font.pixelSize: 9; font.letterSpacing: 1.4; color: Galaxy.label }
                    Item { Layout.fillWidth: true }
                    Text { text: e.available ? "ready" : (e.status || ""); font.family: Galaxy.font; font.pixelSize: 10; color: e.available ? Galaxy.ok : e.disabled_reason ? Galaxy.err : Galaxy.warn
                        elide: Text.ElideRight; Layout.maximumWidth: contents.width * 0.6 }
                }
                TextHint { text: "Clone " + (e.source || "") + " and check out " + String(e.revision || "").slice(0, 12)
                    + (e.live ? ". It runs in this Python environment." : ". It runs in its own Python environment, never in this one.") }
                GridLayout { Layout.fillWidth: true; columns: app.narrow ? 1 : 2; columnSpacing: 8; rowSpacing: 8
                    GalaxyField { id: checkoutField; placeholderText: "Source checkout, e.g. ~/src/" + String(e.source || "").split("/").pop() }
                    GalaxyField { id: pythonField; placeholderText: e.live ? "Python (empty = this environment)" : "Worker Python, e.g. ~/.venvs/deepfake-" + card.name + "/bin/python" }
                }
                RowLayout { spacing: 8
                    GalaxyButton { text: e.configured ? "Update setup" : "Save setup"; primary: !e.configured; enabled: checkoutField.text.length > 0 && !app.busy
                        onClicked: app.enqueue("configure_backend", {backend: card.name, checkout: checkoutField.text, python: pythonField.text}) }
                    GalaxyButton { visible: !!e.disabled_reason; text: "Enable with memory guard…"; onClicked: app.askEnable(card.name) }
                    TextHint { text: (e.checkout_ok ? "✓ source" : "✗ source") + "   " + (e.python_ok ? "✓ python" : "✗ python") + "   " + (card.installed ? "✓ weights" : "✗ weights") }
                }
                TextHint { visible: !!e.disabled_reason; color: Galaxy.err; text: "Disabled: " + (e.disabled_reason || "") }
                TextHint { visible: !!e.memory; text: e.memory ? "Memory guard: the worker is stopped above " + e.memory.limit_gb + " GiB of host RAM (" + e.memory.total_gb + " GiB installed, "
                    + e.memory.available_gb + " GiB free now; " + e.memory.method + ")." : "" }
            }
            // Fill the fields once from the saved setup (or a checkout found in a conventional place); never while typing.
            property bool seeded: false
            readonly property string savedCheckout: e.checkout || (e.suggested ? e.suggested.checkout : "") || ""
            readonly property string savedPython: e.python || (e.suggested ? e.suggested.python : "") || ""
            onSavedCheckoutChanged: seed()
            onSavedPythonChanged: seed()
            Component.onCompleted: seed()
            function seed() {
                if (seeded || !card.isEngine || (!savedCheckout && !savedPython)) return
                checkoutField.text = savedCheckout; pythonField.text = savedPython; seeded = true
            }
        }
    }
}
