import QtQuick

// Purple Galaxy window backdrop, shared by every rice widget (canonical copy: ~/.config/quickshell/galaxy/Nebula.qml;
// the other widget folders hold identical copies). Solid #0e0a1c fill + soft nebula clouds + sparse stars + a
// luminous rim, all painted into ONE Canvas raster that repaints only on resize. No Gradient on a rounded
// Rectangle anywhere: the scene graph interpolates those per triangle and shows a diagonal seam.
Item {
    id: neb
    property real radius: 22
    property int seed: 7               // vary per widget so siblings don't share a sky
    property real stars: 1.0           // star density multiplier
    property real glow: 1.0            // nebula strength multiplier (0 = plain fill)
    property color fill: "#0e0a1c"
    property bool rim: true
    property bool hot: false           // brighter rim (focus / live state)
    onHotChanged: sky.requestPaint()
    onGlowChanged: sky.requestPaint()
    onRadiusChanged: sky.requestPaint()

    Canvas {
        id: sky
        anchors.fill: parent
        renderStrategy: Canvas.Cooperative
        onWidthChanged: requestPaint()
        onHeightChanged: requestPaint()
        onPaint: {
            const c = getContext("2d"); c.reset()
            const w = width, h = height, r = Math.min(neb.radius, w / 2, h / 2)
            if (w < 4 || h < 4) return
            c.beginPath(); c.roundedRect(0.5, 0.5, w - 1, h - 1, r, r); c.closePath()
            c.save(); c.clip()
            c.fillStyle = neb.fill; c.fillRect(0, 0, w, h)

            let s = neb.seed * 7919 + 13
            const rnd = () => { s = (s * 16807) % 2147483647; return s / 2147483647 }
            const big = Math.max(w, h), k = neb.glow
            const blob = (fx, fy, fr, rgb, a) => {
                if (a * k <= 0) return
                const x = w * fx, y = h * fy, rr = big * fr
                const g = c.createRadialGradient(x, y, 0, x, y, rr)
                g.addColorStop(0, "rgba(" + rgb + "," + (a * k).toFixed(3) + ")")
                g.addColorStop(0.55, "rgba(" + rgb + "," + (a * k * 0.35).toFixed(3) + ")")
                g.addColorStop(1, "rgba(" + rgb + ",0)")
                c.fillStyle = g; c.fillRect(0, 0, w, h)
            }
            // main clouds: violet core top-left, lilac bottom-right, a faint magenta wisp, a deep indigo sink
            blob(0.08 + rnd() * 0.15, 0.02 + rnd() * 0.12, 0.62, "124,77,255", 0.15)
            blob(0.80 + rnd() * 0.15, 0.82 + rnd() * 0.15, 0.55, "180,107,255", 0.10)
            blob(0.70 + rnd() * 0.2, 0.10 + rnd() * 0.2, 0.32, "255,107,213", 0.055)
            blob(0.25 + rnd() * 0.3, 0.75 + rnd() * 0.2, 0.40, "72,52,190", 0.08)
            // dust lane: small puffs along a gentle diagonal
            const lane = 7, y0 = 0.25 + rnd() * 0.2, y1 = 0.6 + rnd() * 0.2
            for (let i = 0; i < lane; ++i) {
                const t = i / (lane - 1)
                blob(t, y0 + (y1 - y0) * t + (rnd() - 0.5) * 0.08, 0.10 + rnd() * 0.08, i % 2 ? "180,107,255" : "212,184,255", 0.035)
            }
            // stars
            const n = Math.round(w * h / 7000 * neb.stars)
            for (let i = 0; i < n; ++i) {
                const x = rnd() * w, y = rnd() * h, roll = rnd()
                if (roll > 0.985) {            // sparkle: soft halo + thin cross
                    const hg = c.createRadialGradient(x, y, 0, x, y, 7)
                    hg.addColorStop(0, "rgba(212,184,255,0.45)"); hg.addColorStop(1, "rgba(212,184,255,0)")
                    c.fillStyle = hg; c.fillRect(x - 7, y - 7, 14, 14)
                    c.fillStyle = "rgba(242,240,255,0.55)"
                    c.fillRect(x - 4, y - 0.5, 8, 1); c.fillRect(x - 0.5, y - 4, 1, 8)
                } else if (roll > 0.93) {
                    c.fillStyle = "rgba(212,184,255," + (0.35 + rnd() * 0.3).toFixed(2) + ")"
                    c.fillRect(x, y, 1.6, 1.6)
                } else {
                    c.fillStyle = "rgba(242,240,255," + (0.08 + rnd() * 0.22).toFixed(2) + ")"
                    c.fillRect(x, y, 1, 1)
                }
            }
            c.restore()
            if (!neb.rim) return
            // rim: base edge colour, lit from the top-left like the nebula core
            c.beginPath(); c.roundedRect(0.5, 0.5, w - 1, h - 1, r, r); c.closePath()
            c.lineWidth = 1
            c.strokeStyle = neb.hot ? "#5b3d99" : "#33295a"; c.stroke()
            const lg = c.createRadialGradient(w * 0.22, 0, 0, w * 0.22, 0, Math.max(w, h) * 0.75)
            lg.addColorStop(0, neb.hot ? "rgba(212,184,255,0.85)" : "rgba(180,140,255,0.6)")
            lg.addColorStop(0.5, "rgba(124,77,255,0.25)")
            lg.addColorStop(1, "rgba(124,77,255,0)")
            c.strokeStyle = lg; c.stroke()
        }
    }
}
