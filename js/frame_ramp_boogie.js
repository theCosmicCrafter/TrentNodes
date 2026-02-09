import { app } from "/scripts/app.js";

/**
 * Frame Ramp Boogie - Dynamic Visibility + Live Curve Preview
 *
 * Features:
 * - Hides irrelevant controls based on current settings
 * - Live easing curve preview canvas that updates in real-time
 * - Bezier control point visualization with handle lines
 * - Canvas auto-resizes with the node
 *
 * Widget hiding uses the DA3 pattern (array removal) which
 * works reliably with ComfyUI's new frontend (v1.38+).
 */

// ── Easing math (ported from utils/easing.py) ─────────────

const EASING_FNS = {
    linear: (t) => t,
    ease_in: (t) => t * t,
    ease_out: (t) => 1.0 - (1.0 - t) * (1.0 - t),
    ease_in_out: (t) => 3.0 * t * t - 2.0 * t * t * t,
    smooth: (t) => t * t * (3.0 - 2.0 * t),
};

function cubicBezier(t, p1x, p1y, p2x, p2y) {
    let u = t;
    for (let i = 0; i < 8; i++) {
        const u2 = u * u;
        const u3 = u2 * u;
        const omu = 1.0 - u;
        const omu2 = omu * omu;
        const bx =
            3.0 * p1x * u * omu2 +
            3.0 * p2x * u2 * omu +
            u3;
        let dbx =
            3.0 * p1x * (1.0 - 4.0 * u + 3.0 * u2) +
            3.0 * p2x * (2.0 * u - 3.0 * u2) +
            3.0 * u2;
        dbx = Math.max(dbx, 1e-7);
        u = Math.max(
            0.0,
            Math.min(1.0, u - (bx - t) / dbx)
        );
    }
    const u2 = u * u;
    const u3 = u2 * u;
    const omu = 1.0 - u;
    const omu2 = omu * omu;
    const by =
        3.0 * p1y * u * omu2 +
        3.0 * p2y * u2 * omu +
        u3;
    return Math.max(0.0, Math.min(1.0, by));
}

// Must match Python _PRESET_VALUES exactly
const BEZIER_PRESETS = {
    ease: [0.25, 0.1, 0.25, 1.0],
    ease_in: [0.42, 0.0, 1.0, 1.0],
    ease_out: [0.0, 0.0, 0.58, 1.0],
    ease_in_out: [0.42, 0.0, 0.58, 1.0],
    sharp: [0.75, 0.0, 0.25, 1.0],
    gentle: [0.4, 0.0, 0.6, 1.0],
};

function evaluateEasing(t, easing, p1x, p1y, p2x, p2y) {
    if (easing === "cubic_bezier") {
        return cubicBezier(t, p1x, p1y, p2x, p2y);
    }
    const fn = EASING_FNS[easing] || EASING_FNS.linear;
    return fn(t);
}

// ── Curve preview drawing ─────────────────────────────────

function drawCurve(canvas, easing, p1x, p1y, p2x, p2y) {
    const ctx = canvas.getContext("2d");
    const dpr = window.devicePixelRatio || 1;
    const w = canvas.width / dpr;
    const h = canvas.height / dpr;
    const pad = 14;
    const gw = w - 2 * pad;
    const gh = h - 2 * pad;

    if (gw <= 0 || gh <= 0) return;

    ctx.save();
    ctx.scale(dpr, dpr);

    // Background
    ctx.fillStyle = "#0a1218";
    ctx.fillRect(0, 0, w, h);

    // Subtle grid (4x4)
    ctx.strokeStyle = "#1a2530";
    ctx.lineWidth = 0.5;
    for (let i = 1; i < 4; i++) {
        ctx.beginPath();
        ctx.moveTo(pad + (gw / 4) * i, pad);
        ctx.lineTo(pad + (gw / 4) * i, pad + gh);
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(pad, pad + (gh / 4) * i);
        ctx.lineTo(pad + gw, pad + (gh / 4) * i);
        ctx.stroke();
    }

    // Axes
    ctx.strokeStyle = "#2a3a48";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(pad, pad + gh);
    ctx.lineTo(pad + gw, pad + gh);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(pad, pad);
    ctx.lineTo(pad, pad + gh);
    ctx.stroke();

    // Diagonal reference (linear)
    if (easing !== "linear") {
        ctx.strokeStyle = "#1a2530";
        ctx.lineWidth = 0.5;
        ctx.setLineDash([4, 4]);
        ctx.beginPath();
        ctx.moveTo(pad, pad + gh);
        ctx.lineTo(pad + gw, pad);
        ctx.stroke();
        ctx.setLineDash([]);
    }

    // Bezier control point handles (behind curve)
    if (easing === "cubic_bezier") {
        const cp1x = pad + p1x * gw;
        const cp1y = pad + gh - p1y * gh;
        const cp2x = pad + p2x * gw;
        const cp2y = pad + gh - p2y * gh;

        ctx.strokeStyle = "rgba(255, 110, 64, 0.35)";
        ctx.lineWidth = 1;
        ctx.setLineDash([3, 3]);
        ctx.beginPath();
        ctx.moveTo(pad, pad + gh);
        ctx.lineTo(cp1x, cp1y);
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(cp2x, cp2y);
        ctx.lineTo(pad + gw, pad);
        ctx.stroke();
        ctx.setLineDash([]);
    }

    // Main curve
    ctx.strokeStyle = "#5c9cff";
    ctx.lineWidth = 2.5;
    ctx.lineJoin = "round";
    ctx.beginPath();
    const steps = 80;
    for (let i = 0; i <= steps; i++) {
        const t = i / steps;
        const v = evaluateEasing(
            t, easing, p1x, p1y, p2x, p2y
        );
        const x = pad + t * gw;
        const y = pad + gh - v * gh;
        if (i === 0) {
            ctx.moveTo(x, y);
        } else {
            ctx.lineTo(x, y);
        }
    }
    ctx.stroke();

    // Bezier control point dots (on top)
    if (easing === "cubic_bezier") {
        const cp1x = pad + p1x * gw;
        const cp1y = pad + gh - p1y * gh;
        const cp2x = pad + p2x * gw;
        const cp2y = pad + gh - p2y * gh;

        ctx.fillStyle = "#ff6e40";
        ctx.beginPath();
        ctx.arc(cp1x, cp1y, 4, 0, 2 * Math.PI);
        ctx.fill();
        ctx.beginPath();
        ctx.arc(cp2x, cp2y, 4, 0, 2 * Math.PI);
        ctx.fill();

        ctx.fillStyle = "#ff6e40";
        ctx.font = "bold 8px sans-serif";
        ctx.textAlign = "center";
        ctx.fillText("P1", cp1x, cp1y - 7);
        ctx.fillText("P2", cp2x, cp2y - 7);
    }

    // Axis labels
    ctx.fillStyle = "#556677";
    ctx.font = "8px sans-serif";
    ctx.textAlign = "center";
    ctx.fillText("time", pad + gw / 2, pad + gh + 12);
    ctx.textAlign = "right";
    ctx.fillText("0", pad - 3, pad + gh + 3);
    ctx.fillText("1", pad - 3, pad + 5);
    ctx.textAlign = "left";
    ctx.fillText("1", pad + gw + 2, pad + gh + 3);

    // Easing name label
    ctx.fillStyle = "#7a8a99";
    ctx.font = "9px sans-serif";
    ctx.textAlign = "right";
    const label = easing.replace(/_/g, " ");
    ctx.fillText(label, pad + gw, pad - 4);

    // "blend" Y-axis label
    ctx.fillStyle = "#556677";
    ctx.save();
    ctx.translate(5, pad + gh / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.textAlign = "center";
    ctx.font = "8px sans-serif";
    ctx.fillText("blend", 0, 0);
    ctx.restore();

    ctx.restore();
}

// ── Main extension ────────────────────────────────────────

app.registerExtension({
    name: "Trent.FrameRampBoogie",

    async nodeCreated(node) {
        if (node.constructor.comfyClass !== "FrameRampBoogie") {
            return;
        }

        // ── Store widget references ──────────────────
        // Required widgets stay in the array; optional
        // widgets get removed/re-added by visibility
        // logic. Direct refs let us access values and
        // hook callbacks regardless of array membership.

        const findWidget = (name) =>
            node.widgets?.find((w) => w.name === name);

        const optionalNames = [
            "region_size", "bezier_preset",
            "p1_x", "p1_y", "p2_x", "p2_y",
        ];
        const optRefs = {};
        for (const name of optionalNames) {
            const w = findWidget(name);
            if (w) optRefs[name] = w;
        }

        // ── Create curve preview widget ───────────────

        const container = document.createElement("div");
        container.style.cssText = `
            width: 100%;
            box-sizing: border-box;
            overflow: hidden;
        `;

        const canvas = document.createElement("canvas");
        const dpr = window.devicePixelRatio || 1;
        const canvasH = 110;
        canvas.width = 220 * dpr;
        canvas.height = canvasH * dpr;
        canvas.style.cssText = `
            display: block;
            width: 100%;
            height: ${canvasH}px;
            border-radius: 4px;
        `;
        container.appendChild(canvas);

        const previewWidget = node.addDOMWidget(
            "easing_preview",
            "customCanvas",
            container
        );
        previewWidget.computeSize = () => [
            220, canvasH + 8,
        ];
        previewWidget.serializeValue = () => undefined;

        // ── Curve redraw function ─────────────────────

        const redrawCurve = () => {
            // Sync canvas pixel width with container
            const rect =
                container.getBoundingClientRect();
            if (rect.width > 0) {
                const pw = Math.round(rect.width * dpr);
                if (canvas.width !== pw) {
                    canvas.width = pw;
                }
            }

            const easing =
                findWidget("easing")?.value || "linear";
            const preset =
                optRefs.bezier_preset?.value || "ease";

            let p1x = optRefs.p1_x?.value ?? 0.25;
            let p1y = optRefs.p1_y?.value ?? 0.1;
            let p2x = optRefs.p2_x?.value ?? 0.25;
            let p2y = optRefs.p2_y?.value ?? 1.0;

            if (
                easing === "cubic_bezier" &&
                preset !== "custom" &&
                BEZIER_PRESETS[preset]
            ) {
                [p1x, p1y, p2x, p2y] =
                    BEZIER_PRESETS[preset];
            }

            drawCurve(canvas, easing, p1x, p1y, p2x, p2y);
        };

        // ── Resize observer for canvas ────────────────

        const resizeObserver = new ResizeObserver(() => {
            redrawCurve();
        });
        resizeObserver.observe(container);

        // ── Widget visibility logic ───────────────────
        // Uses DA3 pattern: removes hidden widgets from
        // node.widgets entirely so the frontend cannot
        // render them. Re-inserts when they should show.

        const optSet = new Set(Object.values(optRefs));

        const updateVisibility = () => {
            const easing =
                findWidget("easing")?.value;
            const region =
                findWidget("target_region")?.value;
            const preset =
                optRefs.bezier_preset?.value;

            const isBezier =
                easing === "cubic_bezier";
            const isCustom = preset === "custom";
            const isFullBatch =
                region === "full_batch";

            // Build list of widgets that should show
            // (in display order)
            const toShow = [];
            if (!isFullBatch && optRefs.region_size) {
                toShow.push(optRefs.region_size);
            }
            if (isBezier && optRefs.bezier_preset) {
                toShow.push(optRefs.bezier_preset);
            }
            if (isBezier && isCustom) {
                if (optRefs.p1_x) toShow.push(optRefs.p1_x);
                if (optRefs.p1_y) toShow.push(optRefs.p1_y);
                if (optRefs.p2_x) toShow.push(optRefs.p2_x);
                if (optRefs.p2_y) toShow.push(optRefs.p2_y);
            }

            // Remove ALL optional widgets from array
            node.widgets = node.widgets.filter(
                (w) => !optSet.has(w)
            );

            // Re-insert visible ones before preview
            const pvIdx =
                node.widgets.indexOf(previewWidget);
            const insertAt =
                pvIdx >= 0 ? pvIdx : node.widgets.length;

            for (let i = 0; i < toShow.length; i++) {
                node.widgets.splice(
                    insertAt + i, 0, toShow[i]
                );
            }

            // Redraw curve
            redrawCurve();

            // Resize node to fit (keep width, adjust
            // height)
            requestAnimationFrame(() => {
                const newSize = node.computeSize();
                node.setSize([
                    node.size[0], newSize[1],
                ]);
                if (node.setDirtyCanvas) {
                    node.setDirtyCanvas(true, true);
                }
                if (app.graph) {
                    app.graph.setDirtyCanvas(true, true);
                }
            });
        };

        // ── Hook widget callbacks ─────────────────────
        // Required widgets use findWidget (always in
        // array). Optional widgets use stored refs.

        const easingW = findWidget("easing");
        if (easingW) {
            const orig = easingW.callback;
            easingW.callback = function (value) {
                if (orig) orig.apply(this, arguments);
                setTimeout(updateVisibility, 50);
            };
        }

        const regionW = findWidget("target_region");
        if (regionW) {
            const orig = regionW.callback;
            regionW.callback = function (value) {
                if (orig) orig.apply(this, arguments);
                setTimeout(updateVisibility, 50);
            };
        }

        if (optRefs.bezier_preset) {
            const orig = optRefs.bezier_preset.callback;
            optRefs.bezier_preset.callback =
                function (value) {
                    if (orig) {
                        orig.apply(this, arguments);
                    }
                    setTimeout(updateVisibility, 50);
                };
        }

        // Hook bezier sliders for live curve updates
        for (const name of [
            "p1_x", "p1_y", "p2_x", "p2_y",
        ]) {
            const widget = optRefs[name];
            if (widget) {
                const orig = widget.callback;
                widget.callback = function (value) {
                    if (orig) {
                        orig.apply(this, arguments);
                    }
                    setTimeout(redrawCurve, 10);
                };
            }
        }

        // ── Workflow load restore ─────────────────────

        const origConfigure = node.onConfigure;
        node.onConfigure = function (config) {
            if (origConfigure) {
                origConfigure.apply(this, arguments);
            }
            setTimeout(updateVisibility, 100);
        };

        // ── Initial update ────────────────────────────

        updateVisibility();
        setTimeout(updateVisibility, 100);
        setTimeout(updateVisibility, 500);
    },
});
