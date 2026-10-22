import { app } from '../../scripts/app.js';

// ============================================================
//  LoRA Step Schedule — 节点下方的步强度曲线面板
//
//  数据源与后端 nodes/lora_step_schedule.py 完全一致：
//    1. strength_list 非空 -> 手动列表，面板显示它；
//    2. strength_list 为空 -> 按 start/end/curve/points 生成。
//  拖动圆点会把当前列表写回 strength_list（实时），从此进入手动模式；
//  “清空”恢复曲线生成，“曲线生成”把当前曲线落成显式列表。
//  虚线是 start/end/curve/points 的参考曲线，手动模式下也能看到曲线形状。
//
//  实际采样步数与列表长度的关系由后端处理：步数多于列表时末尾保持，
//  少于列表时取前缀；面板的 x 轴是“调度点序号”，不是采样步号。
// ============================================================

const NODE_ID = 'PSLLoRAStepSchedule';
const PANEL_MIN_H = 205;
const CURVE_LABELS = {
    linear: '线性',
    ease_in: '二次加速',
    ease_out: '二次减速',
    ease_in_out: 'S 曲线',
    cosine: '余弦',
};

let stylesInjected = false;

function injectStyles() {
    if (stylesInjected || document.getElementById('psl-lsc-styles')) return;
    stylesInjected = true;
    const style = document.createElement('style');
    style.id = 'psl-lsc-styles';
    style.textContent = `
        .psl-lsc-root {
            display: flex;
            flex-direction: column;
            gap: 4px;
            padding: 4px 6px;
            box-sizing: border-box;
            width: 100%;
            height: 100%;
            font: 11px/1.25 monospace;
        }
        .psl-lsc-bar {
            display: flex;
            align-items: center;
            gap: 6px;
            min-height: 18px;
            flex: 0 0 auto;
        }
        .psl-lsc-status {
            flex: 1 1 auto;
            min-width: 0;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
            color: #8ab4d8;
        }
        .psl-lsc-btn {
            flex: 0 0 auto;
            background: #3d3d3d;
            color: #ddd;
            border: 1px solid #555;
            border-radius: 3px;
            font: 11px/1.2 monospace;
            padding: 2px 6px;
            cursor: pointer;
        }
        .psl-lsc-btn:hover { filter: brightness(1.2); }
        .psl-lsc-btn:active { filter: brightness(0.9); }
        .psl-lsc-canvas {
            flex: 1 1 auto;
            min-height: 120px;
            width: 100%;
            display: block;
            background: #141414;
            border: 1px solid #333;
            border-radius: 4px;
            touch-action: none;
            cursor: crosshair;
        }
        .psl-lsc-hint {
            flex: 0 0 auto;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
            color: #666;
            font-size: 10px;
        }
    `;
    document.head.appendChild(style);
}

function trimNumber(value, digits = 3) {
    const scale = 10 ** digits;
    let rounded = Math.round(value * scale) / scale;
    if (Object.is(rounded, -0)) rounded = 0;
    return String(rounded);
}

function parseList(text) {
    let raw = typeof text === 'string' ? text.trim() : String(text ?? '').trim();
    if (!raw) return { values: [], error: null };
    if (raw[0] === '[' || raw[0] === '(') raw = raw.slice(1);
    if (raw && (raw[raw.length - 1] === ']' || raw[raw.length - 1] === ')')) {
        raw = raw.slice(0, -1);
    }
    const tokens = raw.split(/[,;\s，、]+/).filter(Boolean);
    const values = [];
    for (const token of tokens) {
        const value = Number(token);
        if (!Number.isFinite(value)) {
            return { values: [], error: `无法解析「${token}」` };
        }
        values.push(value);
    }
    return { values, error: null };
}

function curveValue(curve, t) {
    const x = Math.min(1, Math.max(0, t));
    switch (curve) {
        case 'ease_in':
            return x * x;
        case 'ease_out':
            return 1 - (1 - x) * (1 - x);
        case 'ease_in_out':
            return x * x * (3 - 2 * x);
        case 'cosine':
            return 0.5 - 0.5 * Math.cos(Math.PI * x);
        case 'linear':
        default:
            return x;
    }
}

function generateSchedule(start, end, curve, points) {
    const count = Math.max(2, Math.min(200, Math.round(Number(points) || 20)));
    const span = count - 1;
    const values = [];
    for (let i = 0; i < count; i += 1) {
        const t = i / span;
        values.push(start + (end - start) * curveValue(curve, t));
    }
    return values;
}

function computeRange(values, reference) {
    const all = [...values, ...reference].filter((v) => Number.isFinite(v));
    if (!all.length) return [-1, 1];
    let lo = Math.min(0, ...all);
    let hi = Math.max(0, ...all);
    if (hi - lo < 0.4) {
        const mid = (hi + lo) / 2;
        lo = mid - 0.2;
        hi = mid + 0.2;
    }
    const pad = Math.max(0.02, (hi - lo) * 0.12);
    return [lo - pad, hi + pad];
}

function formatList(values) {
    return values.map((value) => trimNumber(value)).join(', ');
}

app.registerExtension({
    name: 'Comfy.PersonalUtilities.LoRAStepSchedule',

    async nodeCreated(node) {
        if (node.comfyClass !== NODE_ID) return;
        injectStyles();

        const startW = node.widgets?.find((w) => w.name === 'start_strength');
        const endW = node.widgets?.find((w) => w.name === 'end_strength');
        const curveW = node.widgets?.find((w) => w.name === 'curve');
        const pointsW = node.widgets?.find((w) => w.name === 'points');
        const listW = node.widgets?.find((w) => w.name === 'strength_list');
        const missing = [
            ['start_strength', startW],
            ['end_strength', endW],
            ['curve', curveW],
            ['points', pointsW],
            ['strength_list', listW],
        ].filter(([, widget]) => !widget);
        if (missing.length) {
            console.error(
                '[PSLLoRAStepSchedule] 缺少 widget:',
                missing.map(([name]) => name).join(', '),
                '；请重启 ComfyUI 后端并强制刷新页面。'
            );
            return;
        }

        const root = document.createElement('div');
        root.className = 'psl-lsc-root';
        root.style.setProperty('--comfy-widget-min-height', `${PANEL_MIN_H}px`);

        const bar = document.createElement('div');
        bar.className = 'psl-lsc-bar';
        const statusEl = document.createElement('span');
        statusEl.className = 'psl-lsc-status';

        function makeButton(label, title) {
            const button = document.createElement('button');
            button.type = 'button';
            button.className = 'psl-lsc-btn';
            button.textContent = label;
            button.title = title;
            for (const eventName of [
                'pointerdown', 'mousedown', 'pointerup', 'mouseup',
                'click', 'dblclick', 'wheel', 'contextmenu',
            ]) {
                button.addEventListener(eventName, (event) => event.stopPropagation());
            }
            return button;
        }

        const generateBtn = makeButton('曲线生成', '按开始/结束强度、曲线和点数生成列表，写入强度列表');
        const clearBtn = makeButton('清空', '清空强度列表，改为按曲线自动生成');

        bar.append(statusEl, generateBtn, clearBtn);

        const canvas = document.createElement('canvas');
        canvas.className = 'psl-lsc-canvas';
        canvas.title = '拖动圆点调整该调度点的强度；可用“曲线生成”或“清空”重置';

        const hint = document.createElement('div');
        hint.className = 'psl-lsc-hint';
        hint.textContent = '拖动圆点改强度 · 步数多则末尾保持，少则取前缀';

        root.append(bar, canvas, hint);

        node.addDOMWidget('psl_lora_schedule_curve', 'PSL_LORA_SCHEDULE_CURVE', root, {
            serialize: false,
            hideOnZoom: false,
            margin: 6,
            getMinHeight: () => PANEL_MIN_H,
        });

        const ctx = canvas.getContext('2d');
        let drag = null;             // { index, range: [lo, hi], values: number[] }
        let hoverIndex = -1;
        let drawScheduled = false;
        let fitScheduled = false;
        let lastFitHeight = 0;

        function readInputs() {
            const start = Number(startW.value);
            const end = Number(endW.value);
            const points = Number(pointsW.value);
            return {
                start: Number.isFinite(start) ? start : 0,
                end: Number.isFinite(end) ? end : 0,
                curve: String(curveW.value || 'linear'),
                points: Number.isFinite(points) ? points : 20,
            };
        }

        function referenceSchedule() {
            const input = readInputs();
            return generateSchedule(input.start, input.end, input.curve, input.points);
        }

        function manualList() {
            const text = typeof listW.value === 'string' ? listW.value : String(listW.value ?? '');
            if (!text.trim()) return { manual: false, values: [], error: null };
            const parsed = parseList(text);
            return { manual: true, values: parsed.values, error: parsed.error };
        }

        function effectiveSchedule() {
            const reference = referenceSchedule();
            const manual = manualList();
            if (manual.error) {
                return { reference, values: [], error: manual.error, manual: true };
            }
            if (manual.manual) {
                return { reference, values: manual.values, error: null, manual: true };
            }
            return { reference, values: reference, error: null, manual: false };
        }

        function layout() {
            const width = Math.max(1, canvas.clientWidth);
            const height = Math.max(1, canvas.clientHeight);
            const left = 34;
            const right = 10;
            const top = 10;
            const bottom = 18;
            return {
                width,
                height,
                left,
                right,
                top,
                bottom,
                plotW: Math.max(1, width - left - right),
                plotH: Math.max(1, height - top - bottom),
            };
        }

        function xFor(index, count, geo) {
            if (count <= 1) return geo.left + geo.plotW / 2;
            return geo.left + (index / (count - 1)) * geo.plotW;
        }

        function yFor(value, range, geo) {
            const [lo, hi] = range;
            const span = hi - lo || 1;
            return geo.top + ((hi - value) / span) * geo.plotH;
        }

        function valueFor(y, range, geo) {
            const [lo, hi] = range;
            const t = Math.min(1, Math.max(0, (y - geo.top) / geo.plotH));
            return hi - t * (hi - lo);
        }

        function localPoint(event, geo) {
            const rect = canvas.getBoundingClientRect();
            const scaleX = geo.width / Math.max(1, rect.width);
            const scaleY = geo.height / Math.max(1, rect.height);
            return {
                x: (event.clientX - rect.left) * scaleX,
                y: (event.clientY - rect.top) * scaleY,
            };
        }

        function hitTest(x, y, values, range, geo) {
            if (!values.length) return -1;
            let best = -1;
            let bestDx = Infinity;
            for (let i = 0; i < values.length; i += 1) {
                const dx = Math.abs(xFor(i, values.length, geo) - x);
                const dy = Math.abs(yFor(values[i], range, geo) - y);
                if (dx <= 8 && dy <= 24 && dx < bestDx) {
                    best = i;
                    bestDx = dx;
                }
            }
            return best;
        }

        function writeValues(values) {
            const text = formatList(values);
            if (listW.value !== text) {
                listW.value = text;
                node.setDirtyCanvas?.(true, true);
            }
        }

        function strokePolyline(values, range, geo, color, width, dash) {
            if (!values.length) return;
            ctx.save();
            ctx.beginPath();
            ctx.setLineDash(dash || []);
            ctx.strokeStyle = color;
            ctx.lineWidth = width;
            if (values.length === 1) {
                const x = xFor(0, 1, geo);
                const y = yFor(values[0], range, geo);
                ctx.moveTo(geo.left, y);
                ctx.lineTo(geo.left + geo.plotW, y);
            } else {
                for (let i = 0; i < values.length; i += 1) {
                    const x = xFor(i, values.length, geo);
                    const y = yFor(values[i], range, geo);
                    if (i === 0) ctx.moveTo(x, y);
                    else ctx.lineTo(x, y);
                }
            }
            ctx.stroke();
            ctx.restore();
        }

        function draw() {
            drawScheduled = false;
            if (!node.graph) return;
            const geo = layout();
            if (geo.width <= 1 || geo.height <= 1) return;

            const state = effectiveSchedule();
            const range = drag
                ? drag.range
                : computeRange(state.values, state.reference);

            const dpr = window.devicePixelRatio || 1;
            const pixelW = Math.max(1, Math.round(geo.width * dpr));
            const pixelH = Math.max(1, Math.round(geo.height * dpr));
            if (canvas.width !== pixelW || canvas.height !== pixelH) {
                canvas.width = pixelW;
                canvas.height = pixelH;
            }
            ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
            ctx.clearRect(0, 0, geo.width, geo.height);

            // horizontal grid + y labels
            ctx.font = '10px monospace';
            ctx.textBaseline = 'middle';
            ctx.textAlign = 'right';
            for (let i = 0; i <= 4; i += 1) {
                const t = i / 4;
                const y = geo.top + t * geo.plotH;
                ctx.beginPath();
                ctx.strokeStyle = '#242424';
                ctx.lineWidth = 1;
                ctx.moveTo(geo.left, y);
                ctx.lineTo(geo.left + geo.plotW, y);
                ctx.stroke();
                const value = range[1] - t * (range[1] - range[0]);
                ctx.fillStyle = '#666';
                ctx.fillText(trimNumber(value, 2), geo.left - 4, y);
            }

            // x labels
            ctx.textAlign = 'center';
            ctx.textBaseline = 'top';
            ctx.fillStyle = '#666';
            const count = state.values.length || state.reference.length;
            if (count > 1) {
                ctx.fillText('0', geo.left, geo.top + geo.plotH + 4);
                ctx.fillText(String(count - 1), geo.left + geo.plotW, geo.top + geo.plotH + 4);
                if (count > 2) {
                    const mid = Math.floor((count - 1) / 2);
                    ctx.fillText(String(mid), xFor(mid, count, geo), geo.top + geo.plotH + 4);
                }
            } else {
                ctx.fillText('0', geo.left + geo.plotW / 2, geo.top + geo.plotH + 4);
            }

            // zero line
            if (range[0] < 0 && range[1] > 0) {
                const zeroY = yFor(0, range, geo);
                ctx.beginPath();
                ctx.strokeStyle = '#3a3a3a';
                ctx.lineWidth = 1;
                ctx.moveTo(geo.left, zeroY);
                ctx.lineTo(geo.left + geo.plotW, zeroY);
                ctx.stroke();
            }

            // reference curve (dashed); in auto mode the solid line already is it
            if (state.manual) {
                strokePolyline(
                    state.reference, range, geo,
                    '#a5793a', 1.2, [4, 4]
                );
            }

            if (state.error) {
                statusEl.textContent = `强度列表解析失败：${state.error}`;
                statusEl.style.color = '#e07a7a';
                statusEl.title = '强度列表只能包含数字，用逗号/空格/换行分隔';
                return;
            }

            // actual values (solid)
            strokePolyline(state.values, range, geo, '#4ea3e8', 1.8, []);

            const showDots = state.values.length <= 300;
            const radius = state.values.length > 64 ? 2 : 3.5;
            for (let i = 0; i < state.values.length; i += 1) {
                const x = xFor(i, state.values.length, geo);
                const y = yFor(state.values[i], range, geo);
                if (showDots) {
                    ctx.beginPath();
                    ctx.fillStyle = (drag && drag.index === i) || hoverIndex === i
                        ? '#ffd479'
                        : '#7fc4ff';
                    ctx.arc(x, y, radius, 0, Math.PI * 2);
                    ctx.fill();
                }
                if (drag && drag.index === i) {
                    const label = trimNumber(state.values[i], 4);
                    ctx.font = '11px monospace';
                    ctx.textAlign = 'left';
                    ctx.textBaseline = 'middle';
                    ctx.lineWidth = 3;
                    ctx.strokeStyle = '#141414';
                    ctx.strokeText(label, x + 7, y - 8);
                    ctx.fillStyle = '#ffd479';
                    ctx.fillText(label, x + 7, y - 8);
                }
            }

            const curveName = readInputs().curve;
            const curveLabel = CURVE_LABELS[curveName] || curveName;
            if (state.manual) {
                const pointCount = state.values.length;
                const refCount = state.reference.length;
                statusEl.textContent = `手动 · ${pointCount} 点`;
                statusEl.style.color = '#d0a04a';
                statusEl.title = `强度列表来自手动输入；拖动圆点会实时写回。`
                    + `曲线参考是 ${refCount} 点，点“清空”恢复自动生成。`;
            } else {
                statusEl.textContent = `${curveLabel} · ${state.values.length} 点`;
                statusEl.style.color = '#8ab4d8';
                statusEl.title = '强度列表为空时按开始/结束强度和曲线生成；拖动圆点会写入手动列表';
            }
        }

        function requestDraw() {
            if (drawScheduled) return;
            drawScheduled = true;
            requestAnimationFrame(draw);
        }

        function fitNodeHeight() {
            if (fitScheduled) return;
            fitScheduled = true;
            requestAnimationFrame(() => {
                fitScheduled = false;
                if (!node.graph) return;
                const width = node.size?.[0];
                if (!Number.isFinite(width) || width <= 0) return;
                let total = null;
                try {
                    total = node.computeSize?.([width, 0]);
                } catch (error) {
                    console.warn('[PSLLoRAStepSchedule] computeSize 失败', error);
                    return;
                }
                const height = Array.isArray(total) ? total[1] : null;
                if (!Number.isFinite(height) || height <= 0) return;
                if (height <= (lastFitHeight || 0)) return;
                if (height <= (node.size?.[1] || 0)) return;
                lastFitHeight = height;
                node.size = [width, height];
                node.graph.setDirtyCanvas(true, true);
                requestDraw();
            });
        }

        function hookWidget(widget) {
            const previous = widget.callback;
            widget.callback = function (...args) {
                let result;
                if (previous) {
                    try {
                        result = previous.apply(this, args);
                    } catch (error) {
                        console.error('[PSLLoRAStepSchedule] widget callback 失败', error);
                    }
                }
                requestDraw();
                return result;
            };
        }

        for (const widget of [startW, endW, curveW, pointsW, listW]) {
            hookWidget(widget);
        }

        generateBtn.addEventListener('click', (event) => {
            event.preventDefault();
            event.stopPropagation();
            drag = null;
            hoverIndex = -1;
            writeValues(referenceSchedule());
            requestDraw();
        });

        clearBtn.addEventListener('click', (event) => {
            event.preventDefault();
            event.stopPropagation();
            drag = null;
            hoverIndex = -1;
            if (listW.value !== '') {
                listW.value = '';
                node.setDirtyCanvas?.(true, true);
            }
            requestDraw();
        });

        function localEventPoint(event) {
            return localPoint(event, layout());
        }

        function onPointerDown(event) {
            if (event.button !== undefined && event.button !== 0) return;
            event.preventDefault();
            event.stopPropagation();

            const state = effectiveSchedule();
            if (state.error) return;
            const geo = layout();
            const point = localEventPoint(event);
            const range = computeRange(state.values, state.reference);
            const index = hitTest(point.x, point.y, state.values, range, geo);
            if (index < 0) return;

            let values = state.values;
            if (!state.manual) {
                // First drag materializes the generated curve into an explicit list.
                values = state.reference.slice();
                writeValues(values);
            }

            drag = { index, range, values: values.slice() };
            hoverIndex = index;
            try {
                canvas.setPointerCapture?.(event.pointerId);
            } catch (error) {
                console.warn('[PSLLoRAStepSchedule] setPointerCapture 失败', error);
            }
            canvas.style.cursor = 'grabbing';
            requestDraw();
        }

        function onPointerMove(event) {
            event.stopPropagation();
            const geo = layout();
            const point = localEventPoint(event);

            if (drag) {
                event.preventDefault();
                const value = valueFor(point.y, drag.range, geo);
                drag.values[drag.index] = Math.round(value * 1000) / 1000;
                writeValues(drag.values);
                requestDraw();
                return;
            }

            const state = effectiveSchedule();
            if (state.error) {
                if (hoverIndex !== -1) {
                    hoverIndex = -1;
                    canvas.style.cursor = 'crosshair';
                    requestDraw();
                }
                return;
            }
            const range = computeRange(state.values, state.reference);
            const index = hitTest(point.x, point.y, state.values, range, geo);
            if (index !== hoverIndex) {
                hoverIndex = index;
                canvas.style.cursor = index >= 0 ? 'grab' : 'crosshair';
                requestDraw();
            }
        }

        function endDrag(event) {
            if (!drag) return;
            event.stopPropagation();
            drag = null;
            hoverIndex = -1;
            canvas.style.cursor = 'crosshair';
            try {
                if (event.pointerId !== undefined && canvas.hasPointerCapture?.(event.pointerId)) {
                    canvas.releasePointerCapture(event.pointerId);
                }
            } catch (error) {
                console.warn('[PSLLoRAStepSchedule] releasePointerCapture 失败', error);
            }
            requestDraw();
        }

        for (const eventName of [
            'pointerdown', 'pointermove', 'pointerup', 'pointercancel',
            'mousedown', 'mouseup', 'dblclick', 'wheel', 'contextmenu',
        ]) {
            canvas.addEventListener(eventName, (event) => event.stopPropagation());
        }
        canvas.addEventListener('pointerdown', onPointerDown);
        canvas.addEventListener('pointermove', onPointerMove);
        canvas.addEventListener('pointerup', endDrag);
        canvas.addEventListener('pointercancel', endDrag);
        canvas.addEventListener('pointerleave', () => {
            if (drag) return;
            if (hoverIndex !== -1) {
                hoverIndex = -1;
                canvas.style.cursor = 'crosshair';
                requestDraw();
            }
        });
        canvas.addEventListener('wheel', (event) => event.preventDefault(), { passive: false });

        const previousResize = node.onResize;
        node.onResize = function (...args) {
            const result = previousResize?.apply(this, args);
            requestDraw();
            return result;
        };

        const previousConfigure = node.onConfigure;
        node.onConfigure = function (...args) {
            const result = previousConfigure?.apply(this, args);
            requestAnimationFrame(() => {
                fitNodeHeight();
                requestDraw();
            });
            return result;
        };

        let observer = null;
        if (typeof ResizeObserver !== 'undefined') {
            observer = new ResizeObserver(() => requestDraw());
            observer.observe(canvas);
        }

        const previousRemoved = node.onRemoved;
        node.onRemoved = function (...args) {
            observer?.disconnect();
            observer = null;
            return previousRemoved?.apply(this, args);
        };

        requestAnimationFrame(() => {
            fitNodeHeight();
            requestDraw();
        });
    },
});
