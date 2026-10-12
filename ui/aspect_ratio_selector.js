import { app } from '../../scripts/app.js';

// ============================================================
//  Aspect Ratio Selector — 可视化分辨率选择器
//  增量拖拽无漂移 + 自定义渐入下拉
// ============================================================

const PRESETS = [
    { label: '自由比例', w: null, h: null },
    { label: '1:1 正方形',  w: 1, h: 1 },
    { label: '4:3 传统',    w: 4, h: 3 },
    { label: '3:2 照片',    w: 3, h: 2 },
    { label: '16:9 宽屏',   w: 16, h: 9 },
    { label: '21:9 超宽',   w: 21, h: 9 },
    { label: '9:16 竖屏',   w: 9, h: 16 },
    { label: '2:3 肖像',    w: 2, h: 3 },
    { label: '3:4',         w: 3, h: 4 },
];

const PAD = 16;
const HANDLE = 14;
const MIN = 24;

function gcd(a, b) {
    a = Math.abs(Math.round(a));
    b = Math.abs(Math.round(b));
    while (b) { [a, b] = [b, a % b]; }
    return a || 1;
}

function computeResolution(totalMp, minStep, aspectW, aspectH) {
    const totalPx = totalMp * 10000;
    const ratio = aspectW / aspectH;
    const baseH = Math.sqrt(totalPx / ratio);
    const baseW = totalPx / baseH;
    let w = Math.round(baseW / minStep) * minStep;
    let h = Math.round(baseH / minStep) * minStep;
    w = Math.max(w, minStep);
    h = Math.max(h, minStep);
    return { w, h };
}

// ── CSS ──
function injectStyles() {
    if (document.getElementById('ars-styles')) return;
    const s = document.createElement('style');
    s.id = 'ars-styles';
    s.textContent = `
        .ars-container {
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 6px;
            width: 100%;
            height: 100%;
            padding: 4px 6px;
            background: #1f1f1f;
            border-radius: 6px;
            margin-top: 5px;
            box-sizing: border-box;
        }
        .ars-dropdown-wrap {
            position: relative;
            width: 30%;
            flex-shrink: 0;
            z-index: 100;
        }
        .ars-dropdown-btn {
            width: 100%;
            background: #2a2a2a;
            color: #eee;
            border: 1px solid #444;
            border-radius: 4px;
            padding: 4px 4px;
            font-size: 11px;
            text-align: center;
            cursor: pointer;
            box-sizing: border-box;
            user-select: none;
            transition: border-color 0.2s, box-shadow 0.2s;
        }
        .ars-dropdown-btn:hover { border-color: #1e90ff; }
        .ars-dropdown-btn.open { border-color: #1e90ff; box-shadow: 0 0 8px rgba(30,144,255,0.3); }
        .ars-dropdown-btn::after {
            content: ' ▾';
            font-size: 10px;
            opacity: 0.6;
        }
        .ars-dropdown-options {
            position: absolute;
            top: 100%;
            left: 0;
            right: 0;
            margin-top: 2px;
            background: #2a2a2a;
            border: 1px solid #444;
            border-radius: 4px;
            overflow: hidden;
            opacity: 0;
            pointer-events: none;
            transform: translateY(-4px);
            transition: opacity 0.18s ease-out, transform 0.18s ease-out;
        }
        .ars-dropdown-options.open {
            opacity: 1;
            pointer-events: auto;
            transform: translateY(0);
        }
        .ars-option {
            padding: 5px 12px;
            font-size: 13px;
            color: #ccc;
            cursor: pointer;
            transition: background 0.12s, color 0.12s;
            text-align: center;
        }
        .ars-option:hover { background: rgba(30,144,255,0.2); color: #fff; }
        .ars-option.selected { color: #1e90ff; font-weight: bold; }
        .ars-canvas {
            position: relative;
            flex: 1;
            width: 100%;
            background-color: #161616;
            background-image:
                linear-gradient(rgba(255,255,255,0.035) 1px, transparent 1px),
                linear-gradient(90deg, rgba(255,255,255,0.035) 1px, transparent 1px);
            background-size: 16px 16px;
            border: 1px solid #333;
            border-radius: 4px;
            overflow: hidden;
        }
        .ars-rect {
            position: absolute;
            left: ${PAD}px;
            top: ${PAD}px;
            pointer-events: none;
            transition: width 0.15s ease-out, height 0.15s ease-out;
        }
        .ars-rect.free {
            background: rgba(30,144,255,0.08);
            border: 1.5px solid rgba(30,144,255,0.5);
        }
        .ars-rect.locked {
            background: rgba(30,144,255,0.14);
            border: 2px solid #1e90ff;
        }
        .ars-handle {
            position: absolute;
            width: ${HANDLE}px;
            height: ${HANDLE}px;
            background: #1e90ff;
            border: 2px solid #fff;
            border-radius: 3px;
            z-index: 10;
            box-sizing: border-box;
            cursor: nwse-resize;
            transition: left 0.15s ease-out, top 0.15s ease-out, background 0.1s;
        }
        .ars-handle:hover { background: #4aa8ff; }
        .ars-handle.locked {
            background: #555;
            border-color: #888;
            cursor: not-allowed;
        }
        .ars-handle.dragging {
            transition: none !important;
            background: #0a70df;
        }
        .ars-rect.dragging {
            transition: none !important;
        }
        .ars-info {
            color: #aaa;
            font-size: 12px;
            font-family: monospace;
            text-align: right;
            width: 100%;
            padding: 0 4px;
            flex-shrink: 0;
            box-sizing: border-box;
        }
    `;
    document.head.appendChild(s);
}

// ============================================================
app.registerExtension({
    name: 'Comfy.AspectRatioSelectorNode',

    async nodeCreated(node) {
        if (node.comfyClass !== 'AspectRatioSelectorNode') return;
        injectStyles();

        const totalMpW = node.widgets.find(w => w.name === 'total_megapixels');
        const minStepW = node.widgets.find(w => w.name === 'min_step');
        const dataW     = node.widgets.find(w => w.name === 'aspect_data_json');
        if (dataW) dataW.hidden = true;

        // ── 状态 ──
        let state = {
            presetIdx: 0,
            dragW: 16,
            dragH: 9,
            rectW: 200,
            rectH: 100,
        };

        try {
            const ex = JSON.parse(dataW?.value || '{}');
            if (ex.w && ex.h) {
                state.dragW = ex.w;
                state.dragH = ex.h;
                if (ex.presetIdx !== undefined) {
                    state.presetIdx = ex.presetIdx;
                } else {
                    for (let i = 0; i < PRESETS.length; i++) {
                        if (PRESETS[i].w === ex.w && PRESETS[i].h === ex.h) {
                            state.presetIdx = i;
                            break;
                        }
                    }
                }
            }
        } catch (_) {}

        const getTotalMp = () => parseFloat(totalMpW?.value) || 100.0;
        const getMinStep = () => parseInt(minStepW?.value) || 8;

        function syncToPython() {
            if (dataW) dataW.value = JSON.stringify({ w: state.dragW, h: state.dragH, presetIdx: state.presetIdx });
        }

        function fitRatioToCanvas(cw, ch) {
            const ratio = state.dragW / state.dragH;
            const maxW = cw - PAD * 2;
            const maxH = ch - PAD * 2;
            if (ratio >= maxW / maxH) {
                state.rectW = maxW;
                state.rectH = maxW / ratio;
            } else {
                state.rectH = maxH;
                state.rectW = maxH * ratio;
            }
            state.rectW = Math.max(state.rectW, MIN);
            state.rectH = Math.max(state.rectH, MIN);
        }

        // ── DOM ──
        const container = document.createElement('div');
        container.className = 'ars-container';

        // 自定义下拉
        const ddWrap = document.createElement('div');
        ddWrap.className = 'ars-dropdown-wrap';
        const ddBtn = document.createElement('div');
        ddBtn.className = 'ars-dropdown-btn';
        ddBtn.textContent = PRESETS[state.presetIdx].label;
        const ddOpts = document.createElement('div');
        ddOpts.className = 'ars-dropdown-options';
        const optionEls = [];
        PRESETS.forEach((p, i) => {
            const opt = document.createElement('div');
            opt.className = 'ars-option' + (i === state.presetIdx ? ' selected' : '');
            opt.textContent = p.label;
            opt.addEventListener('mousedown', (e) => {
                e.preventDefault();
                e.stopPropagation();
                selectPreset(i);
            });
            optionEls.push(opt);
            ddOpts.appendChild(opt);
        });
        ddWrap.appendChild(ddBtn);
        ddWrap.appendChild(ddOpts);

        function openDropdown() {
            ddBtn.classList.add('open');
            ddOpts.classList.add('open');
        }
        function closeDropdown() {
            ddBtn.classList.remove('open');
            ddOpts.classList.remove('open');
        }
        ddBtn.addEventListener('mousedown', (e) => {
            e.preventDefault();
            e.stopPropagation();
            ddOpts.classList.contains('open') ? closeDropdown() : openDropdown();
        });

        const canvas = document.createElement('div');
        canvas.className = 'ars-canvas';
        const rectEl = document.createElement('div');
        const handle = document.createElement('div');
        canvas.appendChild(rectEl);
        canvas.appendChild(handle);

        const info = document.createElement('div');
        info.className = 'ars-info';

        // ── 渲染 ──
        function render() {
            const rw = state.rectW;
            const rh = state.rectH;
            rectEl.style.width  = rw + 'px';
            rectEl.style.height = rh + 'px';
            handle.style.left = (PAD + rw - HANDLE * 0.5) + 'px';
            handle.style.top  = (PAD + rh - HANDLE * 0.5) + 'px';

            if (state.presetIdx === 0) {
                rectEl.className  = dragging ? 'ars-rect free dragging' : 'ars-rect free';
                handle.className  = dragging ? 'ars-handle dragging'   : 'ars-handle';
            } else {
                rectEl.className  = 'ars-rect locked';
                handle.className  = 'ars-handle locked';
            }

            const { w, h } = computeResolution(getTotalMp(), getMinStep(), state.dragW, state.dragH);
            const mp = (w * h / 10000).toFixed(2);
            info.textContent = `${w} × ${h}  (${mp} 万像素)`;
        }

        function updateAll() {
            render();
            syncToPython();
        }

        // ── 预设选择 ──
        function selectPreset(idx) {
            const p = PRESETS[idx];
            state.presetIdx = idx;
            ddBtn.textContent = p.label;
            optionEls.forEach((el, i) => {
                el.className = 'ars-option' + (i === idx ? ' selected' : '');
            });
            closeDropdown();
            if (p.w === null) {
                fitRatioToCanvas(canvas.clientWidth || 260, canvas.clientHeight || 150);
            } else {
                state.dragW = p.w;
                state.dragH = p.h;
                fitRatioToCanvas(canvas.clientWidth || 260, canvas.clientHeight || 150);
            }
            updateAll();
        }

        // ── 初始化 ──
        function initRectSize() {
            fitRatioToCanvas(canvas.clientWidth || 260, canvas.clientHeight || 150);
        }
        initRectSize();

        // ==================== 拖拽 ====================
        // 参考点统一为 handle 的 left/top (即 PAD + rw - HANDLE/2),
        // canvasRect 在 mousedown 固定, 防止 LiteGraph 下每帧 getBoundingClientRect 漂移
        let dragging = false;
        let offX = 0, offY = 0;

        function toCanvas(e) {
            const rect = canvas.getBoundingClientRect();

            const sx = canvas.clientWidth / rect.width;
            const sy = canvas.clientHeight / rect.height;

            return {
                x: (e.clientX - rect.left) * sx,
                y: (e.clientY - rect.top) * sy,
            };
        }

        handle.addEventListener('mousedown', (e) => {
            if (state.presetIdx !== 0) return;
            e.preventDefault();
            e.stopPropagation();

            const p = toCanvas(e);
            // 与 handle.style.left/top 完全一致的参考点
            const handleX = PAD + state.rectW - HANDLE / 2;
            const handleY = PAD + state.rectH - HANDLE / 2;
            offX = p.x - handleX;
            offY = p.y - handleY;

            rectEl.style.transition = 'none';
            handle.style.transition = 'none';
            rectEl.classList.add('dragging');
            handle.classList.add('dragging');
            dragging = true;
        });

        window.addEventListener('mousemove', (e) => {
            if (!dragging) return;
            e.preventDefault();

            const p = toCanvas(e);
            let hx = p.x - offX;
            let hy = p.y - offY;

            const maxHX = canvas.clientWidth  - PAD - HANDLE / 2;
            const maxHY = canvas.clientHeight - PAD - HANDLE / 2;
            const minHX = PAD + MIN - HANDLE / 2;
            const minHY = PAD + MIN - HANDLE / 2;
            hx = Math.max(minHX, Math.min(hx, maxHX));
            hy = Math.max(minHY, Math.min(hy, maxHY));

            const rw = hx - PAD + HANDLE / 2;
            const rh = hy - PAD + HANDLE / 2;

            rectEl.style.width  = rw + 'px';
            rectEl.style.height = rh + 'px';
            handle.style.left = hx + 'px';
            handle.style.top  = hy + 'px';

            state.rectW = rw;
            state.rectH = rh;
            const g = gcd(Math.round(rw), Math.round(rh));
            state.dragW = Math.round(rw) / g;
            state.dragH = Math.round(rh) / g;

            const { w, h } = computeResolution(getTotalMp(), getMinStep(), state.dragW, state.dragH);
            info.textContent = `${w} × ${h}  (${(w * h / 10000).toFixed(2)} 万像素)`;
            syncToPython();
        });

        window.addEventListener('mouseup', () => {
            if (!dragging) return;
            dragging = false;
            rectEl.style.transition = '';
            handle.style.transition = '';
            rectEl.classList.remove('dragging');
            handle.classList.remove('dragging');
            syncToPython();
        });

        // ==================== 全局点击关闭下拉 ====================
        window.addEventListener('mousedown', (e) => {
            if (!ddWrap.contains(e.target)) closeDropdown();
        }, true);

        // ==================== ResizeObserver ====================
        const ro = new ResizeObserver(() => {
            const cw = canvas.clientWidth;
            const ch = canvas.clientHeight;
            if (state.presetIdx !== 0) {
                fitRatioToCanvas(cw, ch);
            }
            if (state.rectW > cw - PAD) state.rectW = Math.max(cw - PAD, MIN);
            if (state.rectH > ch - PAD) state.rectH = Math.max(ch - PAD, MIN);
            updateAll();
        });
        ro.observe(canvas);

        // ==================== widget 变更 ====================
        function onWidgetChanged() { updateAll(); }
        if (totalMpW) totalMpW.callback = onWidgetChanged;
        if (minStepW) minStepW.callback = onWidgetChanged;

        // ==================== 组装 ====================
        container.appendChild(ddWrap);
        container.appendChild(canvas);
        container.appendChild(info);
        node.addDOMWidget('aspect_view', 'HTML', container);

        updateAll();

        // ==================== onConfigure: 刷新页面后从保存的 widget 数据恢复状态 ====================
        node.onConfigure = function () {
            try {
                const ex = JSON.parse(dataW?.value || '{}');
                if (ex.w && ex.h) {
                    state.dragW = ex.w;
                    state.dragH = ex.h;
                    if (ex.presetIdx !== undefined) {
                        state.presetIdx = ex.presetIdx;
                    }
                    fitRatioToCanvas(canvas.clientWidth || 260, canvas.clientHeight || 150);
                    ddBtn.textContent = PRESETS[state.presetIdx].label;
                    optionEls.forEach((el, i) => {
                        el.className = 'ars-option' + (i === state.presetIdx ? ' selected' : '');
                    });
                    updateAll();
                }
            } catch (_) {}
        };
    },
});
