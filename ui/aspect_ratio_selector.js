import { app } from '../../scripts/app.js';

// ============================================================
//  Aspect Ratio Selector — 可视化分辨率选择器
//  增量拖拽无漂移 + 自定义渐入下拉
//  可选输入图：多出「输入图比例」预设，运行后由后端 ui 回填真实比例
// ============================================================

const INPUT_LABEL = '输入图比例';

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
            max-height: 220px;
            overflow-y: auto;
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
        .ars-rect.input-locked {
            background: rgba(150,150,150,0.06);
            border: 1.5px dashed rgba(190,190,190,0.5);
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
        .ars-info.unknown { color: #777; font-style: italic; }
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
            presetIdx: 0,       // 基础预设索引，0 = 自由比例
            useInput: false,    // true = 输入图比例（仅输入图接线时可选）
            inputW: 0,          // 最近一次运行得到的输入图尺寸
            inputH: 0,
            savedPresetIdx: 0,  // 进入输入图比例前的手动选择，断线时恢复
            savedDragW: 16,
            savedDragH: 9,
            dragW: 16,
            dragH: 9,
            rectW: 200,
            rectH: 100,
        };

        const clampPresetIdx = (i) => (
            Number.isInteger(i) && i >= 0 && i < PRESETS.length ? i : 0
        );

        function findPresetIndex(w, h) {
            for (let i = 0; i < PRESETS.length; i++) {
                if (PRESETS[i].w === w && PRESETS[i].h === h) return i;
            }
            return 0;
        }

        // aspect_data_json 只由前端维护：比例 + 选择模式 + 最近一次输入图尺寸
        function loadAspectData(raw) {
            let ex;
            try { ex = JSON.parse(raw || '{}'); } catch (_) { return; }
            if (!ex || typeof ex !== 'object') return;

            const hasRatio = Number.isFinite(ex.w) && Number.isFinite(ex.h) && ex.w > 0 && ex.h > 0;
            state.dragW = hasRatio ? ex.w : 16;
            state.dragH = hasRatio ? ex.h : 9;
            state.presetIdx = Number.isInteger(ex.presetIdx)
                ? clampPresetIdx(ex.presetIdx)
                : findPresetIndex(state.dragW, state.dragH);

            state.useInput = ex.useInput === true;
            state.inputW = Number.isFinite(ex.inputW) && ex.inputW > 0 ? Math.round(ex.inputW) : 0;
            state.inputH = Number.isFinite(ex.inputH) && ex.inputH > 0 ? Math.round(ex.inputH) : 0;

            state.savedPresetIdx = Number.isInteger(ex.savedPresetIdx)
                ? clampPresetIdx(ex.savedPresetIdx)
                : state.presetIdx;
            state.savedDragW = Number.isFinite(ex.savedW) && ex.savedW > 0 ? ex.savedW : state.dragW;
            state.savedDragH = Number.isFinite(ex.savedH) && ex.savedH > 0 ? ex.savedH : state.dragH;
        }
        loadAspectData(dataW?.value);

        const getTotalMp = () => parseFloat(totalMpW?.value) || 100.0;
        const getMinStep = () => parseInt(minStepW?.value) || 8;

        function syncToPython() {
            if (dataW) dataW.value = JSON.stringify({
                w: state.dragW, h: state.dragH,
                presetIdx: state.presetIdx,
                useInput: state.useInput,
                inputW: state.inputW, inputH: state.inputH,
                savedPresetIdx: state.savedPresetIdx,
                savedW: state.savedDragW, savedH: state.savedDragH,
            });
        }

        // ── 输入图连接检测 ──
        // 槽位要现查：nodeCreated 执行时 node.inputs 可能还没填好
        function imageSlotIndex() {
            return node.inputs?.findIndex((i) => i.name === 'image') ?? -1;
        }

        function imageConnected() {
            const slot = imageSlotIndex();
            if (slot < 0) return false;
            return node.isInputConnected
                ? node.isInputConnected(slot)
                : node.inputs[slot].link != null;
        }

        const inputSizeKnown = () => state.inputW > 0 && state.inputH > 0;

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
        const ddOpts = document.createElement('div');
        ddOpts.className = 'ars-dropdown-options';
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

        // ── 下拉选项（接图时在自由比例前插入「输入图比例」）──
        const INPUT_KEY = 'input';
        const presetKey = (i) => `p${i}`;

        let optionEls = [];   // [{ el, key }]

        function visibleOptions() {
            const list = [];
            if (imageConnected()) list.push({ key: INPUT_KEY, label: INPUT_LABEL });
            PRESETS.forEach((p, i) => list.push({ key: presetKey(i), label: p.label }));
            return list;
        }

        function selectionKey() {
            return state.useInput ? INPUT_KEY : presetKey(state.presetIdx);
        }

        function refreshOptionSelection() {
            const key = selectionKey();
            optionEls.forEach(({ el, key: k }) => {
                el.className = 'ars-option' + (k === key ? ' selected' : '');
            });
            ddBtn.textContent = state.useInput ? INPUT_LABEL : PRESETS[state.presetIdx].label;
        }

        function rebuildDropdown() {
            ddOpts.innerHTML = '';
            optionEls = [];
            visibleOptions().forEach((p) => {
                const el = document.createElement('div');
                el.textContent = p.label;
                el.addEventListener('mousedown', (e) => {
                    e.preventDefault();
                    e.stopPropagation();
                    if (p.key === INPUT_KEY) selectInputPreset();
                    else selectPreset(Number(p.key.slice(1)));
                });
                optionEls.push({ el, key: p.key });
                ddOpts.appendChild(el);
            });
            refreshOptionSelection();
        }

        // ── 渲染 ──
        function render() {
            const rw = state.rectW;
            const rh = state.rectH;
            rectEl.style.width  = rw + 'px';
            rectEl.style.height = rh + 'px';
            handle.style.left = (PAD + rw - HANDLE * 0.5) + 'px';
            handle.style.top  = (PAD + rh - HANDLE * 0.5) + 'px';

            if (state.useInput) {
                rectEl.className = 'ars-rect input-locked';
                handle.className = 'ars-handle locked';
            } else if (state.presetIdx === 0) {
                rectEl.className  = dragging ? 'ars-rect free dragging' : 'ars-rect free';
                handle.className  = dragging ? 'ars-handle dragging'   : 'ars-handle';
            } else {
                rectEl.className  = 'ars-rect locked';
                handle.className  = 'ars-handle locked';
            }

            if (state.useInput && !inputSizeKnown()) {
                info.textContent = `${INPUT_LABEL} · 运行后刷新`;
                info.classList.add('unknown');
                return;
            }
            info.classList.remove('unknown');
            const { w, h } = computeResolution(getTotalMp(), getMinStep(), state.dragW, state.dragH);
            const mp = (w * h / 10000).toFixed(2);
            info.textContent = `${w} × ${h}  (${mp} 万像素)` + (state.useInput ? ' · 输入图比例' : '');
        }

        function updateAll() {
            render();
            syncToPython();
        }

        // ── 输入图比例 ──
        function applyInputAspect() {
            if (!inputSizeKnown()) return;
            const g = gcd(state.inputW, state.inputH);
            state.dragW = state.inputW / g;
            state.dragH = state.inputH / g;
            fitRatioToCanvas(canvas.clientWidth || 260, canvas.clientHeight || 150);
        }

        function selectInputPreset() {
            if (!state.useInput) {
                state.savedPresetIdx = state.presetIdx;
                state.savedDragW = state.dragW;
                state.savedDragH = state.dragH;
            }
            state.useInput = true;
            applyInputAspect();
            closeDropdown();
            refreshOptionSelection();
            updateAll();
        }

        function fallbackFromInput() {
            state.useInput = false;
            state.presetIdx = clampPresetIdx(state.savedPresetIdx);
            state.dragW = state.savedDragW > 0 ? state.savedDragW : 16;
            state.dragH = state.savedDragH > 0 ? state.savedDragH : 9;
            fitRatioToCanvas(canvas.clientWidth || 260, canvas.clientHeight || 150);
        }

        // 接线变化 / 工作流载入后调用：选项随连接状态增删，输入图比例在断线时回退
        function refreshInputPresetAvailability() {
            if (state.useInput && !imageConnected()) fallbackFromInput();
            if (state.useInput) applyInputAspect();
            rebuildDropdown();
            updateAll();
        }

        // ── 预设选择 ──
        function selectPreset(idx) {
            const p = PRESETS[idx];
            state.useInput = false;
            state.presetIdx = idx;
            if (p.w === null) {
                fitRatioToCanvas(canvas.clientWidth || 260, canvas.clientHeight || 150);
            } else {
                state.dragW = p.w;
                state.dragH = p.h;
                fitRatioToCanvas(canvas.clientWidth || 260, canvas.clientHeight || 150);
            }
            closeDropdown();
            refreshOptionSelection();
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

        const canvasDraggable = () => !state.useInput && state.presetIdx === 0;

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
            if (!canvasDraggable()) return;
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
            if (state.useInput || state.presetIdx !== 0) {
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

        // ==================== 连接变化：输入图比例选项随连接状态增删 ====================
        // 载入工作流时连线可能晚于 onConfigure 完成，统一延后一帧刷新
        let refreshQueued = false;
        function scheduleRefresh() {
            if (refreshQueued) return;
            refreshQueued = true;
            setTimeout(() => {
                refreshQueued = false;
                try {
                    refreshInputPresetAvailability();
                } catch (e) {
                    console.warn('[AspectRatioSelector] 刷新输入图比例选项失败', e);
                }
            }, 0);
        }

        const prevOnConnectionsChange = node.onConnectionsChange;
        node.onConnectionsChange = function (type, index, connected, link, ioSlot) {
            const r = prevOnConnectionsChange?.apply(this, arguments);
            if (type === 1) scheduleRefresh();   // 1 = INPUT
            return r;
        };

        // ==================== 运行后回填输入图真实尺寸 ====================
        const prevOnExecuted = node.onExecuted;
        node.onExecuted = function (output) {
            prevOnExecuted?.apply(this, arguments);
            try {
                const size = output?.psl_input_size;
                const iw = Array.isArray(size) ? Number(size[0]) : NaN;
                const ih = Array.isArray(size) ? Number(size[1]) : NaN;
                if (Number.isFinite(iw) && iw > 0 && Number.isFinite(ih) && ih > 0) {
                    state.inputW = Math.round(iw);
                    state.inputH = Math.round(ih);
                    if (state.useInput) applyInputAspect();
                    updateAll();
                }
            } catch (e) {
                console.warn('[AspectRatioSelector] 回填输入图尺寸失败', e);
            }
        };

        // ==================== 组装 ====================
        container.appendChild(ddWrap);
        container.appendChild(canvas);
        container.appendChild(info);
        node.addDOMWidget('aspect_view', 'HTML', container);

        refreshInputPresetAvailability();
        scheduleRefresh();

        // ==================== onConfigure: 刷新页面后从保存的 widget 数据恢复状态 ====================
        const prevOnConfigure = node.onConfigure;
        node.onConfigure = function (infoObj) {
            const r = prevOnConfigure?.apply(this, arguments);
            loadAspectData(dataW?.value);
            rebuildDropdown();
            updateAll();
            scheduleRefresh();   // 连线状态可能在此之后才恢复
            return r;
        };
    },
});
