import { app } from '../../scripts/app.js';

// ============================================================
//  Text Container — 文本容器
//  上游文本按分隔符切成若干条，点选条目占用 6 路输出。
//
//  三种模式（工具条左侧，随工作流持久化、不进 prompt）：
//    multi      多选 —— 默认，什么都不选；按点击先后占用输出槽 1..6，最多 6 条。
//    single     单选 —— 同时只选一条，点另一条整体替换；没有选中时默认第一条。
//    sequential 顺序 —— 单选 + 排队即推进：本次运行用当前条目，随后推进到下一条；
//                连点多次排队依次用第 1、2、3…条，最后一条停住。取消/失败也推进。
//
//  缓存策略（与 Refreshable Text Box 同源）：
//    - ui_state_json（当前列表、模式、行高、来源）的 serializeValue 固定返回空串：
//      完整状态随工作流持久化，prompt 里永远只是一个常量，不改变缓存签名。
//    - prompt 里只有 selection_json / manual_json / upstream_mode 三个派生字面量，
//      外加 ui_state_json 的常量空串。
//    - 上游接入且未被静音/绕过期间 manual_json 恒为空数组、upstream_mode 恒为 linked；
//      运行到此节点刷新列表只改 ui_state_json，prompt 字节不变 => 本节点与下游缓存不失效。
//    - 上游节点被静音/绕过时，排队改用 manual 模式并提交当前列表，运行不会清空列表。
//    - 顺序模式的推进挂在 selection_json.serializeValue 上（每次排队构建 prompt 时
//      调用）：本次仍用推进前的选中集，随后游标推进。这个字面量本来就会变，
//      逐条步进必然重跑，这是顺序模式的内在代价。
//
//  列表模型：只有一份当前列表。
//    - 上游接入时运行会直接把列表覆盖成上游分割结果（预填内容被覆盖）。
//    - 断开上游后，这份列表变成可编辑的手动槽；手动槽填满后自动追加一个空槽。
//    - 上游接入期间列表只读，点条目选中；手动状态下点左侧方框/行空白选中，
//      点文本区正常编辑。
//    - 每行底部可拖动调整高度（双击复位），行高按槽位索引随工作流持久化。
// ============================================================

const SLOT_COUNT = 6;
const DEFAULT_ROW_HEIGHT = 46;
const MIN_ROW_HEIGHT = 30;
const MAX_ROW_HEIGHT = 480;
const ROW_CLS = 'tc-row';

const MODES = [
    { id: 'multi', label: '多选', title: '多选：按点击先后占用输出槽 1..6，最多 6 条（默认）' },
    { id: 'single', label: '单选', title: '单选：同时只选中一条；点另一条自动替换，再点已选中的不变' },
    { id: 'sequential', label: '顺序', title: '顺序单选：本次运行用当前条目，排队后自动推进到下一条；最后一条停住' },
];
const MODE_IDS = MODES.map((m) => m.id);
const MODE_NAMES = Object.fromEntries(MODES.map((m) => [m.id, m.label]));

// 新版前端默认按“位置”恢复 widgets_values，但隐藏 widget 的值在加载时会
// 从位置数组里被过滤掉。给出完整的 fallback 名称列表后，LGraphNode.configure
// 会改用 widgets_values_named 按名字恢复，隐藏的 ui_state_json 也能正确还原。
const WIDGET_NAMES = [
    'delimiter',
    'unselected_output',
    'selection_json',
    'manual_json',
    'upstream_mode',
    'ui_state_json',
    'text_container_view',
];

const CHECK_SVG = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3.5" stroke-linecap="round" stroke-linejoin="round"><path d="M20 6 9 17l-5-5"/></svg>';

let stylesInjected = false;

function injectStyles() {
    if (stylesInjected || document.getElementById('tc-styles')) return;
    stylesInjected = true;
    const s = document.createElement('style');
    s.id = 'tc-styles';
    s.textContent = `
        .tc-root {
            display: flex;
            flex-direction: column;
            width: 100%;
            height: 100%;
            min-height: 0;
            background: #1b1b1b;
            border: 1px solid #3a3a3a;
            border-radius: 8px;
            box-sizing: border-box;
            overflow: hidden;
            color: #ddd;
            font-size: 12px;
        }
        .tc-header {
            display: flex;
            align-items: center;
            gap: 6px;
            row-gap: 4px;
            flex-wrap: wrap;
            padding: 6px 8px;
            background: linear-gradient(180deg, #2b2b2b, #232323);
            border-bottom: 1px solid #3a3a3a;
            flex: 0 0 auto;
        }
        .tc-modes {
            display: inline-flex;
            gap: 2px;
            padding: 2px;
            background: #171717;
            border: 1px solid #3d3d3d;
            border-radius: 6px;
        }
        .tc-mode {
            appearance: none;
            border: 0;
            background: transparent;
            color: #b8b8b8;
            border-radius: 4px;
            padding: 3px 9px;
            font-size: 12px;
            line-height: 18px;
            cursor: pointer;
            white-space: nowrap;
            transition: background 0.12s, color 0.12s;
        }
        .tc-mode:hover { background: #333; color: #eee; }
        .tc-mode.active {
            background: linear-gradient(180deg, #2f87d8, #2270b8);
            color: #fff;
            box-shadow: 0 1px 4px rgba(0, 120, 220, 0.45);
        }
        .tc-clear {
            appearance: none;
            border: 1px solid #4a4a4a;
            background: #333;
            color: #ccc;
            border-radius: 5px;
            padding: 3px 8px;
            font-size: 12px;
            line-height: 18px;
            cursor: pointer;
            white-space: nowrap;
        }
        .tc-clear:hover:not(.disabled) { background: #444; color: #fff; }
        .tc-clear.disabled { opacity: 0.4; cursor: default; }
        .tc-spacer { flex: 1 1 auto; }
        .tc-pill {
            display: inline-flex;
            align-items: center;
            gap: 5px;
            padding: 2px 8px;
            border-radius: 999px;
            font-size: 11px;
            line-height: 16px;
            white-space: nowrap;
            border: 1px solid transparent;
        }
        .tc-pill::before {
            content: "";
            width: 6px;
            height: 6px;
            border-radius: 50%;
            background: currentColor;
        }
        .tc-pill.upstream {
            color: #7fc4ff;
            background: rgba(47, 135, 216, 0.14);
            border-color: rgba(47, 135, 216, 0.35);
        }
        .tc-pill.manual {
            color: #f0b429;
            background: rgba(240, 180, 41, 0.12);
            border-color: rgba(240, 180, 41, 0.3);
        }
        .tc-scroll {
            flex: 1 1 auto;
            min-height: 72px;
            overflow-y: auto;
            overflow-x: hidden;
            padding: 8px;
            box-sizing: border-box;
        }
        .tc-list { display: flex; flex-direction: column; gap: 6px; }
        .tc-row {
            position: relative;
            display: flex;
            align-items: stretch;
            flex: 0 0 auto;
            min-height: ${MIN_ROW_HEIGHT}px;
            background: linear-gradient(180deg, #262626, #222);
            border: 1px solid #3b3b3b;
            border-radius: 6px;
            box-sizing: border-box;
            cursor: pointer;
            overflow: hidden;
            transition: border-color 0.12s, box-shadow 0.12s, background 0.12s;
        }
        .tc-row:hover { border-color: #555; background: linear-gradient(180deg, #2b2b2b, #262626); }
        .tc-row.selected {
            border-color: #0095ff;
            box-shadow: 0 0 0 1px rgba(0, 149, 255, 0.35), 0 0 10px rgba(0, 149, 255, 0.12);
        }
        .tc-row.resizing { border-color: #0095ff; user-select: none; }
        .tc-check {
            flex: 0 0 auto;
            align-self: center;
            width: 17px;
            height: 17px;
            margin: 0 4px 0 8px;
            padding: 0;
            border-radius: 3px;
            border: 1.5px solid #666;
            background: #1a1a1a;
            color: #fff;
            display: flex;
            align-items: center;
            justify-content: center;
            cursor: pointer;
            opacity: 0.75;
            transition: background 0.12s, border-color 0.12s, opacity 0.12s;
        }
        .tc-check svg { width: 10px; height: 10px; opacity: 0; }
        .tc-row:hover .tc-check { opacity: 1; border-color: #888; }
        .tc-row.selected .tc-check { background: #0095ff; border-color: #0095ff; opacity: 1; }
        .tc-row.selected .tc-check svg { opacity: 1; }
        .tc-body {
            flex: 1 1 auto;
            min-width: 0;
            display: flex;
            padding: 5px 32px 10px 6px;
            box-sizing: border-box;
        }
        .tc-text {
            width: 100%;
            height: 100%;
            min-height: 0;
            box-sizing: border-box;
            align-self: stretch;
            overflow-y: auto;
            white-space: pre-wrap;
            word-break: break-word;
            overflow-wrap: anywhere;
            line-height: 1.45;
            color: #e3e3e3;
            user-select: text;
            scrollbar-width: thin;
            border: 1px solid #3f3f3f;
            border-radius: 4px;
            padding: 4px 8px;
        }
        .tc-text .tc-ph { color: #6b6b6b; font-style: italic; }
        .tc-input {
            width: 100%;
            min-width: 0;
            height: 100%;
            min-height: 0;
            box-sizing: border-box;
            margin: 0;
            padding: 4px 8px;
            border: 1px solid #3f3f3f;
            border-radius: 4px;
            outline: 0;
            resize: none;
            background: transparent;
            color: #e8e8e8;
            font-family: inherit;
            font-size: 12px;
            line-height: 1.45;
            overflow-y: auto;
            scrollbar-width: thin;
            transition: border-color 0.12s;
        }
        .tc-input:focus { border-color: #0095ff; }
        .tc-input::placeholder { color: #666; }
        .tc-badge {
            position: absolute;
            top: 5px;
            right: 6px;
            min-width: 18px;
            height: 18px;
            padding: 0 4px;
            border-radius: 9px;
            background: #0095ff;
            color: #fff;
            font-size: 11px;
            font-weight: 700;
            line-height: 18px;
            text-align: center;
            box-shadow: 0 2px 6px rgba(0, 0, 0, 0.35);
            opacity: 0;
            transform: scale(0.8);
            transition: opacity 0.12s, transform 0.12s;
            pointer-events: none;
        }
        .tc-row.selected .tc-badge { opacity: 1; transform: scale(1); }
        .tc-grip {
            position: absolute;
            left: 0;
            right: 0;
            bottom: 0;
            height: 7px;
            display: flex;
            align-items: center;
            justify-content: center;
            cursor: ns-resize;
            touch-action: none;
            opacity: 0;
            transition: opacity 0.12s;
        }
        .tc-grip::before {
            content: "";
            width: 34px;
            height: 2px;
            border-radius: 1px;
            background: #666;
        }
        .tc-row:hover .tc-grip, .tc-row.resizing .tc-grip { opacity: 1; }
        .tc-grip:hover::before, .tc-row.resizing .tc-grip::before { background: #0095ff; }
        .tc-status {
            flex: 0 0 auto;
            padding: 4px 8px;
            background: #202020;
            border-top: 1px solid #383838;
            color: #9a9a9a;
            font-size: 11px;
            font-family: monospace;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        .tc-scroll::-webkit-scrollbar, .tc-text::-webkit-scrollbar, .tc-input::-webkit-scrollbar { width: 8px; }
        .tc-scroll::-webkit-scrollbar-thumb,
        .tc-text::-webkit-scrollbar-thumb,
        .tc-input::-webkit-scrollbar-thumb { background: #444; border-radius: 4px; }
        .tc-scroll::-webkit-scrollbar-thumb:hover,
        .tc-text::-webkit-scrollbar-thumb:hover,
        .tc-input::-webkit-scrollbar-thumb:hover { background: #5a5a5a; }
        .tc-scroll::-webkit-scrollbar-track,
        .tc-text::-webkit-scrollbar-track,
        .tc-input::-webkit-scrollbar-track { background: transparent; }
    `;
    document.head.appendChild(s);
}

function clampHeight(value) {
    return Math.min(MAX_ROW_HEIGHT, Math.max(MIN_ROW_HEIGHT, Math.round(value)));
}

function padSlots(list) {
    const out = list.slice();
    while (out.length < SLOT_COUNT) out.push('');
    return out;
}

function isStringArray(value) {
    return Array.isArray(value) && value.every((item) => typeof item === 'string');
}

function normalizeHeights(raw) {
    const out = {};
    if (!raw || typeof raw !== 'object' || Array.isArray(raw)) return out;
    for (const [key, value] of Object.entries(raw)) {
        const index = Number.parseInt(key, 10);
        const height = Number(value);
        if (!Number.isInteger(index) || index < 0 || !Number.isFinite(height)) continue;
        out[index] = clampHeight(height);
    }
    return out;
}

function normalizeSelection(raw) {
    if (!Array.isArray(raw)) return null;
    const out = [];
    const seen = new Set();
    for (const value of raw) {
        if (!Number.isInteger(value) || value < 0 || seen.has(value)) continue;
        seen.add(value);
        out.push(value);
        if (out.length >= SLOT_COUNT) break;
    }
    return out;
}

app.registerExtension({
    name: 'Comfy.PersonalUtilities.TextContainer',

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== 'TextContainer') return;
        nodeData.fallbackWidgetsValuesNames = WIDGET_NAMES;
    },

    async nodeCreated(node) {
        if (node.comfyClass !== 'TextContainer') return;
        injectStyles();

        const selW = node.widgets?.find((w) => w.name === 'selection_json');
        const manualW = node.widgets?.find((w) => w.name === 'manual_json');
        const uiW = node.widgets?.find((w) => w.name === 'ui_state_json');
        const upstreamW = node.widgets?.find((w) => w.name === 'upstream_mode');
        const inertW = node.widgets?.find((w) => w.name === 'unselected_output');
        if (!selW || !manualW || !uiW || !upstreamW) {
            console.error('[TextContainer] 找不到 selection_json / manual_json / ui_state_json / upstream_mode widget');
            return;
        }
        const loadingGraph = !!app.configuringGraph;
        selW.hidden = true;
        manualW.hidden = true;
        uiW.hidden = true;
        upstreamW.hidden = true;

        // ui_state_json 存完整 UI 状态，但不能原样进 prompt，否则运行刷新、点选、
        // 拖行高都会改变 prompt 字节、让下游缓存白丢。这里不能用
        // options.serialize=false —— 新版前端加载工作流时会把这些 widget 的值
        // 从 widgets_values 里过滤掉，导致状态恢复不了。改为保留正常序列化、
        // 但 serializeValue 始终提交常量空串：工作流照常恢复 widget.value，
        // prompt 里永远只是一个常量，缓存签名保持稳定。
        uiW.serializeValue = () => '';

        // 新建节点默认只有标准 widget 撑出的窄宽度，六行列表会挤成两行工具条。
        // 只给新节点默认尺寸；加载的工作流保留用户自己拖出来的尺寸。
        const freshNode = !loadingGraph;

        // ---------- 状态 ----------
        let mode = 'multi';
        let slots = Array(SLOT_COUNT).fill('');   // 当前列表：手动内容 / 上游分割结果
        let selection = [];                       // 按输出槽顺序排列的槽索引
        let source = 'manual';                    // upstream | manual（最近一次生效的来源）
        let heights = {};                         // { 槽索引: px }
        let lastShowUpstream = null;
        let notice = '';
        let noticeTimer = null;
        let syncScheduled = false;

        // ---------- 工具 ----------
        function textInConnected() {
            // 槽位要现查：nodeCreated 执行时 node.inputs 可能还没填好。
            const slot = node.inputs?.findIndex((input) => input.name === 'text_in') ?? -1;
            if (slot < 0) return false;
            return node.isInputConnected
                ? node.isInputConnected(slot)
                : node.inputs[slot].link != null;
        }

        function showUpstream() {
            // source=upstream 但线被拔掉时先按手动显示，syncSourceView 会正式切回 manual。
            return source === 'upstream' && textInConnected();
        }

        function upstreamOriginNode() {
            const slot = node.inputs?.findIndex((input) => input.name === 'text_in') ?? -1;
            if (slot < 0) return null;
            let link = null;
            if (typeof node.getInputLink === 'function') {
                link = node.getInputLink(slot);
            } else {
                const linkId = node.inputs?.[slot]?.link;
                if (linkId != null) {
                    const links = node.graph?.links;
                    link = typeof links?.get === 'function' ? links.get(linkId) : links?.[linkId];
                }
            }
            if (!link) return null;
            return node.graph?.getNodeById?.(link.origin_id) ?? app.graph?.getNodeById?.(link.origin_id) ?? null;
        }

        function upstreamBlocked() {
            const origin = upstreamOriginNode();
            if (!origin) return false;
            // LiteGraph 的 NEVER=2（静音）、BYPASS=4（绕过）。上游被禁用时
            // ComfyUI 不会给出文本，这里回退到当前列表，避免把已有内容清空。
            const neverMode = typeof LiteGraph !== 'undefined' && Number.isFinite(LiteGraph.NEVER) ? LiteGraph.NEVER : 2;
            const bypassMode = typeof LiteGraph !== 'undefined' && Number.isFinite(LiteGraph.BYPASS) ? LiteGraph.BYPASS : 4;
            return origin.mode === neverMode || origin.mode === bypassMode;
        }

        function useUpstreamSource() {
            return textInConnected() && !upstreamBlocked();
        }

        function clipSelection(list, length, fallbackOnMissing = false) {
            const before = list.length;
            let out = list.filter((index) => index >= 0 && index < length);
            out = Array.from(new Set(out));
            if (out.length > SLOT_COUNT) out = out.slice(0, SLOT_COUNT);
            if (mode !== 'multi') {
                if (out.length > 1) out = out.slice(0, 1);
                if (fallbackOnMissing && before > 0 && out.length === 0 && length > 0) out = [0];
            }
            return out;
        }

        function setNotice(message) {
            notice = message;
            clearTimeout(noticeTimer);
            noticeTimer = setTimeout(() => {
                notice = '';
                updateStatus();
            }, 2200);
            updateStatus();
        }

        // ---------- 持久化 ----------
        function readState() {
            let state = {};
            try {
                state = JSON.parse(uiW.value || '{}');
            } catch (e) {
                console.warn('[TextContainer] ui_state_json 解析失败，回退默认状态', e, uiW.value);
                state = {};
            }
            mode = MODE_IDS.includes(state.mode) ? state.mode : 'multi';
            slots = isStringArray(state.slots) ? padSlots(state.slots) : Array(SLOT_COUNT).fill('');
            source = state.source === 'upstream' ? 'upstream' : 'manual';
            heights = normalizeHeights(state.heights);

            const loadedSelection = normalizeSelection(state.selection);
            selection = loadedSelection ?? [];
            // 没有保存过 selection 字段时才套用模式默认值；显式保存的空数组表示用户清空过。
            if (loadedSelection === null && mode !== 'multi' && slots.length) selection = [0];
            selection = clipSelection(selection, slots.length, true);
        }

        function writeState() {
            uiW.value = JSON.stringify({
                version: 1,
                mode,
                slots,
                selection,
                source,
                heights,
            });
            // 这三个是 prompt 字面量。value 同时用于工作流持久化；真正提交给后端时
            // 走下面的 serializeValue，从闭包现算，避免 PresetText 之类包装器
            // 在回调返回后把 widget.value 恢复成旧快照。
            selW.value = JSON.stringify(selection);
            manualW.value = JSON.stringify(slots);
            upstreamW.value = textInConnected() ? 'linked' : 'manual';
        }

        // 上游接入且可用期间 manual 内容不参与输出：提交常量空数组。
        // 这样从“预填手动内容”切到“上游结果”的那一次不会改变 prompt，
        // 后续每次运行到此节点刷新列表也保持 prompt 字节稳定。
        // 上游节点被静音/绕过时改用 manual，输出当前列表而不是空占位槽。
        manualW.serializeValue = () => (useUpstreamSource() ? '[]' : JSON.stringify(slots));
        upstreamW.serializeValue = () => (useUpstreamSource() ? 'linked' : 'manual');

        // 顺序模式：排队即推进。先从闭包取推进前的选中集作为本次运行的值，
        // 再推进游标供下次排队使用。取消/失败/缓存命中都已推进，与 Image Container 一致。
        selW.serializeValue = () => {
            if (mode !== 'sequential' || !slots.length) return JSON.stringify(selection);
            if (!selection.length) selection = [0];
            selection = clipSelection(selection, slots.length);
            if (!selection.length) selection = [0];
            const value = JSON.stringify(selection);
            const current = selection[0];
            selection = [Math.min(current + 1, slots.length - 1)];
            // PresetText 包装器会在回调返回后再恢复一次 widget.value，
            // 所以推进结果等这一轮宏任务结束再落盘（不影响本次返回值）。
            setTimeout(() => {
                refreshSelectionUI();
                writeState();
                updateStatus();
            }, 0);
            return value;
        };

        // ---------- DOM ----------
        const root = document.createElement('div');
        root.className = 'tc-root';
        // 只给下限、不给上限：节点拉高时列表跟着变高；节点缩小时回落到下限，
        // 列表内部滚动，不会溢出节点。下限只保证头部/状态栏/一行可见，不锁死
        // 6 行——默认新建的 400x620 能完整显示 6 行，用户仍可继续缩小节点。
        root.style.setProperty('--comfy-widget-min-height', '180px');

        const header = document.createElement('div');
        header.className = 'tc-header';

        const modesWrap = document.createElement('div');
        modesWrap.className = 'tc-modes';
        const modeBtns = {};
        for (const item of MODES) {
            const button = document.createElement('button');
            button.type = 'button';
            button.className = 'tc-mode';
            button.textContent = item.label;
            button.title = item.title;
            button.addEventListener('click', (e) => {
                e.preventDefault();
                e.stopPropagation();
                setMode(item.id);
            });
            modeBtns[item.id] = button;
            modesWrap.appendChild(button);
        }

        const clearBtn = document.createElement('button');
        clearBtn.type = 'button';
        clearBtn.className = 'tc-clear';
        clearBtn.textContent = '全不选';
        clearBtn.title = '清空所有选中（顺序模式下次排队会从第一条重新开始）';
        clearBtn.addEventListener('click', (e) => {
            e.preventDefault();
            e.stopPropagation();
            if (!selection.length) return;
            selection = [];
            refreshSelectionUI();
            writeState();
        });

        const spacer = document.createElement('div');
        spacer.className = 'tc-spacer';

        const sourcePill = document.createElement('span');
        sourcePill.className = 'tc-pill manual';
        sourcePill.title = '当前列表来源：上游 = 由上游文本分割得到（只读）；手动 = 可直接编辑';

        header.append(modesWrap, clearBtn, spacer, sourcePill);

        const scroll = document.createElement('div');
        scroll.className = 'tc-scroll';
        const list = document.createElement('div');
        list.className = 'tc-list';
        scroll.appendChild(list);
        scroll.addEventListener('wheel', (e) => e.stopPropagation());

        const status = document.createElement('div');
        status.className = 'tc-status';

        root.append(header, scroll, status);

        // ---------- 渲染 ----------
        function updateStatus() {
            const upstreamView = showUpstream();
            const cap = mode === 'multi' ? SLOT_COUNT : 1;
            const inertId = inertW?.value || 'none';
            const inertLabel = inertId === 'empty' ? '""' : inertId;
            const filled = slots.reduce((count, text) => count + (text.trim() ? 1 : 0), 0);
            const label = upstreamView ? '上游' : '手动';
            let hint = '';
            if (textInConnected()) {
                if (upstreamBlocked()) hint = ' · 上游已禁用，运行将使用当前列表';
                else if (!upstreamView) hint = ' · 上游已接入，运行到此节点刷新';
            }
            const text =
                `${label} ${filled} 条 / ${slots.length} 槽位 · 已选 ${selection.length}/${cap}` +
                ` · ${MODE_NAMES[mode]} · 缺失槽 ${inertLabel}` +
                `${notice ? ` · ${notice}` : ''}${hint}`;
            status.textContent = text;
            status.title = text;
            sourcePill.textContent = label;
            sourcePill.classList.toggle('upstream', upstreamView);
            sourcePill.classList.toggle('manual', !upstreamView);
            clearBtn.disabled = selection.length === 0;
            clearBtn.classList.toggle('disabled', selection.length === 0);
        }

        function syncModeButtons() {
            for (const item of MODES) {
                modeBtns[item.id]?.classList.toggle('active', mode === item.id);
            }
        }

        function refreshSelectionUI() {
            const ranks = new Map(selection.map((index, rank) => [index, rank + 1]));
            for (const row of list.children) {
                const index = Number(row.dataset.index);
                const rank = ranks.get(index);
                row.classList.toggle('selected', rank !== undefined);
                const badge = row.querySelector('.tc-badge');
                if (badge) badge.textContent = rank === undefined ? '' : String(rank);
            }
            updateStatus();
        }

        function buildRow(index) {
            const row = document.createElement('div');
            row.className = ROW_CLS;
            row.dataset.index = String(index);
            row.style.height = `${heights[index] ?? DEFAULT_ROW_HEIGHT}px`;

            const check = document.createElement('button');
            check.type = 'button';
            check.className = 'tc-check';
            check.title = '选中 / 取消选中';
            check.innerHTML = CHECK_SVG;
            check.addEventListener('pointerdown', (e) => e.stopPropagation());
            check.addEventListener('click', (e) => {
                e.preventDefault();
                e.stopPropagation();
                toggleSelect(index);
            });
            row.appendChild(check);

            const body = document.createElement('div');
            body.className = 'tc-body';
            const text = slots[index] ?? '';
            if (showUpstream()) {
                const view = document.createElement('div');
                view.className = 'tc-text';
                if (text) {
                    view.textContent = text;
                } else {
                    const placeholder = document.createElement('span');
                    placeholder.className = 'tc-ph';
                    placeholder.textContent = '（空占位槽）';
                    view.appendChild(placeholder);
                }
                body.appendChild(view);
            } else {
                const input = document.createElement('textarea');
                input.className = 'tc-input';
                input.value = text;
                input.placeholder = '输入内容…';
                input.spellcheck = false;
                input.addEventListener('pointerdown', (e) => e.stopPropagation());
                input.addEventListener('click', (e) => e.stopPropagation());
                input.addEventListener('input', () => {
                    if (source !== 'manual') {
                        // source=upstream 但线刚被拔掉、syncSourceView 还没跑到的极短窗口
                        source = 'manual';
                    }
                    slots[index] = input.value;
                    let appended = false;
                    if (willOverflowTail()) {
                        slots.push('');
                        appended = true;
                    }
                    if (appended) appendTailRow();
                    writeState();
                    updateStatus();
                });
                body.appendChild(input);
            }
            row.appendChild(body);

            const badge = document.createElement('div');
            badge.className = 'tc-badge';
            row.appendChild(badge);

            const grip = document.createElement('div');
            grip.className = 'tc-grip';
            grip.title = '拖动调整该槽高度（双击复位）';
            attachResize(row, grip, index);
            row.appendChild(grip);

            row.addEventListener('pointerdown', (e) => {
                if (e.target.closest('textarea') || e.target.closest('.tc-grip') || e.target.closest('.tc-check')) return;
                // 阻止事件冒泡到画布，避免点选条目时把节点拖走；不 preventDefault，
                // 上游只读文本仍可正常划选复制。
                e.stopPropagation();
            });
            row.addEventListener('click', (e) => {
                if (e.target.closest('.tc-check') || e.target.closest('.tc-grip')) return;
                if (!showUpstream() && e.target.closest('textarea')) return;
                if (showUpstream() && window.getSelection()?.toString()) return;
                e.preventDefault();
                e.stopPropagation();
                toggleSelect(index);
            });

            return row;
        }

        function appendTailRow() {
            const index = slots.length - 1;
            list.appendChild(buildRow(index));
            // 新槽出现在末尾；把列表滚到底，保证刚追加的空槽可见
            scroll.scrollTop = scroll.scrollHeight;
        }

        function render() {
            const top = scroll.scrollTop;
            list.innerHTML = '';
            for (let index = 0; index < slots.length; index += 1) {
                list.appendChild(buildRow(index));
            }
            scroll.scrollTop = Math.min(top, scroll.scrollHeight);
            syncModeButtons();
            refreshSelectionUI();
        }

        // 手动模式：所有槽都有内容时，末尾追加一个空槽；每次填满都追加一个。
        // 上游模式不追加：分割结果 + 占位槽已经是完整列表。
        function willOverflowTail() {
            if (source !== 'manual') return false;
            return slots.every((text) => text.trim() !== '');
        }

        function ensureManualTail() {
            if (source !== 'manual') return false;
            if (!willOverflowTail()) return false;
            slots.push('');
            return true;
        }

        // ---------- 选中 ----------
        function toggleSelect(index) {
            if (mode === 'multi') {
                const at = selection.indexOf(index);
                if (at >= 0) {
                    selection.splice(at, 1);
                } else {
                    if (selection.length >= SLOT_COUNT) {
                        setNotice(`最多只能选 ${SLOT_COUNT} 条`);
                        return;
                    }
                    selection.push(index);
                }
            } else {
                // 单选 / 顺序：radio 语义，点已选中的保持不变，点另一条整体替换。
                if (selection[0] === index) return;
                selection = [index];
            }
            refreshSelectionUI();
            writeState();
        }

        function setMode(next) {
            if (!MODE_IDS.includes(next) || next === mode) return;
            mode = next;
            if (mode !== 'multi') {
                // 多选收拢为第一条被选中的条目；原本没选就默认第一个占位槽。
                if (selection.length > 1) selection = selection.slice(0, 1);
                if (!selection.length && slots.length) selection = [0];
            }
            syncModeButtons();
            refreshSelectionUI();
            writeState();
        }

        // ---------- 行高拖动 ----------
        function attachResize(row, grip, index) {
            grip.addEventListener('pointerdown', (e) => {
                if (e.button !== 0) return;
                e.preventDefault();
                e.stopPropagation();
                const startY = e.clientY;
                const startHeight = row.offsetHeight;
                // DOM widget 会被画布缩放（CSS transform）：offsetHeight 是未缩放的
                // CSS 像素，clientY 是屏幕像素，按实际缩放比换算拖拽距离。
                const rect = row.getBoundingClientRect();
                const visualScale = rect.height > 0 ? rect.height / startHeight : 1;
                let moved = false;
                row.classList.add('resizing');
                try {
                    grip.setPointerCapture?.(e.pointerId);
                } catch (error) {
                    console.warn('[TextContainer] 行高拖动无法捕获指针，改用普通事件', error);
                }

                const onMove = (ev) => {
                    moved = true;
                    const reversed = ev.clientY - startY;
                    const height = clampHeight(startHeight + reversed / (visualScale || 1));
                    row.style.height = `${height}px`;
                    heights[index] = height;
                };
                const onUp = () => {
                    grip.removeEventListener('pointermove', onMove);
                    grip.removeEventListener('pointerup', onUp);
                    grip.removeEventListener('pointercancel', onUp);
                    try {
                        grip.releasePointerCapture?.(e.pointerId);
                    } catch (error) {
                        // 指针捕获本来就没成功时忽略即可
                    }
                    row.classList.remove('resizing');
                    if (moved) writeState();
                };
                grip.addEventListener('pointermove', onMove);
                grip.addEventListener('pointerup', onUp);
                grip.addEventListener('pointercancel', onUp);
            });
            grip.addEventListener('dblclick', (e) => {
                e.preventDefault();
                e.stopPropagation();
                delete heights[index];
                row.style.height = `${DEFAULT_ROW_HEIGHT}px`;
                writeState();
            });
        }

        // ---------- 来源切换 / 上游刷新 ----------
        function syncSourceView({ force = false } = {}) {
            let sourceChanged = false;
            if (source === 'upstream' && !textInConnected()) {
                // 上游被拔掉：保留最后一次列表内容，转为可编辑的手动槽。
                source = 'manual';
                sourceChanged = true;
            }
            const upstreamView = showUpstream();
            if (force || sourceChanged || upstreamView !== lastShowUpstream) {
                lastShowUpstream = upstreamView;
                selection = clipSelection(selection, slots.length, true);
                if (ensureManualTail()) { /* render 会带上新槽 */ }
                render();
                writeState();
                return;
            }
            updateStatus();
        }

        function scheduleSourceSync(force = false) {
            if (syncScheduled) return;
            syncScheduled = true;
            setTimeout(() => {
                syncScheduled = false;
                syncSourceView({ force });
            }, 0);
        }

        function pickUiValue(output, key) {
            const values = output?.[key];
            return Array.isArray(values) && values.length ? values[0] : undefined;
        }

        function handleExecuted(output) {
            const nextSource = pickUiValue(output, 'source');
            const split = pickUiValue(output, 'split');

            if (nextSource === 'upstream') {
                if (!Array.isArray(split)) {
                    console.warn('[TextContainer] 后端没有回传 split 数组，保留当前列表', output);
                    return;
                }
                // 上游结果直接覆盖当前列表（包括用户预填内容）。
                source = 'upstream';
                slots = padSlots(split.map((item) => (typeof item === 'string' ? item : '')));
                selection = clipSelection(selection, slots.length, true);
            } else if (nextSource === 'manual') {
                source = 'manual';
                // 手动列表是用户自己的内容，后端只是回显来源，不改动。
                selection = clipSelection(selection, slots.length);
            } else {
                return;
            }
            syncSourceView({ force: true });
        }

        // ---------- 上游接线状态 ----------
        const prevOnConnectionsChange = node.onConnectionsChange;
        node.onConnectionsChange = function (type, slotIndex, isConnected, link, ioSlot) {
            prevOnConnectionsChange?.apply(this, arguments);
            const slot = ioSlot || node.inputs?.[slotIndex];
            if (!slot || slot.name !== 'text_in' || !(node.inputs || []).includes(slot)) return;
            scheduleSourceSync();
        };

        // 控件变化时同步状态栏（缺槽档位、分隔符提示）
        for (const widget of [inertW]) {
            if (!widget) continue;
            const previous = widget.callback;
            widget.callback = function (value) {
                previous?.apply(this, arguments);
                updateStatus();
            };
        }

        // serialize:false 防进 prompt；hideOnZoom:false 防画布缩小时变占位矩形
        node.addDOMWidget('text_container_view', 'HTML', root, {
            serialize: false,
            hideOnZoom: false,
        });

        readState();
        if (source === 'manual') ensureManualTail();
        render();
        writeState();
        scheduleSourceSync(true);

        if (freshNode) {
            // nodeCreated 时节点还没被布局器量过，等一帧后再给默认尺寸，
            // 否则会被随后的自动尺寸覆盖。
            setTimeout(() => {
                if (!node.graph) return;
                const width = Math.max(node.size[0], 400);
                const height = Math.max(node.size[1], 620);
                if (width === node.size[0] && height === node.size[1]) return;
                node.setSize([width, height]);
                node.graph.setDirtyCanvas(true, true);
            }, 80);
        }

        const prevOnConfigure = node.onConfigure;
        node.onConfigure = function (info) {
            prevOnConfigure?.apply(this, arguments);
            readState();
            if (source === 'manual') ensureManualTail();
            lastShowUpstream = null;
            render();
            writeState();
            // 链接是在 configure 之后陆续恢复的，等这一轮结束再判断来源。
            scheduleSourceSync(true);
        };

        const prevOnExecuted = node.onExecuted;
        node.onExecuted = function (output) {
            prevOnExecuted?.apply(this, arguments);
            handleExecuted(output);
        };
    },
});
