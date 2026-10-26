import { app } from '../../scripts/app.js';
import { api } from '../../scripts/api.js';

// Start/End 都是纯 Any 透传：不靠 execute() 副作用，所以不需要 nonce，
// 也不会改变缓存签名。监控由前端 execution_start/success/error 事件驱动。
// 曲线按各自上限归一化（物理/commit、专用/共享），悬浮显示 GB 与百分比。
const END_CLASS = 'MemoryMonitorEnd';
const RUN_PROP = 'psl_monitor_run';
const endControllers = new Set();
let monitorEventsReady = false;
let monitorActive = false;
let startPromise = null;
let stopPromise = null;

const COLORS = {
    ram: '#58c4dd',
    commit: '#f2b84b',
    dedicated: '#6dd47e',
    shared: '#c792ea',
    grid: 'rgba(255,255,255,0.075)',
    axis: '#7c8892',
    text: '#c8d3dd',
    muted: '#7f8b95',
    capacity: 'rgba(255,255,255,0.18)',
    crosshair: 'rgba(255,255,255,0.38)',
};

let stylesInjected = false;

function injectStyles() {
    if (stylesInjected || document.getElementById('psl-mm-styles')) return;
    stylesInjected = true;
    const style = document.createElement('style');
    style.id = 'psl-mm-styles';
    style.textContent = `
        .psl-mm-root {
            position: relative;
            box-sizing: border-box;
            width: 100%;
            height: 100%;
            min-height: 0;
            padding: 7px 8px 6px;
            display: flex;
            flex-direction: column;
            gap: 5px;
            color: ${COLORS.text};
            background: #121518;
            border: 1px solid #2a3036;
            border-radius: 6px;
            font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif;
            overflow: hidden;
        }
        .psl-mm-head { display: flex; align-items: center; gap: 8px; flex: 0 0 auto; }
        .psl-mm-title { font-size: 12px; font-weight: 600; letter-spacing: .2px; white-space: nowrap; }
        .psl-mm-gpu { color: ${COLORS.muted}; font-size: 10px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; flex: 1 1 auto; }
        .psl-mm-actions { display: flex; align-items: center; gap: 4px; flex: 0 0 auto; }
        .psl-mm-select {
            max-width: 190px;
            height: 21px;
            background: #1b2025;
            color: ${COLORS.text};
            border: 1px solid #343c44;
            border-radius: 4px;
            font-size: 10px;
            padding: 0 3px;
            outline: none;
        }
        .psl-mm-refresh {
            height: 21px;
            padding: 0 6px;
            background: #20262c;
            color: ${COLORS.text};
            border: 1px solid #343c44;
            border-radius: 4px;
            font-size: 10px;
            cursor: pointer;
        }
        .psl-mm-refresh:hover { background: #29323a; border-color: #4a5864; }
        .psl-mm-meta {
            flex: 0 0 auto;
            color: ${COLORS.muted};
            font-size: 10px;
            line-height: 1.4;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        .psl-mm-chart {
            flex: 1 1 0;
            min-height: 48px;
            min-width: 0;
            display: flex;
            flex-direction: column;
            background: #171b1f;
            border: 1px solid #2a3036;
            border-radius: 5px;
            padding: 4px 6px 3px;
            box-sizing: border-box;
            overflow: hidden;
        }
        .psl-mm-chart-head {
            flex: 0 0 auto;
            display: flex;
            align-items: center;
            flex-wrap: wrap;
            gap: 2px 8px;
            min-height: 15px;
            font-size: 10px;
            color: #aab6c0;
        }
        .psl-mm-chart-name { font-weight: 600; color: #d7e1e9; }
        .psl-mm-legend { display: flex; gap: 8px; flex-wrap: wrap; }
        .psl-mm-chip { display: inline-flex; align-items: center; gap: 3px; color: ${COLORS.muted}; }
        .psl-mm-dot { width: 6px; height: 6px; border-radius: 50%; display: inline-block; }
        .psl-mm-axis-note { margin-left: auto; color: #5f6b75; font-size: 9px; }
        .psl-mm-canvas-wrap { position: relative; flex: 1 1 auto; min-height: 0; }
        .psl-mm-canvas { display: block; width: 100%; height: 100%; cursor: crosshair; }
        .psl-mm-empty {
            position: absolute; inset: 0; display: flex; align-items: center; justify-content: center;
            color: #5d6a74; font-size: 11px; pointer-events: none;
        }
        .psl-mm-foot {
            flex: 0 0 auto; min-height: 12px; color: ${COLORS.muted}; font-size: 9px;
            white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
        }
        .psl-mm-foot.error { color: #e08a8a; }
        .psl-mm-tooltip {
            position: fixed;
            z-index: 100000;
            display: none;
            pointer-events: none;
            min-width: 210px;
            max-width: min(380px, calc(100vw - 16px));
            padding: 6px 8px;
            background: rgba(16, 19, 22, 0.96);
            border: 1px solid #3b4650;
            border-radius: 5px;
            box-shadow: 0 6px 18px rgba(0,0,0,.45);
            color: ${COLORS.text};
            font-size: 10px;
            line-height: 1.45;
            backdrop-filter: blur(3px);
            white-space: nowrap;
        }
        .psl-mm-tt-head { color: #e3ebf1; font-weight: 600; margin-bottom: 3px; }
        .psl-mm-tt-row { display: flex; justify-content: space-between; gap: 12px; }
        .psl-mm-tt-label { color: #94a1ac; }
        .psl-mm-tt-val { font-family: ui-monospace, SFMono-Regular, Consolas, monospace; color: #e7eef4; }
        .psl-mm-tt-pct { color: #8fb8cf; font-family: ui-monospace, SFMono-Regular, Consolas, monospace; }
        .psl-mm-start {
            display: flex; flex-direction: column; gap: 4px;
            box-sizing: border-box; width: 100%; height: 100%; min-height: 0;
            padding: 5px 6px;
            background: #121518; border: 1px solid #2a3036; border-radius: 6px;
            color: ${COLORS.text};
            font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif;
            overflow: hidden;
        }
        .psl-mm-start-row { display: flex; align-items: center; gap: 6px; height: 22px; }
        .psl-mm-start-label { flex: 0 0 92px; color: ${COLORS.muted}; font-size: 10px; }
        .psl-mm-start-input {
            flex: 1 1 74px; width: 74px; min-width: 74px; max-width: 220px;
            height: 20px; box-sizing: border-box;
            background: #1b2025; color: ${COLORS.text};
            border: 1px solid #343c44; border-radius: 4px;
            font-size: 10px; padding: 0 4px; outline: none;
        }
        .psl-mm-start-input:focus { border-color: #4a7ab0; }
        .psl-mm-start-note { margin-top: auto; color: #5f6b75; font-size: 9px; line-height: 1; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
    `;
    document.head.appendChild(style);
}

function readStartNumber(node, prop, fallback) {
    const value = Number(node.properties?.[prop]);
    return Number.isFinite(value) ? value : fallback;
}

function setupStartUI(node, { defaultSize = false } = {}) {
    // 纯前端 DOM 面板：值存 node.properties，不进 prompt，因此改采样参数
    // 不会改变缓存签名。node.widgets 还没建好也不影响，因为面板是自己加的。
    if (node.widgets?.some((w) => w.name === 'psl_start_controls')) return;
    injectStyles();
    const root = document.createElement('div');
    root.className = 'psl-mm-start';
    // 面板高度跟随节点；下限只保证两行输入 + 底部说明不被压没。
    root.style.setProperty('--comfy-widget-min-height', '84px');
    const inputs = new Map();

    const addRow = (label, prop, min, max, step, fallback) => {
        const row = document.createElement('div');
        row.className = 'psl-mm-start-row';
        const name = document.createElement('span');
        name.className = 'psl-mm-start-label';
        name.textContent = label;
        const input = document.createElement('input');
        input.type = 'number';
        input.className = 'psl-mm-start-input';
        input.min = String(min);
        input.max = String(max);
        input.step = String(step);
        input.dataset.prop = prop;
        input.value = String(readStartNumber(node, prop, fallback));
        input.addEventListener('keydown', (event) => event.stopPropagation());
        const commit = () => {
            let value = Number(input.value);
            if (!Number.isFinite(value)) value = fallback;
            value = Math.min(max, Math.max(min, value));
            input.value = String(value);
            node.properties = node.properties || {};
            node.properties[prop] = String(value);
        };
        input.addEventListener('input', commit);
        input.addEventListener('change', commit);
        inputs.set(prop, input);
        row.append(name, input);
        root.appendChild(row);
    };

    addRow('采样间隔 (s)', 'psl_interval', 0.2, 60, 0.1, 1.0);
    addRow('最长记录 (min)', 'psl_max_duration_min', 1, 1440, 1, 120);
    const note = document.createElement('div');
    note.className = 'psl-mm-start-note';
    note.textContent = '前端事件驱动 · 面板值不进 prompt';
    root.appendChild(note);

    node.addDOMWidget('psl_start_controls', 'HTML', root, { serialize: false, hideOnZoom: false });
    if (defaultSize) {
        // 节点创建时 DOM 还没被布局器量过，等一帧再给默认尺寸，避免被自动尺寸覆盖。
        setTimeout(() => {
            if (!node.graph) return;
            const width = Math.max(node.size[0], 320);
            const height = Math.max(node.size[1], 170);
            if (width === node.size[0] && height === node.size[1]) return;
            node.setSize([width, height]);
            node.graph.setDirtyCanvas(true, true);
        }, 80);
    }
    node.onResize?.(node.size);
    node.setDirtyCanvas?.(true, true);

    const prevConfigure = node.onConfigure;
    node.onConfigure = function (...args) {
        const result = prevConfigure?.apply(this, args);
        for (const [prop, input] of inputs) {
            const fallback = prop === 'psl_interval' ? 1.0 : 120;
            input.value = String(readStartNumber(node, prop, fallback));
        }
        return result;
    };
}

function findStartNode() {
    return app.graph?._nodes?.find((n) => n.comfyClass === 'MemoryMonitorStart') || null;
}

function refreshEndUIs(runId = '') {
    for (const controller of [...endControllers]) {
        try {
            controller.refresh(runId);
        } catch (error) {
            console.warn('[MemoryMonitor] 刷新结束节点失败', error);
        }
    }
}

async function startMonitor() {
    if (stopPromise) {
        try { await stopPromise; } catch (error) { /* 已有错误已上报 */ }
    }
    const node = findStartNode();
    if (!node || node.mode === 2 || node.mode === 4) return;
    const interval = readStartNumber(node, 'psl_interval', 1.0);
    const maxDuration = readStartNumber(node, 'psl_max_duration_min', 120);
    monitorActive = true;
    startPromise = (async () => {
        try {
            const response = await fetch('/psl/memory_monitor/start', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ interval_s: interval, max_duration_min: maxDuration }),
            });
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            return await response.json();
        } catch (error) {
            monitorActive = false;
            console.warn('[MemoryMonitor] 启动采样失败', error);
            return null;
        } finally {
            startPromise = null;
        }
    })();
}

async function stopMonitor(reason) {
    if (stopPromise) return stopPromise;
    if (!monitorActive && !startPromise) return null;
    monitorActive = false;
    stopPromise = (async () => {
        try {
            if (startPromise) await startPromise;
            const response = await fetch('/psl/memory_monitor/stop', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ reason }),
            });
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            const data = await response.json();
            refreshEndUIs(data?.run_id || '');
            return data;
        } catch (error) {
            console.warn('[MemoryMonitor] 停止采样失败', error);
            refreshEndUIs('');
            return null;
        } finally {
            stopPromise = null;
        }
    })();
    return stopPromise;
}

function setupMonitorEvents() {
    if (monitorEventsReady) return;
    monitorEventsReady = true;
    api.addEventListener('execution_start', () => { startMonitor(); });
    api.addEventListener('execution_success', () => { stopMonitor('end_node'); });
    api.addEventListener('execution_error', () => { stopMonitor('error'); });
    api.addEventListener('execution_interrupted', () => { stopMonitor('interrupted'); });
    api.addEventListener('status', ({ detail }) => {
        if ((detail?.status?.exec_info?.queue_remaining ?? 1) !== 0) return;
        stopMonitor('end_node');
    });
}

function fmtDuration(seconds) {
    if (seconds == null || !Number.isFinite(seconds)) return '—';
    const s = Math.max(0, Number(seconds));
    if (s < 60) return `${s.toFixed(s < 10 ? 1 : 0)}s`;
    const total = Math.round(s);
    const h = Math.floor(total / 3600);
    const m = Math.floor((total % 3600) / 60);
    const sec = total % 60;
    if (h > 0) return `${h}:${String(m).padStart(2, '0')}:${String(sec).padStart(2, '0')}`;
    return `${m}:${String(sec).padStart(2, '0')}`;
}

function fmtBytes(bytes) {
    if (bytes == null || !Number.isFinite(Number(bytes))) return '—';
    const n = Number(bytes);
    if (n >= 1024 ** 3) return `${(n / 1024 ** 3).toFixed(2)} GB`;
    if (n >= 1024 ** 2) return `${(n / 1024 ** 2).toFixed(0)} MB`;
    if (n >= 1024) return `${(n / 1024).toFixed(0)} KB`;
    return `${Math.round(n)} B`;
}

function pct(value, total) {
    if (value == null || total == null || !Number.isFinite(Number(value)) || Number(total) <= 0) return null;
    return (Number(value) / Number(total)) * 100;
}

function fmtPct(value) {
    return value == null ? '—' : `${value.toFixed(1)}%`;
}

function fmtStart(startedAt, epochMs) {
    const date = startedAt ? new Date(startedAt) : (epochMs ? new Date(epochMs) : null);
    if (!date || Number.isNaN(date.getTime())) return '—';
    const p = (n) => String(n).padStart(2, '0');
    return `${p(date.getMonth() + 1)}-${p(date.getDate())} ${p(date.getHours())}:${p(date.getMinutes())}:${p(date.getSeconds())}`;
}

function reasonLabel(reason) {
    switch (reason) {
        case 'end_node': return '正常结束';
        case 'timeout': return '超时收尾';
        case 'superseded': return '被新记录顶替';
        case 'error': return '采样异常';
        case 'interrupted': return '中断收尾';
        case 'incomplete': return '未完整结束';
        default: return reason || '未知';
    }
}

function plotGeometry(width, height) {
    return {
        left: 36,
        right: 8,
        top: 6,
        bottom: 16,
        width,
        height,
    };
}

function niceAxisStep(maxGB) {
    if (!(maxGB > 0) || !Number.isFinite(maxGB)) return 1;
    const rough = maxGB / 4;
    const magnitude = 10 ** Math.floor(Math.log10(rough || 1));
    const normalized = rough / magnitude;
    let step;
    if (normalized <= 1) step = 1;
    else if (normalized <= 2) step = 2;
    else if (normalized <= 5) step = 5;
    else step = 10;
    return step * magnitude;
}

function fmtAxisGB(value) {
    if (value >= 10) return String(Math.round(value));
    if (value >= 1) return value.toFixed(1).replace(/\.0$/, '');
    return value.toFixed(2).replace(/0+$/, '').replace(/\.$/, '');
}

function seriesFor(sample, kind) {
    if (kind === 'ram') {
        return [
            { key: 'ram', label: '物理', color: COLORS.ram, value: sample.ram_used, total: sample.ram_total },
            { key: 'commit', label: 'Commit', color: COLORS.commit, value: sample.commit_used, total: sample.commit_limit },
        ];
    }
    return [
        { key: 'dedicated', label: '专用', color: COLORS.dedicated, value: sample.vram_dedicated_used, total: sample.vram_dedicated_total },
        { key: 'shared', label: '共享', color: COLORS.shared, value: sample.vram_shared_used, total: sample.vram_shared_budget },
    ];
}

function setupEndUI(node, { defaultSize = false } = {}) {
    if (node.widgets?.some((w) => w.name === 'psl_memory_monitor')) return;
    injectStyles();
    const root = document.createElement('div');
    root.className = 'psl-mm-root';
    // 面板跟着节点长/缩；两张图各自 flex，缩小到下限时仍留在面板内。
    root.style.setProperty('--comfy-widget-min-height', '200px');

    const head = document.createElement('div');
    head.className = 'psl-mm-head';
    const title = document.createElement('div');
    title.className = 'psl-mm-title';
    title.textContent = 'Memory Monitor';
    const gpu = document.createElement('div');
    gpu.className = 'psl-mm-gpu';
    gpu.textContent = '等待运行…';
    const actions = document.createElement('div');
    actions.className = 'psl-mm-actions';
    const select = document.createElement('select');
    select.className = 'psl-mm-select';
    select.title = '最近 3 次记录';
    const refresh = document.createElement('button');
    refresh.type = 'button';
    refresh.className = 'psl-mm-refresh';
    refresh.textContent = '刷新';
    refresh.title = '重新读取最近 3 次日志';
    actions.append(select, refresh);
    head.append(title, gpu, actions);

    const meta = document.createElement('div');
    meta.className = 'psl-mm-meta';
    meta.textContent = '尚无记录';

    function makeChart(name, legend) {
        const block = document.createElement('div');
        block.className = 'psl-mm-chart';
        const chartHead = document.createElement('div');
        chartHead.className = 'psl-mm-chart-head';
        const chartName = document.createElement('span');
        chartName.className = 'psl-mm-chart-name';
        chartName.textContent = name;
        const legendEl = document.createElement('span');
        legendEl.className = 'psl-mm-legend';
        for (const item of legend) {
            const chip = document.createElement('span');
            chip.className = 'psl-mm-chip';
            const dot = document.createElement('span');
            dot.className = 'psl-mm-dot';
            dot.style.background = item.color;
            chip.append(dot, document.createTextNode(item.label));
            legendEl.appendChild(chip);
        }
        const note = document.createElement('span');
        note.className = 'psl-mm-axis-note';
        note.textContent = '绝对占用 (GB) · 虚线=上限';
        chartHead.append(chartName, legendEl, note);
        const wrap = document.createElement('div');
        wrap.className = 'psl-mm-canvas-wrap';
        const canvas = document.createElement('canvas');
        canvas.className = 'psl-mm-canvas';
        const empty = document.createElement('div');
        empty.className = 'psl-mm-empty';
        empty.textContent = '运行结束后显示曲线';
        wrap.append(canvas, empty);
        block.append(chartHead, wrap);
        return { block, canvas, empty, legendEl };
    }

    const ramChart = makeChart('系统内存', [
        { label: '物理', color: COLORS.ram },
        { label: 'Commit', color: COLORS.commit },
    ]);
    const vramChart = makeChart('显存', [
        { label: '专用', color: COLORS.dedicated },
        { label: '共享', color: COLORS.shared },
    ]);

    const foot = document.createElement('div');
    foot.className = 'psl-mm-foot';
    foot.textContent = '日志最多保留最近 3 次；悬浮曲线查看 GB / 百分比。';

    const tooltip = document.createElement('div');
    tooltip.className = 'psl-mm-tooltip';

    root.append(head, meta, ramChart.block, vramChart.block, foot);
    node.addDOMWidget('psl_memory_monitor', 'HTML', root, { serialize: false, hideOnZoom: false });
    // Tooltip lives on body so the node's overflow:hidden cannot clip it.
    document.body.appendChild(tooltip);

    if (defaultSize) {
        setTimeout(() => {
            if (!node.graph) return;
            const width = Math.max(node.size[0], 560);
            const height = Math.max(node.size[1], 560);
            if (width === node.size[0] && height === node.size[1]) return;
            node.setSize([width, height]);
            node.graph.setDirtyCanvas(true, true);
        }, 80);
    }
    node.onResize?.(node.size);
    node.setDirtyCanvas?.(true, true);

    let current = null;
    let hoverIndex = null;
    let selectedRunId = node.properties?.[RUN_PROP] || '';

    const charts = [
        { el: ramChart, kind: 'ram' },
        { el: vramChart, kind: 'vram' },
    ];

    function setFoot(text, isError = false) {
        foot.textContent = text;
        foot.classList.toggle('error', isError);
    }

    function renderMeta() {
        if (!current) return;
        const m = current.meta || {};
        const end = current.end || {};
        gpu.textContent = m.gpu_name || 'GPU';
        const parts = [
            `开始 ${fmtStart(m.started_at, m.start_epoch_ms)}`,
            `时长 ${fmtDuration(end.duration_s)}`,
            `间隔 ${Number(m.interval_s || 0).toFixed(1)}s`,
            `${end.samples ?? current.samples?.length ?? 0} 点`,
            reasonLabel(end.reason),
        ];
        meta.textContent = parts.join(' · ');
        const warnings = Array.isArray(end.warnings) ? end.warnings.filter(Boolean) : [];
        if (warnings.length) {
            setFoot(warnings.join('；'), true);
        } else {
            setFoot('日志最多保留最近 3 次；悬浮曲线查看 GB / 百分比。');
        }
    }

    function resizeCanvas(canvas) {
        const dpr = window.devicePixelRatio || 1;
        const rect = canvas.parentElement.getBoundingClientRect();
        const w = Math.max(1, Math.floor(rect.width));
        const h = Math.max(1, Math.floor(rect.height));
        if (canvas.width !== Math.floor(w * dpr) || canvas.height !== Math.floor(h * dpr)) {
            canvas.width = Math.floor(w * dpr);
            canvas.height = Math.floor(h * dpr);
        }
        const ctx = canvas.getContext('2d');
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        return { ctx, w, h };
    }

    function xFor(t, duration, w, g) {
        return g.left + (duration > 0 ? Math.max(0, Math.min(1, t / duration)) : 0) * (w - g.left - g.right);
    }

    function yFor(value, g, h, yMax) {
        const clamped = Math.max(0, Math.min(yMax * 1.25, Number(value) || 0));
        return g.top + (1 - clamped / yMax) * (h - g.top - g.bottom);
    }

    function drawChart(chart, kind) {
        const { canvas, empty } = chart;
        const { ctx, w, h } = resizeCanvas(canvas);
        const g = plotGeometry(w, h);
        ctx.clearRect(0, 0, w, h);
        empty.style.display = current && current.samples?.length ? 'none' : 'flex';
        if (!current || !current.samples?.length) return;

        const samples = current.samples;
        const duration = Math.max(
            Number(current.end?.duration_s) || 0,
            Number(samples[samples.length - 1]?.t) || 0,
            Number(current.meta?.interval_s) || 0,
            1,
        );
        const capacities = {};
        let maxBytes = 0;
        for (const sample of samples) {
            for (const s of seriesFor(sample, kind)) {
                const value = Number(s.value);
                const total = Number(s.total);
                if (Number.isFinite(value) && value > maxBytes) maxBytes = value;
                if (Number.isFinite(total) && total > maxBytes) maxBytes = total;
                if (capacities[s.key] == null && Number.isFinite(total) && total > 0) {
                    capacities[s.key] = total;
                }
            }
        }
        if (!(maxBytes > 0)) maxBytes = 1024 ** 3;
        const maxGB = maxBytes / 1024 ** 3;
        const stepGB = niceAxisStep(maxGB);
        const yMaxGB = Math.max(stepGB, Math.ceil(maxGB / stepGB) * stepGB);
        const yMax = yMaxGB * 1024 ** 3;

        ctx.font = '9px ui-monospace, SFMono-Regular, Consolas, monospace';
        ctx.textBaseline = 'middle';
        ctx.textAlign = 'right';
        ctx.strokeStyle = COLORS.grid;
        ctx.lineWidth = 1;
        for (let valueGB = 0; valueGB <= yMaxGB + 1e-9; valueGB += stepGB) {
            const y = g.top + (1 - valueGB / yMaxGB) * (h - g.top - g.bottom);
            ctx.beginPath();
            ctx.moveTo(g.left, y);
            ctx.lineTo(w - g.right, y);
            ctx.stroke();
            ctx.fillStyle = COLORS.axis;
            ctx.fillText(fmtAxisGB(valueGB), g.left - 5, y);
        }
        ctx.textAlign = 'center';
        ctx.textBaseline = 'top';
        for (let i = 0; i <= 4; i++) {
            const frac = i / 4;
            const x = g.left + frac * (w - g.left - g.right);
            ctx.strokeStyle = i === 0 || i === 4 ? COLORS.grid : 'rgba(255,255,255,0.045)';
            ctx.beginPath();
            ctx.moveTo(x, g.top);
            ctx.lineTo(x, h - g.bottom);
            ctx.stroke();
            ctx.fillStyle = COLORS.axis;
            ctx.fillText(fmtDuration(duration * frac), x, h - g.bottom + 3);
        }
        // Capacity lines: dashed in each series colour, behind the data.
        for (const s of seriesFor(samples[0], kind)) {
            const total = capacities[s.key];
            if (total == null) continue;
            const y = yFor(total, g, h, yMax);
            ctx.save();
            ctx.globalAlpha = 0.3;
            ctx.setLineDash([4, 4]);
            ctx.strokeStyle = s.color;
            ctx.beginPath();
            ctx.moveTo(g.left, y);
            ctx.lineTo(w - g.right, y);
            ctx.stroke();
            ctx.restore();
        }

        const stride = Math.max(1, Math.ceil(samples.length / Math.max(120, (w - g.left - g.right) * 2)));
        for (const def of seriesFor(samples[0], kind)) {
            ctx.strokeStyle = def.color;
            ctx.lineWidth = 1.7;
            ctx.lineJoin = 'round';
            ctx.lineCap = 'round';
            let started = false;
            for (let i = 0; i < samples.length; i += stride) {
                const sample = samples[i];
                const series = seriesFor(sample, kind).find((x) => x.key === def.key);
                const value = series ? Number(series.value) : NaN;
                if (!Number.isFinite(value)) {
                    started = false;
                    continue;
                }
                const x = xFor(sample.t, duration, w, g);
                const y = yFor(value, g, h, yMax);
                if (!started) {
                    ctx.beginPath();
                    ctx.moveTo(x, y);
                    started = true;
                } else {
                    ctx.lineTo(x, y);
                }
            }
            if (started) {
                ctx.stroke();
                const last = samples[samples.length - 1];
                const lastSeries = seriesFor(last, kind).find((x) => x.key === def.key);
                const value = lastSeries ? Number(lastSeries.value) : NaN;
                if (Number.isFinite(value)) {
                    const lx = xFor(last.t, duration, w, g);
                    const ly = yFor(value, g, h, yMax);
                    ctx.fillStyle = def.color;
                    ctx.beginPath();
                    ctx.arc(lx, ly, 2, 0, Math.PI * 2);
                    ctx.fill();
                }
            }
        }

        if (hoverIndex != null && samples[hoverIndex]) {
            const sample = samples[hoverIndex];
            const x = xFor(sample.t, duration, w, g);
            ctx.strokeStyle = COLORS.crosshair;
            ctx.lineWidth = 1;
            ctx.setLineDash([3, 3]);
            ctx.beginPath();
            ctx.moveTo(x, g.top);
            ctx.lineTo(x, h - g.bottom);
            ctx.stroke();
            ctx.setLineDash([]);
            for (const s of seriesFor(sample, kind)) {
                const value = Number(s.value);
                if (!Number.isFinite(value)) continue;
                const y = yFor(value, g, h, yMax);
                ctx.fillStyle = s.color;
                ctx.strokeStyle = '#11161a';
                ctx.lineWidth = 1.5;
                ctx.beginPath();
                ctx.arc(x, y, 3, 0, Math.PI * 2);
                ctx.fill();
                ctx.stroke();
            }
        }
    }

    function drawAll() {
        for (const chart of charts) drawChart(chart.el, chart.kind);
    }

    function nearestIndex(t) {
        const samples = current?.samples || [];
        if (!samples.length) return null;
        let lo = 0;
        let hi = samples.length - 1;
        while (lo < hi) {
            const mid = (lo + hi) >> 1;
            if ((samples[mid].t ?? 0) < t) lo = mid + 1;
            else hi = mid;
        }
        if (lo > 0) {
            const prev = lo - 1;
            if (Math.abs((samples[prev].t ?? 0) - t) <= Math.abs((samples[lo].t ?? 0) - t)) return prev;
        }
        return lo;
    }

    function tooltipRows(sample) {
        const rows = [
            ['物理内存', sample.ram_used, sample.ram_total],
            ['Commit', sample.commit_used, sample.commit_limit],
            ['专用显存', sample.vram_dedicated_used, sample.vram_dedicated_total],
            ['共享显存', sample.vram_shared_used, sample.vram_shared_budget],
        ];
        return rows.map(([label, value, total]) => {
            const p = pct(value, total);
            return { label, value, total, pct: p };
        });
    }

    function renderTooltip(sample) {
        while (tooltip.firstChild) tooltip.removeChild(tooltip.firstChild);
        const head = document.createElement('div');
        head.className = 'psl-mm-tt-head';
        head.textContent = `${fmtStart(current.meta?.started_at, current.meta?.start_epoch_ms)} · +${fmtDuration(sample.t)}`;
        tooltip.appendChild(head);
        for (const row of tooltipRows(sample)) {
            const line = document.createElement('div');
            line.className = 'psl-mm-tt-row';
            const label = document.createElement('span');
            label.className = 'psl-mm-tt-label';
            label.textContent = row.label;
            const val = document.createElement('span');
            val.className = 'psl-mm-tt-val';
            const pctText = fmtPct(row.pct);
            val.textContent = `${fmtBytes(row.value)} / ${fmtBytes(row.total)}  ${pctText}`;
            line.append(label, val);
            tooltip.appendChild(line);
        }
        tooltip.style.display = 'block';
    }

    function onMouseMove(event, canvas) {
        if (!current?.samples?.length) return;
        const rect = canvas.getBoundingClientRect();
        const w = rect.width;
        const h = rect.height;
        const g = plotGeometry(w, h);
        const x = Math.max(g.left, Math.min(rect.width - g.right, event.clientX - rect.left));
        const duration = Math.max(
            Number(current.end?.duration_s) || 0,
            Number(current.samples[current.samples.length - 1]?.t) || 0,
            Number(current.meta?.interval_s) || 0,
            1,
        );
        const t = duration * ((x - g.left) / Math.max(1, w - g.left - g.right));
        const idx = nearestIndex(t);
        if (idx == null) return;
        hoverIndex = idx;
        renderTooltip(current.samples[idx]);
        drawAll();
        const tw = tooltip.offsetWidth || 230;
        const th = tooltip.offsetHeight || 90;
        let left = event.clientX + 14;
        let top = event.clientY + 12;
        if (left + tw > window.innerWidth - 8) left = event.clientX - tw - 14;
        if (top + th > window.innerHeight - 8) top = event.clientY - th - 12;
        left = Math.max(8, Math.min(left, window.innerWidth - tw - 8));
        top = Math.max(8, Math.min(top, window.innerHeight - th - 8));
        tooltip.style.left = `${left}px`;
        tooltip.style.top = `${top}px`;
    }

    function clearHover() {
        if (hoverIndex == null) return;
        hoverIndex = null;
        for (const chart of charts) chart.el.empty.style.display = 'none';
        tooltip.style.display = 'none';
        drawAll();
    }

    for (const chart of charts) {
        chart.el.canvas.addEventListener('mousemove', (event) => onMouseMove(event, chart.el.canvas));
        chart.el.canvas.addEventListener('mouseleave', clearHover);
    }

    async function loadRun(runId) {
        if (!runId) return;
        try {
            const response = await fetch(`/psl/memory_monitor/run?run_id=${encodeURIComponent(runId)}`, { cache: 'no-store' });
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            current = await response.json();
            selectedRunId = runId;
            node.properties = node.properties || {};
            node.properties[RUN_PROP] = runId;
            select.value = runId;
            hoverIndex = null;
            tooltip.style.display = 'none';
            renderMeta();
            drawAll();
            node.setDirtyCanvas?.(true, true);
        } catch (error) {
            console.warn('[MemoryMonitor] 读取记录失败', error);
            setFoot(`读取记录失败：${error?.message || error}`, true);
        }
    }

    async function refreshRuns(autoLoad, preferredRunId = '') {
        try {
            const response = await fetch('/psl/memory_monitor/runs', { cache: 'no-store' });
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            const data = await response.json();
            const runs = Array.isArray(data.runs) ? data.runs : [];
            select.innerHTML = '';
            if (!runs.length) {
                const option = document.createElement('option');
                option.value = '';
                option.textContent = '暂无记录';
                select.appendChild(option);
                select.disabled = true;
                current = null;
                hoverIndex = null;
                gpu.textContent = '等待运行…';
                meta.textContent = '尚无记录';
                setFoot('日志最多保留最近 3 次；悬浮曲线查看 GB / 百分比。');
                drawAll();
                return;
            }
            select.disabled = false;
            for (const run of runs) {
                const option = document.createElement('option');
                option.value = run.run_id;
                option.textContent = `${fmtStart(run.started_at, run.start_epoch_ms)} · ${fmtDuration(run.duration_s)} · ${reasonLabel(run.reason)}`;
                select.appendChild(option);
            }
            const wanted = preferredRunId || selectedRunId;
            const pick = runs.some((r) => r.run_id === wanted) ? wanted : runs[0].run_id;
            select.value = pick;
            if (autoLoad || !current || current.run_id !== pick) {
                await loadRun(pick);
            } else {
                selectedRunId = pick;
            }
        } catch (error) {
            console.warn('[MemoryMonitor] 读取历史失败', error);
            setFoot(`读取历史失败：${error?.message || error}`, true);
        }
    }

    select.addEventListener('change', () => {
        const runId = select.value;
        if (runId) loadRun(runId);
    });
    refresh.addEventListener('click', (event) => {
        event.preventDefault();
        event.stopPropagation();
        refreshRuns(true, selectedRunId);
    });

    const observer = new ResizeObserver(() => {
        if (!node.graph) return;
        drawAll();
    });
    observer.observe(root);

    const controller = {
        refresh: (runId = '') => refreshRuns(true, runId || selectedRunId),
    };
    endControllers.add(controller);
    const prevOnRemoved = node.onRemoved;
    node.onRemoved = function () {
        endControllers.delete(controller);
        tooltip.remove();
        return prevOnRemoved?.apply(this, arguments);
    };

    const prevOnConfigure = node.onConfigure;
    node.onConfigure = function (info) {
        const result = prevOnConfigure?.apply(this, arguments);
        selectedRunId = node.properties?.[RUN_PROP] || selectedRunId;
        refreshRuns(false, selectedRunId);
        return result;
    };

    refreshRuns(false, selectedRunId);
}

app.registerExtension({
    name: 'Comfy.PersonalUtilities.MemoryMonitor',
    async setup() {
        setupMonitorEvents();
    },
    async nodeCreated(node) {
        const defaultSize = !app.configuringGraph;
        if (node.comfyClass === 'MemoryMonitorStart') {
            setupStartUI(node, { defaultSize });
        } else if (node.comfyClass === END_CLASS) {
            setupEndUI(node, { defaultSize });
        }
    },
    async loadedGraphNode(node) {
        // 加载工作流时保留节点自己保存的尺寸，不套默认尺寸。
        if (node.comfyClass === 'MemoryMonitorStart') {
            setupStartUI(node);
        } else if (node.comfyClass === END_CLASS) {
            setupEndUI(node);
        }
    },
});
