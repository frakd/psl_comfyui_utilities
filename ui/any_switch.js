import { app } from '../../scripts/app.js';

// ============================================================
//  Any Switch — 可视化多路开关（单选勾选 / 随机模式）
//
//  输入槽由核心 Autogrow 机制自动管理（applyAutogrow / addAutogrowGroup：
//  连满自动加、断开自动收），本前端只负责面板：
//    1. 行列表：每个「已存在的 autogrow 输入槽」一行，显示连接状态，
//       手动模式下点行即选中（单选）；随机模式下行点击禁用。
//    2. 模式切换：手动 ⇄ 随机。
//    3. 随机：在 selection widget 的 serializeValue 里掷骰子并写回——
//       selection 进 prompt，签名每次都变 ⇒ 每次都从本节点重新运行；
//       onExecuted 读后端回传的 psl_pick 把高亮同步到实际命中的那路。
//  持久化：mode / selection 都是普通 widget（进 prompt + 随工作流保存），
//  onConfigure 恢复。连接数变化实时刷新列表（onConnectionsChange）。
// ============================================================

const GROUP_PREFIX = 'inputs.';
const ORDINAL_RE = /(\d+)$/;

let stylesInjected = false;

function injectStyles() {
    if (stylesInjected || document.getElementById('asw-styles')) return;
    stylesInjected = true;
    const s = document.createElement('style');
    s.id = 'asw-styles';
    s.textContent = `
        .asw-root {
            display: flex;
            flex-direction: column;
            gap: 4px;
            padding: 4px 6px;
            box-sizing: border-box;
            width: 100%;
            height: 100%;
        }
        .asw-modes {
            display: flex;
            gap: 4px;
            flex: 0 0 auto;
        }
        .asw-mode {
            flex: 1 1 0;
            background: #3d3d3d;
            color: #bbb;
            border: 1px solid #555;
            border-radius: 4px;
            font-size: 12px;
            padding: 3px 0;
            cursor: pointer;
        }
        .asw-mode:hover { filter: brightness(1.15); }
        .asw-mode.active {
            background: #2c5d8f;
            border-color: #3f79ad;
            color: #fff;
            font-weight: 600;
        }
        .asw-mode.active.random {
            background: #7a5a2c;
            border-color: #ad8a3f;
        }
        .asw-list {
            display: flex;
            flex-direction: column;
            gap: 3px;
            overflow-y: auto;
            flex: 1 1 auto;
            min-height: 0;
        }
        .asw-row {
            display: flex;
            align-items: center;
            gap: 6px;
            background: #191919;
            border: 1px solid #333;
            border-radius: 4px;
            padding: 3px 7px;
            font-size: 12px;
            color: #ccc;
            user-select: none;
            min-height: 22px;
            box-sizing: border-box;
        }
        .asw-row .asw-mark {
            width: 12px;
            height: 12px;
            border: 1px solid #666;
            border-radius: 50%;
            flex: 0 0 auto;
            position: relative;
        }
        .asw-row.selected { border-color: #0095ff; color: #fff; }
        .asw-row.selected .asw-mark { background: #0095ff; border-color: #0095ff; }
        .asw-row.selected.random-pick .asw-mark { background: #d09a2c; border-color: #d09a2c; }
        .asw-row.unconnected { opacity: 0.45; }
        .asw-row.selectable { cursor: pointer; }
        .asw-row.selectable:hover { border-color: #557; }
        .asw-row .asw-type { margin-left: auto; color: #777; font-size: 11px; }
        .asw-empty { color: #666; font-size: 11px; padding: 4px 2px; }
    `;
    document.head.appendChild(s);
}

function slotOrdinal(name) {
    const m = ORDINAL_RE.exec(name);
    return m ? parseInt(m[1], 10) : null;
}

function autogrowSlots(node) {
    return (node.inputs || [])
        .filter((inp) => inp.name && inp.name.startsWith(GROUP_PREFIX) && slotOrdinal(inp.name) !== null)
        .sort((a, b) => slotOrdinal(a.name) - slotOrdinal(b.name));
}

app.registerExtension({
    name: 'Comfy.PersonalUtilities.AnySwitch',

    async nodeCreated(node) {
        if (node.comfyClass !== 'AnySwitch') return;
        injectStyles();

        const modeW = node.widgets?.find((w) => w.name === 'mode');
        const selW = node.widgets?.find((w) => w.name === 'selection');
        if (!modeW || !selW) {
            console.error('[AnySwitch] 找不到 mode / selection widget');
            return;
        }
        // 面板代管两个 widget；hidden 不影响进 prompt 和存工作流
        modeW.hidden = true;
        selW.hidden = true;

        const root = document.createElement('div');
        root.className = 'asw-root';
        root.style.setProperty('--comfy-widget-min-height', '120px');
        root.style.setProperty('--comfy-widget-max-height', '220px');

        // ---- 模式切换 ----
        const modes = document.createElement('div');
        modes.className = 'asw-modes';
        const manualBtn = document.createElement('button');
        manualBtn.type = 'button';
        manualBtn.className = 'asw-mode';
        manualBtn.textContent = '手动选择';
        manualBtn.title = '点选列表里的一路，输出那一 路（单选）';
        const randomBtn = document.createElement('button');
        randomBtn.type = 'button';
        randomBtn.className = 'asw-mode random';
        randomBtn.textContent = '随机输出';
        randomBtn.title = '每次运行从已连接的输入里随机挑一路；每次都会重新运行';
        modes.append(manualBtn, randomBtn);

        const list = document.createElement('div');
        list.className = 'asw-list';
        root.append(modes, list);

        const isRandom = () => modeW.value === 'random';

        function setMode(v) {
            modeW.value = v;
            syncUI();
        }
        manualBtn.addEventListener('click', (e) => { e.stopPropagation(); setMode('manual'); });
        randomBtn.addEventListener('click', (e) => { e.stopPropagation(); setMode('random'); });

        // ---- 随机掷骰子：排队时写入 selection，签名每跑必变 ----
        // 排除上一轮的选择避免连续重复：上一轮结果就是掷骰前的 selW.value
        // （selection 随工作流持久化，刷新页面后由 onConfigure 恢复，跨刷新也有效）。
        // 只约束相邻两次；候选里没有上一轮选择（首轮/上一路刚断开）时不排除，照常随机。
        selW.serializeValue = () => {
            if (isRandom()) {
                const connected = autogrowSlots(node).filter((s) => s.link != null);
                if (connected.length) {
                    let candidates = connected;
                    if (connected.length > 1) {
                        const prev = selW.value;
                        const filtered = connected.filter((s) => String(slotOrdinal(s.name)) !== prev);
                        if (filtered.length) candidates = filtered;
                    }
                    const pick = candidates[Math.floor(Math.random() * candidates.length)];
                    selW.value = String(slotOrdinal(pick.name));
                }
            }
            return selW.value;
        };

        // ---- 面板渲染 ----
        function syncUI() {
            const random = isRandom();
            manualBtn.classList.toggle('active', !random);
            randomBtn.classList.toggle('active', random);
            renderRows();
        }

        function renderRows() {
            const random = isRandom();
            const slots = autogrowSlots(node);
            const selected = selW.value;
            list.innerHTML = '';
            if (!slots.length) {
                const hint = document.createElement('div');
                hint.className = 'asw-empty';
                hint.textContent = '在画布上连接输入后，这里会出现可勾选的列表';
                list.appendChild(hint);
                return;
            }
            for (const slot of slots) {
                const ordinal = slotOrdinal(slot.name);
                const connected = slot.link != null;
                const row = document.createElement('div');
                row.className = 'asw-row';
                row.dataset.ordinal = String(ordinal);
                if (connected && String(ordinal) === selected) {
                    row.classList.add('selected');
                    if (random) row.classList.add('random-pick');
                }
                if (!connected) row.classList.add('unconnected');
                if (connected && !random) row.classList.add('selectable');

                const mark = document.createElement('div');
                mark.className = 'asw-mark';
                const label = document.createElement('span');
                label.textContent = `输入 ${ordinal + 1}`;
                row.append(mark, label);

                if (connected && slot.type) {
                    const type = document.createElement('span');
                    type.className = 'asw-type';
                    type.textContent = slot.type === '*' ? 'Any' : slot.type;
                    row.appendChild(type);
                }

                if (connected && !random) {
                    row.title = '点击输出这一路（再点一次取消选择）';
                    row.addEventListener('click', (e) => {
                        e.preventDefault();
                        e.stopPropagation();
                        // 单选：点已选中的行 = 取消选择（输出 None）
                        selW.value = String(ordinal) === selW.value ? '' : String(ordinal);
                        renderRows();
                    });
                } else if (random) {
                    row.title = '随机模式：运行后这里会显示实际命中的那一路';
                } else {
                    row.title = '未连接';
                }
                list.appendChild(row);
            }
        }

        // 连接数实时变化 → 刷新列表。核心在 Vue 渲染器下同样触发此回调
        // （连线/断线/删除重排均有调用点）；必须链式调用原函数。
        const prevOnConnectionsChange = node.onConnectionsChange;
        node.onConnectionsChange = function (type, index, connected, link, ioSlot) {
            const r = prevOnConnectionsChange?.apply(this, arguments);
            try {
                const name = ioSlot?.name ?? this.inputs?.[index]?.name ?? '';
                if (type === 1 && name.startsWith(GROUP_PREFIX)) renderRows();   // 1 = INPUT
            } catch (e) {
                console.warn('[AnySwitch] 刷新输入列表失败', e);
            }
            return r;
        };

        // 运行后同步随机命中（后端回传实际输出的序数）
        const prevOnExecuted = node.onExecuted;
        node.onExecuted = function (output) {
            prevOnExecuted?.apply(this, arguments);
            const pick = output?.psl_pick;
            const ordinal = Array.isArray(pick) ? pick[0] : pick;
            if (ordinal !== undefined && ordinal !== null) selW.value = String(ordinal);
            renderRows();
        };

        node.addDOMWidget('switch_view', 'HTML', root, { serialize: false, hideOnZoom: false });

        syncUI();

        // 工作流加载：widget 值此时才恢复，重读并重绘
        const prevConfigure = node.onConfigure;
        node.onConfigure = function (info) {
            const r = prevConfigure?.apply(this, arguments);
            syncUI();
            return r;
        };
    },
});
