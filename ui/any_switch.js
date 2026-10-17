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
//    4. 条目名：双击行内的「输入 N」原地改名（回车确认 / Esc 取消 / 留空回落
//       默认名），存 labels_json 隐藏 widget。
//  持久化：mode / selection 都是普通 widget（进 prompt + 随工作流保存），
//  labels_json 只随工作流保存（不进 prompt，改名不该让下游缓存失效），
//  onConfigure 恢复。连接数变化实时刷新列表（onConnectionsChange）。
// ============================================================

const GROUP_PREFIX = 'inputs.';
const ORDINAL_RE = /(\d+)$/;
// 双击判定窗口。不用原生 dblclick：行是按选中状态整表重绘的，两次点击之间
// 元素会被换掉，浏览器就不认这是「同一个目标」的连击了（detail 归 1、
// dblclick 不触发）。改为按「同一个序数 + 时间窗」自己判定，重绘不影响。
const DOUBLE_CLICK_MS = 400;
// 面板高度的下限（px）：DOM 元素还没上屏时 offsetHeight 全为 0，不能据此把节点
// 钉成 0 高。120 也是改造前的固定高度，等于「行数少时保持原样」。
const WIDGET_MIN_H = 120;
// 内容高度之外多留的余量（px）：不强求严丝合缝，避免测量误差吃掉最后一行。
const WIDGET_SLACK = 10;

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
            /* 不许随容器收缩：否则容器不够高时行被压到 min-height，offsetHeight
               量出来偏小 → 面板高度算少 → 列表留滚动条、最后一行被挤掉，且
               量到的值会稳定在「压扁」状态上再也纠正不回来。 */
            flex: 0 0 auto;
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
        .asw-row .asw-name {
            flex: 1 1 auto;
            min-width: 0;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }
        .asw-row .asw-name-input {
            flex: 1 1 auto;
            min-width: 0;
            background: #101010;
            color: #fff;
            border: 1px solid #0095ff;
            border-radius: 3px;
            padding: 1px 4px;
            font-size: 12px;
            font-family: inherit;
            line-height: 1.2;
            outline: none;
            user-select: text;
        }
        .asw-row .asw-name-input::placeholder { color: #666; }
        .asw-empty { color: #666; font-size: 11px; padding: 4px 2px; flex: 0 0 auto; }
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

        // 条目名存在 labels_json 隐藏 widget 里。options.serialize=false 只挡住
        // prompt（改名是纯 UI 行为，不该让下游缓存失效），widget 值仍随工作流
        // 文件保存；恢复点在 onConfigure——工作流加载时 widget 值那时才就位。
        // 拿不到它（后端还没重启，object_info 里没这个输入）时不让整个面板陪葬：
        // 面板照常可用，只是不接改名，并在控制台明确报错。
        const labelsW = node.widgets?.find((w) => w.name === 'labels_json');
        if (labelsW) {
            labelsW.hidden = true;
            labelsW.options = labelsW.options || {};
            labelsW.options.serialize = false;
        } else {
            console.error(
                '[AnySwitch] 找不到 labels_json widget：条目名无法持久化。'
                + '请重启 ComfyUI 后端（新增的输入要重新加载）并强制刷新页面（Ctrl+F5）。'
            );
        }

        let labels = {};             // 序数（字符串）-> 自定义条目名
        let editing = null;          // 编辑中的条目 { ordinal, text, input, name }
        let preClickSelection = null;  // 最近一次单击前的选中值（双击改名时撤销首跳）
        let lastClick = { ordinal: null, time: 0 };   // 上一次单击的序数与时刻
        let repaintScheduled = false;                 // 同一帧只补一次重绘
        let fitScheduled = false;                     // 同一帧只补一次量高
        let lastFitH = 0;                             // 上次写入的 min-height，避免白算尺寸

        function readLabels() {
            labels = {};
            if (!labelsW) return;
            const text = typeof labelsW.value === 'string' ? labelsW.value.trim() : '';
            if (!text) return;                       // 旧工作流里这一格可能是空的
            let raw;
            try {
                raw = JSON.parse(text);
            } catch (e) {
                console.warn('[AnySwitch] labels_json 不是合法 JSON，已按无标注处理：', e);
                return;
            }
            if (!raw || typeof raw !== 'object' || Array.isArray(raw)) {
                console.warn('[AnySwitch] labels_json 不是对象，已按无标注处理：', raw);
                return;
            }
            for (const [key, val] of Object.entries(raw)) {
                if (/^\d+$/.test(key) && typeof val === 'string' && val.trim()) {
                    labels[key] = val.trim();
                }
            }
        }

        function writeLabels() {
            if (!labelsW) return;    // 已在 nodeCreated 里报过错，此处不再重复刷屏
            labelsW.value = JSON.stringify(labels);
        }

        function labelOf(ordinal) {
            // 默认名与端口名 0 基对齐：input0/input1 -> 输入 0/输入 1（此前是 +1 偏移，
            // 面板写「输入 1」端口却是 input0，两边对不上）。用户双击改的名字
            // 存 labels_json（键本来就是 0 基序数），不受此改动影响。
            return labels[String(ordinal)] || `输入 ${ordinal}`;
        }

        // 落盘一个条目名；空名 = 删除标注、回落默认名。
        function setLabel(ordinal, text) {
            const key = String(ordinal);
            const name = (text ?? '').trim();
            if (name) labels[key] = name;
            else delete labels[key];
            writeLabels();
        }

        const root = document.createElement('div');
        root.className = 'asw-root';
        // 只钉 min-height，不设 max：节点被手动拉高时多出来的空间留给列表
        // （核心口径是 maxSize = maxHeight ?? Infinity）。真实高度由
        // renderRows → fitNodeHeight 实测内容后写入；这里先给个不裁字的兜底值。
        root.style.setProperty('--comfy-widget-min-height', `${WIDGET_MIN_H}px`);

        // ---- 模式切换 ----
        const modes = document.createElement('div');
        modes.className = 'asw-modes';
        const manualBtn = document.createElement('button');
        manualBtn.type = 'button';
        manualBtn.className = 'asw-mode';
        manualBtn.textContent = '手动选择';
        manualBtn.title = '点选列表里的一路，输出那一路（单选）';
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

        /** 面板内容高度 + 余量（px）。间距/内边距一律从 computedStyle 现读，
         *  以后调 CSS 不会和这里脱节。 */
        function contentHeight() {
            const rootCS = getComputedStyle(root);
            const listCS = getComputedStyle(list);
            let inner = 0;
            for (const el of list.children) inner += el.offsetHeight;
            if (list.children.length > 1) {
                inner += (list.children.length - 1) * (parseFloat(listCS.rowGap) || 0);
            }
            const total = (parseFloat(rootCS.paddingTop) || 0) + (parseFloat(rootCS.paddingBottom) || 0)
                + modes.offsetHeight + (parseFloat(rootCS.rowGap) || 0) + inner;
            // 不追求严丝合缝：多留一点，免得最后一行被那一两像素的测量误差挤掉，
            // 面板也不至于贴得那么局促。多出来的空间由列表吸收（行不改高）。
            return Math.max(WIDGET_MIN_H, Math.round(total) + WIDGET_SLACK);
        }

        /** 把面板高度钉到内容高度，行数变了才让节点重算总尺寸。
         *  与核心 autogrow 增删槽后的做法一致（size = [w, computeSize(w)[1]]）：
         *  行数变化时节点自己长高/收回，不需要手动拉；而内容没变时什么都不做，
         *  手动拉过的尺寸不会被下一次运行/刷新顶回去。 */
        function fitNodeHeight() {
            const h = contentHeight();
            if (h === lastFitH) return;              // 高度没变：不碰节点尺寸
            lastFitH = h;
            root.style.setProperty('--comfy-widget-min-height', `${h}px`);
            if (!node.graph) return;                 // 还没上画布，等下一次重绘
            const w = node.size?.[0];
            if (!Number.isFinite(w) || w <= 0) return;
            try {
                const total = node.computeSize?.([w, 0]);
                const nodeH = Array.isArray(total) ? total[1] : null;
                if (Number.isFinite(nodeH) && nodeH > 0) node.size = [w, nodeH];
                node.graph.setDirtyCanvas(true, true);
            } catch (e) {
                // 自适应失败不影响功能：面板仍有滚动条可以看全列表
                console.warn('[AnySwitch] 自适应高度失败（列表仍可滚动查看）', e);
            }
        }

        /** 下一帧补一次重绘。核心的增删槽是在 requestAnimationFrame 里做的
         *  （autogrowInputDisconnected → replaceNodeInputs）：连接回调这一刻
         *  同步渲染看到的是**裁剪前**的槽位，于是断开线后会残留多余行
         *  （实测「3 个入口 4 行选项」）。本回调比核心那次注册得晚，同一帧内
         *  排在它后面执行，因此渲染到的是裁剪后的真实槽位。 */
        function scheduleRepaint() {
            if (repaintScheduled) return;
            repaintScheduled = true;
            requestAnimationFrame(() => {
                repaintScheduled = false;
                if (!node.graph) return;             // 节点已被删除
                renderRows();
            });
        }

        /** 下一帧只重量一次高度（不重绘行）。给 ResizeObserver 用：在观察器回调里
         *  直接改节点尺寸会撞上「loop limit exceeded」告警，排到下一帧就没有了；
         *  fitNodeHeight 自身按高度去重，因此不会来回触发。 */
        function scheduleFit() {
            if (fitScheduled) return;
            fitScheduled = true;
            requestAnimationFrame(() => {
                fitScheduled = false;
                if (!node.graph) return;
                fitNodeHeight();
            });
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
                editing = null;   // 行都没了，编辑状态一并作废
                fitNodeHeight();
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
                const name = document.createElement('span');
                name.className = 'asw-name';
                name.textContent = labelOf(ordinal);
                name.title = labels[String(ordinal)]
                    ? `${labels[String(ordinal)]}（双击改名）`
                    : '双击改条目名';
                row.append(mark, name);

                if (connected && slot.type) {
                    const type = document.createElement('span');
                    type.className = 'asw-type';
                    type.textContent = slot.type === '*' ? 'Any' : slot.type;
                    row.appendChild(type);
                }

                if (connected && !random) {
                    row.title = '单击输出这一路（再点取消）；双击条目名可改名';
                } else if (random) {
                    row.title = '随机模式：运行后这里会显示实际命中的那一路；双击条目名可改名';
                } else {
                    row.title = '未连接；双击条目名可改名';
                }

                row.addEventListener('click', (e) => {
                    e.preventDefault();
                    e.stopPropagation();
                    if (editing) return;                 // 编辑中不响应行点击
                    const now = Date.now();
                    const isDouble = lastClick.ordinal === ordinal
                        && now - lastClick.time < DOUBLE_CLICK_MS;
                    // 双击判定用掉后清零，三连击不会又触发一次改名
                    lastClick = isDouble ? { ordinal: null, time: 0 } : { ordinal, time: now };

                    if (isDouble) {
                        // 双击 = 改名手势：撤销首跳点击造成的选中切换，否则「已选中
                        // 的那一路」双击后会变成未选中（输出 None），纯属误伤。
                        if (connected && !random && preClickSelection !== null) {
                            selW.value = preClickSelection;
                            renderRows();
                        }
                        preClickSelection = null;
                        if (labelsW) openEditor(ordinal, labels[String(ordinal)] || '');
                        return;
                    }

                    if (!connected || random) return;    // 这两类行只有改名可点
                    preClickSelection = selW.value;
                    // 单选：点已选中的行 = 取消选择（输出 None）
                    selW.value = String(ordinal) === selW.value ? '' : String(ordinal);
                    renderRows();
                });
                list.appendChild(row);
            }
            // 编辑中的条目：重绘（连线变化 / 加载工作流）后把输入框连同已敲的字一起恢复
            if (editing) openEditor(editing.ordinal, editing.text);
            fitNodeHeight();     // 行数定了才算高度（输入框比文字高一点，放在编辑恢复之后）
        }

        // ---- 条目名内联编辑 ----
        /** 在指定条目行上打开输入框；重绘后调用可恢复未提交的输入。 */
        function openEditor(ordinal, text) {
            const row = list.querySelector(`.asw-row[data-ordinal="${ordinal}"]`);
            const name = row?.querySelector('.asw-name');
            if (!row || !name) {      // 该槽已被 autogrow 收回，放弃编辑
                editing = null;
                return;
            }
            const input = document.createElement('input');
            input.type = 'text';
            input.className = 'asw-name-input';
            input.value = text ?? '';
            input.placeholder = `输入 ${ordinal}`;   // 与 labelOf 同源：0 基对齐端口名
            input.spellcheck = false;
            input.title = '回车确认，Esc 取消；留空 = 恢复默认名';
            editing = { ordinal, text: input.value, input, name };
            name.style.display = 'none';
            name.after(input);
            input.focus();
            input.select();

            // 画布快捷键（Delete 删节点、Ctrl+Z 等）与画布手势不许被输入框冒泡上去
            for (const ev of ['keydown', 'keyup', 'keypress', 'input', 'pointerdown',
                              'mousedown', 'mouseup', 'click', 'dblclick', 'wheel']) {
                input.addEventListener(ev, (e) => e.stopPropagation());
            }
            input.addEventListener('input', () => {
                if (editing) editing.text = input.value;
            });
            input.addEventListener('keydown', (e) => {
                if (e.isComposing || e.keyCode === 229) return;   // 输入法组合中，回车是选字
                if (e.key === 'Enter') { e.preventDefault(); finishEdit(true); }
                else if (e.key === 'Escape') { e.preventDefault(); finishEdit(false); }
            });
            input.addEventListener('blur', () => finishEdit(true));
        }

        /** 收尾编辑：commit=false 丢弃改动。提交即写 labels_json（随工作流持久化）。 */
        function finishEdit(commit) {
            if (!editing) return;
            const { ordinal, input, name } = editing;
            editing = null;      // 先清空：input.remove() 触发的 blur 才不会递归进来
            const text = input.value;
            input.remove();
            name.style.display = '';
            if (commit) setLabel(ordinal, text);
            name.textContent = labelOf(ordinal);
            name.title = labels[String(ordinal)]
                ? `${labels[String(ordinal)]}（双击改名）`
                : '双击改条目名';
        }

        // 连接数实时变化 → 刷新列表。核心在 Vue 渲染器下同样触发此回调
        // （连线/断线/删除重排均有调用点）；必须链式调用原函数。
        const prevOnConnectionsChange = node.onConnectionsChange;
        node.onConnectionsChange = function (type, index, connected, link, ioSlot) {
            const r = prevOnConnectionsChange?.apply(this, arguments);
            try {
                const name = ioSlot?.name ?? this.inputs?.[index]?.name ?? '';
                if (type === 1 && name.startsWith(GROUP_PREFIX)) {   // 1 = INPUT
                    renderRows();       // 先给即时反馈（连线那一路立刻出现）
                    scheduleRepaint();  // 再于核心裁剪槽位后补一次（断开线时的残留行靠它清掉）
                }
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

        // getMinHeight 是核心 DOM widget 的布局契约：computeLayoutSize 每次布局都调它
        // （不是缓存值），节点总高按它累加（computeSize 里 t += minHeight），所以列表
        // 长了节点自己长高。不传 getMaxHeight：maxSize 落到 Infinity，手动拉高时多出的
        // 空间归列表，不会被钉死。
        node.addDOMWidget('switch_view', 'HTML', root, {
            serialize: false,
            hideOnZoom: false,
            getMinHeight: contentHeight,
        });

        readLabels();
        syncUI();

        // 元素上屏（或盒子尺寸变化）后再量一次高度：工作流加载时 DOM widget 的挂载
        // 可能晚于 onConfigure，那一刻 offsetHeight 还是 0、只能取兜底高度——列表会不会
        // 溢出成滚动条就靠这一次补量。回调里只排下一帧，不在观察器回调中改尺寸。
        if (typeof ResizeObserver === 'function') {
            const observer = new ResizeObserver(() => scheduleFit());
            observer.observe(root);
            const prevOnRemoved = node.onRemoved;
            node.onRemoved = function () {
                const r = prevOnRemoved?.apply(this, arguments);
                observer.disconnect();       // 节点删除后不再观察，不留悬空回调
                return r;
            };
        }

        // 工作流加载：widget 值此时才恢复，重读条目名并重绘
        const prevConfigure = node.onConfigure;
        node.onConfigure = function (info) {
            const r = prevConfigure?.apply(this, arguments);
            editing = null;      // 换工作流：放弃正在编辑的输入，一律以文件里的为准
            preClickSelection = null;
            readLabels();
            syncUI();
            // 加载时元素可能还没上屏（offsetHeight 为 0，只能取兜底高度），
            // 下一帧再量一次真实内容高度。
            scheduleRepaint();
            return r;
        };
    },
});
