import { app } from '../../scripts/app.js';
import { api } from '../../scripts/api.js';

// ============================================================
//  Persistent Output — 跨运行保留输入的桥接/回放节点（3 组槽位）
//
//  存储在后端内存里（键 = 节点 unique_id），前端负责：
//    1. 每组一行：行首标注文本框（像标题，纯展示，随工作流持久化）
//       + 行尾「释放」按钮（整组释放：图像+音频一起）
//    2. 释放按槽整组控制；节点从画布删除时自动全清，不留孤儿内存
//  标注存 labels_json 隐藏 widget（serialize=false 不进 prompt，随工作流保存）。
//
//  释放按钮三态反馈：空闲（有内容才可点）→ 释放中…（请求期间）→
//  ✓ 已释放（短暂成功反馈后回落）。行左侧色点+文本显示该槽概要，
//  点击有没有生效一眼可见。
// ============================================================

const SLOT_COUNT = 3;
const SUCCESS_FLASH_MS = 1400;

let stylesInjected = false;

function injectStyles() {
    if (stylesInjected || document.getElementById('pi-styles')) return;
    stylesInjected = true;
    const s = document.createElement('style');
    s.id = 'pi-styles';
    s.textContent = `
        .pi-root {
            display: flex;
            flex-direction: column;
            gap: 5px;
            padding: 5px 6px;
            box-sizing: border-box;
            width: 100%;
            height: 100%;
        }
        /* 行高由内容决定（26px），不被容器均分压缩——之前 flex:1 均分 84px 容器，
           每行只剩 ~22px，按钮文字被上下裁掉；钉死行高后字完整、按钮也更好点 */
        .pi-row {
            display: flex;
            align-items: stretch;
            gap: 6px;
            flex: 0 0 26px;
            min-width: 0;
        }
        /* 标注框 = 行首标题：占满剩余宽度，随节点拉宽变长 */
        .pi-label {
            flex: 1 1 auto;
            min-width: 0;
            background: #191919;
            border: 1px solid #3a3a3a;
            border-radius: 4px;
            color: #9ecbe8;
            font-size: 12px;
            font-family: monospace;
            padding: 0 7px;
            box-sizing: border-box;
        }
        .pi-label::placeholder { color: #666; }
        .pi-label:focus { outline: 1px solid #4a7ab0; border-color: #4a7ab0; }
        /* 空槽的标注框压暗：一眼看出这组没内容 */
        .pi-row.empty .pi-label { color: #556; border-color: #2a2a2a; }
        .pi-row.empty .pi-label::placeholder { color: #4a4a4a; }
        /* 释放按钮：固定宽度不收缩，保证每行的按钮列对齐；行高 26px 全高 */
        .pi-release {
            flex: 0 0 auto;
            width: 62px;
            height: 100%;
            background: #3d3d3d;
            color: #e5e5e5;
            border: 1px solid #555;
            border-radius: 4px;
            font-size: 12px;
            padding: 0;
            line-height: 1;
            cursor: pointer;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
            transition: background 0.25s, border-color 0.25s, color 0.25s;
        }
        .pi-release:hover:not(:disabled) { filter: brightness(1.18); }
        .pi-release:disabled { cursor: default; filter: none; }
        .pi-row.empty .pi-release { opacity: 0.35; }
        /* 释放中：按钮变暗蓝，文字换成 释放中… */
        .pi-release.busy {
            background: #2c5d8f;
            border-color: #3f79ad;
            color: #fff;
            opacity: 1;
        }
        /* 成功：绿色一闪 */
        .pi-release.done {
            background: #2c7a4b;
            border-color: #3f9a63;
            color: #fff;
            opacity: 1;
        }
    `;
    document.head.appendChild(s);
}

async function fetchStatus(nodeId) {
    try {
        const r = await fetch(`/psl/persist/status?node_id=${encodeURIComponent(nodeId)}`);
        if (!r.ok) return null;
        return await r.json();
    } catch (e) {
        console.warn('[PersistentOutput] 状态查询失败', e);
        return null;
    }
}

function releaseRequest(nodeId, target) {
    return fetch('/psl/persist/release', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ node_id: nodeId, target }),
    }).catch((e) => {
        console.warn('[PersistentOutput] 释放请求失败', e);
        return null;
    });
}

app.registerExtension({
    name: 'Comfy.PersonalUtilities.PersistentOutput',

    async nodeCreated(node) {
        if (node.comfyClass !== 'PersistentOutput') return;
        injectStyles();

        const labelsW = node.widgets?.find((w) => w.name === 'labels_json');
        if (!labelsW) {
            console.error('[PersistentOutput] 找不到 labels_json widget');
            return;
        }
        labelsW.hidden = true;
        labelsW.options = labelsW.options || {};
        labelsW.options.serialize = false;   // 不进 prompt（缓存不受标注影响），随工作流文件保存

        let labels = ['', '', ''];

        function readLabels() {
            try {
                const arr = JSON.parse(labelsW.value || '[]');
                labels = [0, 1, 2].map((i) => (Array.isArray(arr) && typeof arr[i] === 'string' ? arr[i] : ''));
            } catch (e) {
                labels = ['', '', ''];
            }
        }

        function writeLabels() {
            labelsW.value = JSON.stringify(labels);
        }

        const root = document.createElement('div');
        root.className = 'pi-root';
        // 高度钉死 = 3 行 x 26px + 2 gap x 5px + 上下 padding 10px = 98px。
        // 行高是内容决定不再均分，容器必须给足，否则又会裁字。
        root.style.setProperty('--comfy-widget-min-height', '98px');
        root.style.setProperty('--comfy-widget-max-height', '98px');

        const rows = [];
        for (let i = 0; i < SLOT_COUNT; i++) {
            const row = document.createElement('div');
            row.className = 'pi-row';

            const label = document.createElement('input');
            label.type = 'text';
            label.className = 'pi-label';
            label.placeholder = `槽 ${i + 1}：标注这组存了什么…`;
            label.value = labels[i];
            label.title = '纯展示标注，随工作流保存，不参与运行';
            // 不让画布吃键盘事件
            label.addEventListener('keydown', (e) => e.stopPropagation());
            label.addEventListener('input', () => {
                labels[i] = label.value;
                writeLabels();
            });

            const releaseBtn = document.createElement('button');
            releaseBtn.type = 'button';
            releaseBtn.className = 'pi-release';
            releaseBtn.textContent = '释放';
            releaseBtn.addEventListener('click', async (e) => {
                e.preventDefault();
                e.stopPropagation();
                if (releaseBtn.disabled) return;
                releaseBtn.disabled = true;
                releaseBtn.classList.remove('done');
                releaseBtn.classList.add('busy');
                releaseBtn.textContent = '释放中…';
                try {
                    await releaseRequest(node.id, `slot${i + 1}`);
                    releaseBtn.classList.remove('busy');
                    releaseBtn.classList.add('done');
                    releaseBtn.textContent = '✓ 已释放';
                } finally {
                    refreshStatus();
                    // 成功反馈停留片刻后回落（期间按钮保持禁用，防连点）
                    setTimeout(() => {
                        releaseBtn.classList.remove('done');
                        releaseBtn.disabled = false;
                        refreshStatus();
                    }, SUCCESS_FLASH_MS);
                }
            });

            row.append(label, releaseBtn);
            root.appendChild(row);
            rows.push({ row, label, releaseBtn });
        }

        // DOM 更新抽出来：事件数据与 HTTP 数据走同一条应用路径。
        // busy/done 两态的文案不被覆盖；只有空闲态才同步文案与禁用。
        function applySlots(slots) {
            for (let i = 0; i < SLOT_COUNT; i++) {
                const held = typeof slots?.[i] === 'string' && slots[i] !== '未保留';
                rows[i].row.classList.toggle('empty', !held);
                const summary = held ? slots[i] : '';
                const btn = rows[i].releaseBtn;
                if (!btn.classList.contains('busy') && !btn.classList.contains('done')) {
                    btn.disabled = !held;
                    btn.textContent = '释放';
                    btn.title = held
                        ? `释放槽 ${i + 1} 保留的全部内容（${summary}），内存立即回收`
                        : `槽 ${i + 1} 当前没有保留内容`;
                }
            }
        }

        async function refreshStatus() {
            const d = await fetchStatus(node.id);
            applySlots(d?.slots);
        }

        // 状态刷新加去重窗口：onExecuted 与 status 事件可能几乎同时到达，
        // 连续拉两次浪费且可能与释放请求竞态。
        let refreshTimer = null;
        function scheduleRefresh(delay = 0) {
            clearTimeout(refreshTimer);
            refreshTimer = setTimeout(() => { refreshStatus(); }, delay);
        }

        node.addDOMWidget('persist_view', 'HTML', root, { serialize: false, hideOnZoom: false });

        // 运行后同步状态，三保险：
        // 1) onExecuted：后端现在带 ui 输出（psl_slots），executed 事件必然发出——
        //    直接用事件里的数据更新（零延迟零 HTTP），再 fetch 对账一次
        // 2) status 事件（队列排空）兜底
        // 3) 3 秒轮询兜底：不依赖任何事件，覆盖一切未知时序；后台标签页暂停
        const prevOnExecuted = node.onExecuted;
        node.onExecuted = function (output) {
            prevOnExecuted?.apply(this, arguments);
            const slots = output?.psl_slots;
            if (Array.isArray(slots) && slots.length === SLOT_COUNT) applySlots(slots);
            scheduleRefresh(0);
            scheduleRefresh(600);
        };

        api.addEventListener('status', ({ detail }) => {
            if ((detail?.status?.exec_info?.queue_remaining ?? 1) !== 0) return;
            if (!node.graph) return;
            scheduleRefresh(200);
        });

        setInterval(() => {
            if (!node.graph || document.hidden) return;
            refreshStatus();
        }, 3000);

        // 节点删除 = 垃圾清理：全部释放，不留孤儿内存
        const prevOnRemoved = node.onRemoved;
        node.onRemoved = function () {
            const r = prevOnRemoved?.apply(this, arguments);
            releaseRequest(node.id, 'all');
            return r;
        };

        readLabels();
        refreshStatus();

        const prevConfigure = node.onConfigure;
        node.onConfigure = function (info) {
            const r = prevConfigure?.apply(this, arguments);
            readLabels();          // 工作流加载后恢复标注
            for (let i = 0; i < SLOT_COUNT; i++) rows[i].label.value = labels[i];
            refreshStatus();
            return r;
        };
    },
});
