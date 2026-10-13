import { app } from '../../scripts/app.js';

// ============================================================
//  Persistent Output — 跨运行保留输入的桥接/回放节点（3 组槽位）
//
//  存储在后端内存里（键 = 节点 unique_id），前端负责：
//    1. 每组一行：行首标注文本框（像标题，纯展示，随工作流持久化）
//       + 行尾「释放」按钮（整组释放：图像+音频一起）
//    2. 释放按槽整组控制；节点从画布删除时自动全清，不留孤儿内存
//  标注存 labels_json 隐藏 widget（serialize=false 不进 prompt，随工作流保存）。
// ============================================================

const SLOT_COUNT = 3;

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
            gap: 4px;
            padding: 4px 6px;
            box-sizing: border-box;
            width: 100%;
            height: 100%;
        }
        .pi-row {
            display: flex;
            align-items: stretch;
            gap: 6px;
            flex: 1 1 0;
            min-height: 0;
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
            padding: 3px 7px;
            box-sizing: border-box;
        }
        .pi-label::placeholder { color: #666; }
        .pi-label:focus { outline: 1px solid #4a7ab0; border-color: #4a7ab0; }
        /* 空槽的标注框压暗：一眼看出这组没内容 */
        .pi-row.empty .pi-label { color: #556; border-color: #2a2a2a; }
        .pi-row.empty .pi-label::placeholder { color: #4a4a4a; }
        .pi-release {
            flex: 0 0 auto;
            background: #3d3d3d;
            color: #e5e5e5;
            border: 1px solid #555;
            border-radius: 4px;
            font-size: 12px;
            padding: 3px 10px;
            cursor: pointer;
            white-space: nowrap;
        }
        .pi-release:hover { filter: brightness(1.18); }
        .pi-release:disabled { opacity: 0.35; cursor: default; filter: none; }
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
        // 三行行高固定，整体钉死（不设 max 会被节点拉高撑开，无意义）
        root.style.setProperty('--comfy-widget-min-height', '84px');
        root.style.setProperty('--comfy-widget-max-height', '84px');

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
                releaseBtn.disabled = true;
                await releaseRequest(node.id, `slot${i + 1}`);
                releaseBtn.disabled = false;
                refreshStatus();
            });

            row.append(label, releaseBtn);
            root.appendChild(row);
            rows.push({ row, label, releaseBtn });
        }

        async function refreshStatus() {
            const d = await fetchStatus(node.id);
            const slots = d?.slots;
            for (let i = 0; i < SLOT_COUNT; i++) {
                const held = typeof slots?.[i] === 'string' && slots[i] !== '未保留';
                rows[i].row.classList.toggle('empty', !held);
                const summary = held ? slots[i] : '';
                rows[i].releaseBtn.disabled = !held;
                rows[i].releaseBtn.title = held
                    ? `释放槽 ${i + 1} 保留的全部内容（${summary}），内存立即回收`
                    : `槽 ${i + 1} 当前没有保留内容`;
            }
        }

        node.addDOMWidget('persist_view', 'HTML', root, { serialize: false, hideOnZoom: false });

        // 每次运行后保留内容可能被替换，状态同步一次
        const prevOnExecuted = node.onExecuted;
        node.onExecuted = function (output) {
            prevOnExecuted?.apply(this, arguments);
            refreshStatus();
        };

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
