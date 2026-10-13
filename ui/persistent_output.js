import { app } from '../../scripts/app.js';

// ============================================================
//  Persistent Output — 跨运行保留输入的桥接/回放节点
//
//  存储在后端内存里（键 = 节点 unique_id），前端只负责两件事：
//    1. 状态行：显示当前保留的图像/音频概要（GET /psl/persist/status）
//    2. 释放按钮：图像 / 音频 / 全部（POST /psl/persist/release）
//  节点从画布删除时自动「全部释放」——删除即清理，不留孤儿内存。
//  页面刷新后后端存储仍在，nodeCreated 时拉一次状态即可如实显示。
// ============================================================

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
            gap: 6px;
            padding: 6px 8px;
            box-sizing: border-box;
            width: 100%;
            height: 100%;
        }
        .pi-status {
            font-family: monospace;
            font-size: 11px;
            color: #9ecbE8;
            background: #191919;
            border: 1px solid #333;
            border-radius: 4px;
            padding: 4px 6px;
            line-height: 1.5;
            white-space: pre-line;
        }
        .pi-row {
            display: flex;
            gap: 6px;
        }
        .pi-btn {
            flex: 1 1 0;
            min-width: 0;
            background: #3d3d3d;
            color: #e5e5e5;
            border: 1px solid #555;
            border-radius: 5px;
            font-size: 12px;
            padding: 4px 4px;
            cursor: pointer;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        .pi-btn:hover { filter: brightness(1.18); }
        /* 全部释放 = 危险操作，红色系与 ImageContainer 的清空按钮一致 */
        .pi-btn.danger { background: rgba(163, 54, 54, 0.85); border-color: #a33; }
        .pi-btn.danger:hover { background: rgba(210, 66, 66, 0.9); }
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

async function releaseAll(nodeId) {
    try {
        await fetch('/psl/persist/release', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ node_id: nodeId, target: 'all' }),
        });
    } catch (e) {
        console.warn('[PersistentOutput] 释放请求失败', e);
    }
}

app.registerExtension({
    name: 'Comfy.PersonalUtilities.PersistentOutput',

    async nodeCreated(node) {
        if (node.comfyClass !== 'PersistentOutput') return;
        injectStyles();

        const root = document.createElement('div');
        root.className = 'pi-root';
        // 状态 + 按钮共两块，钉住高度（不设 max 会被节点拉高撑开，无意义）
        root.style.setProperty('--comfy-widget-min-height', '78px');
        root.style.setProperty('--comfy-widget-max-height', '78px');

        const status = document.createElement('div');
        status.className = 'pi-status';
        status.textContent = '状态查询中…';

        const row = document.createElement('div');
        row.className = 'pi-row';

        function makeBtn(text, cls, target) {
            const b = document.createElement('button');
            b.type = 'button';
            b.className = `pi-btn ${cls || ''}`;
            b.textContent = text;
            b.title = target === 'all'
                ? '释放本节点保留的全部内容（图像 + 音频），内存立即回收'
                : `只释放保留的${text.replace('释放 ', '')}，另一条通道不受影响`;
            b.addEventListener('click', async (e) => {
                e.preventDefault();
                e.stopPropagation();
                b.disabled = true;
                try {
                    await fetch('/psl/persist/release', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ node_id: node.id, target }),
                    });
                } catch (err) {
                    console.warn('[PersistentOutput] 释放请求失败', err);
                } finally {
                    b.disabled = false;
                    refreshStatus();
                }
            });
            return b;
        }

        row.append(
            makeBtn('释放图像', '', 'image'),
            makeBtn('释放音频', '', 'audio'),
            makeBtn('全部释放', 'danger', 'all'),
        );
        root.append(status, row);

        async function refreshStatus() {
            const d = await fetchStatus(node.id);
            if (!d) {
                status.textContent = '状态未知（后端未就绪？）';
                return;
            }
            status.textContent = `图像: ${d.image}\n音频: ${d.audio}`;
        }

        node.addDOMWidget('persist_view', 'HTML', root, { serialize: false, hideOnZoom: false });

        // 每次运行后状态可能变化（新输入替换/回放），同步一次
        const prevOnExecuted = node.onExecuted;
        node.onExecuted = function (output) {
            prevOnExecuted?.apply(this, arguments);
            refreshStatus();
        };

        // 节点删除 = 垃圾清理：删键即释放，不留孤儿内存
        const prevOnRemoved = node.onRemoved;
        node.onRemoved = function () {
            const r = prevOnRemoved?.apply(this, arguments);
            releaseAll(node.id);
            return r;
        };

        refreshStatus();
    },
});
