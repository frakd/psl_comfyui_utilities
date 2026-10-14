import { app } from '../../scripts/app.js';

// DeepVRAMCleanup 诊断上屏：后端把 diag 挂进 ui 输出（见 nodes/deep_vram_cleanup.py），
// 这里在节点上渲染一个等宽文本区，画布上直接看到清理/跳过原因，
// 不用翻命令行日志。多行文本自动换行，超长滚动。
let stylesInjected = false;

function injectStyles() {
    if (stylesInjected || document.getElementById('dvc-styles')) return;
    stylesInjected = true;
    const s = document.createElement('style');
    s.id = 'dvc-styles';
    s.textContent = `
        .dvc-root { box-sizing: border-box; width: 100%; padding: 4px 6px; }
        .dvc-diag {
            margin: 0;
            font: 11px/1.45 monospace;
            white-space: pre-wrap;
            word-break: break-all;
            background: #191919;
            color: #9ecbe8;
            border: 1px solid #3a3a3a;
            border-radius: 4px;
            padding: 6px 8px;
            max-height: 220px;
            overflow: auto;
        }
        .dvc-diag:empty::before { content: '运行后显示诊断…'; color: #556; }
    `;
    document.head.appendChild(s);
}

app.registerExtension({
    name: 'Comfy.PersonalUtilities.DeepVRAMCleanup',

    async nodeCreated(node) {
        if (node.comfyClass !== 'DeepVRAMCleanup') return;
        injectStyles();

        const root = document.createElement('div');
        root.className = 'dvc-root';
        const pre = document.createElement('pre');
        pre.className = 'dvc-diag';
        root.appendChild(pre);
        node.addDOMWidget('dvc_diag', 'DIAG', root, { serialize: false, hideOnZoom: false });

        const prevOnExecuted = node.onExecuted;
        node.onExecuted = function (output) {
            prevOnExecuted?.apply(this, arguments);
            // 后端 ui 标量必须包 list：output.diag 是 [text]
            const d = Array.isArray(output?.diag) ? output.diag[0] : output?.diag;
            if (typeof d === 'string' && d.trim()) {
                pre.textContent = d;
                fitNodeHeight();
            }
        };

        // DOM widget 的高度不参与节点自动布局：diag 更新后把节点增高到
        // 文本实际所需高度，避免节点较小时文本框溢出节点边界。
        function fitNodeHeight() {
            try {
                const w = node.widgets?.find((x) => x.name === 'dvc_diag');
                const needed = (w?.last_y || 0) + root.offsetHeight + 14;
                if (node.size[1] < needed) {
                    node.size[1] = needed;
                    node.onResize?.();
                    node.setDirtyCanvas?.(true, true);
                }
            } catch (e) {
                console.warn('[DeepVRAMCleanup] 节点高度自适应失败', e);
            }
        }
    },
});
