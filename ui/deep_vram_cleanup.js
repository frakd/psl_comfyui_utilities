import { app } from '../../scripts/app.js';

// DeepVRAMCleanup 前端：
//  1. diag 上屏——后端把诊断文本挂进 ui 输出（见 nodes/deep_vram_cleanup.py），
//     这里渲染成节点内的等宽文本区；最小高度一行字，不限最大高度，
//     节点高度自动增高到文本所需（DOM widget 不参与节点自动布局）。
//  2. run_nonce（隐藏 widget）——仅当 clear_source_node_cache 开启时每次
//     排队自增：本节点有副作用，签名不变会被缓存命中直接跳过不执行；
//     开关关闭时返回空串，维持 ComfyUI 正常缓存复用行为。
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
            min-height: 1.45em;   /* 最小一行字 */
            /* 不限最大高度：内容多长撑多长，节点跟着长高 */
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

        // run_nonce：条件自增的防缓存签名（详见文件头注释）
        const nonceW = node.widgets?.find((w) => w.name === 'run_nonce');
        if (nonceW) {
            nonceW.hidden = true;
            nonceW.serializeValue = () => {
                const clearOn = node.widgets?.find(
                    (w) => w.name === 'clear_source_node_cache')?.value;
                if (!clearOn) return '';
                const next = (parseInt(node.properties?.dvc_nonce ?? '0', 10) || 0) + 1;
                node.properties.dvc_nonce = String(next);
                return String(next);
            };
        }

        // diag 更新后把节点增高到文本所需高度。DOM widget 布局在渲染帧之后
        // 才稳定，用 rAF 量一次；last_y 拿不到时按当前节点高度增量补。
        function fitNodeHeight() {
            requestAnimationFrame(() => {
                try {
                    const w = node.widgets?.find((x) => x.name === 'dvc_diag');
                    const h = pre.scrollHeight;
                    if (!h) return;
                    const base = typeof w?.last_y === 'number'
                        ? w.last_y
                        : node.size[1] - (root.offsetHeight || 0);
                    const needed = base + h + 16;
                    if (node.size[1] < needed) {
                        node.size[1] = needed;
                        node.onResize?.();
                        node.setDirtyCanvas?.(true, true);
                    }
                } catch (e) {
                    console.warn('[DeepVRAMCleanup] 节点高度自适应失败', e);
                }
            });
        }

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
    },
});
