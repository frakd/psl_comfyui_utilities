import { app } from '../../scripts/app.js';

// DeepVRAMCleanup 前端：
//  1. diag 上屏——后端把诊断文本挂进 ui 输出（见 nodes/deep_vram_cleanup.py）。
//     文本框高度**跟随节点**：占满节点内剩余空间，内容超出在框内滚动；
//     手动缩小节点时框同步缩小，不会溢出节点边界。首次出 diag 时温和
//     增高一次节点（封顶 400px）保证默认可读。
//  2. run_nonce（隐藏 widget）——仅当 clear_source_node_cache 开启时每次
//     排队自增：本节点有副作用，签名不变会被缓存命中直接跳过不执行；
//     开关关闭时返回空串，维持 ComfyUI 正常缓存复用行为。
let stylesInjected = false;

const GROW_CAP = 400;      // 首次 diag 最多把节点撑到比剩余空间高 400px
const BOTTOM_PAD = 10;     // 框底部留给节点边框/圆角的余量

function injectStyles() {
    if (stylesInjected || document.getElementById('dvc-styles')) return;
    stylesInjected = true;
    const s = document.createElement('style');
    s.id = 'dvc-styles';
    s.textContent = `
        .dvc-root { box-sizing: border-box; width: 100%; overflow: hidden; }
        .dvc-diag {
            box-sizing: border-box;
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
            height: 100%;
            overflow-y: auto;     /* 超出框内滚动 */
            overflow-x: hidden;
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

        // 框高 = 节点高度 − 该 widget 顶部位置 − 底部余量。last_y 在布局完成
        // 后才有值，拿不到时按节点高度兜底一个保守值。带缓存避免每帧写 style。
        let lastAvail = -1;
        function syncHeight() {
            const w = node.widgets?.find((x) => x.name === 'dvc_diag');
            const top = typeof w?.last_y === 'number' ? w.last_y : node.size[1] - 60;
            const avail = Math.max(22, node.size[1] - top - BOTTOM_PAD);
            if (avail === lastAvail) return;
            lastAvail = avail;
            root.style.height = avail + 'px';
        }

        // 首次出 diag：温和增高节点（封顶 GROW_CAP），让内容默认可读；
        // 之后用户手动缩放由 syncHeight 跟随，不再反向撑节点。
        function growForContent() {
            requestAnimationFrame(() => {
                const w = node.widgets?.find((x) => x.name === 'dvc_diag');
                const top = typeof w?.last_y === 'number' ? w.last_y : node.size[1] - 60;
                const needed = top + Math.min(pre.scrollHeight + BOTTOM_PAD + 4, GROW_CAP);
                if (node.size[1] < needed) {
                    node.size[1] = needed;
                    node.onResize?.();
                    node.setDirtyCanvas?.(true, true);
                }
                syncHeight();
            });
        }

        const prevOnExecuted = node.onExecuted;
        node.onExecuted = function (output) {
            prevOnExecuted?.apply(this, arguments);
            // 后端 ui 标量必须包 list：output.diag 是 [text]
            const d = Array.isArray(output?.diag) ? output.diag[0] : output?.diag;
            if (typeof d === 'string' && d.trim()) {
                pre.textContent = d;
                growForContent();
            }
        };

        // 节点被手动缩放/布局变化时，框高跟随
        const prevOnResize = node.onResize;
        node.onResize = function (...args) {
            prevOnResize?.apply(this, args);
            syncHeight();
        };
        node.onDrawForeground = ((prev) => function (ctx, ...rest) {
            prev?.apply(this, arguments);
            // 每帧轻量同步一次（仅当高度变化时写 style，浏览器自行去抖重排）
            syncHeight();
        })(node.onDrawForeground);
    },
});
