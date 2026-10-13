import { app } from '../../scripts/app.js';
import { api } from '../../scripts/api.js';

// ============================================================
//  Mask Paint Output — 底图上传（粘贴 / 拖放）+ 取图模式接管上游
//
//  粘贴有两条入口：官方菜单项「粘贴图像」的判定只有
//  typeof node.pasteFiles === 'function'（useImageMenuOptions.ts:18-20），菜单项本身
//  由核心提供、读 navigator.clipboard 后调用 node.pasteFiles([file])；常驻按钮
//  「📋 粘贴底图」则自己读剪贴板（右键三层菜单太慢）。两者都只走 applyFile。
//
//  image widget 保持 STRING 而非 COMBO：遮罩编辑器保存时写入的值可能带
//  " [input]" 后缀，COMBO 会按成员校验把它拒掉（execution.py:1059-1068 没有
//  针对 image_upload 的豁免），而 STRING 不做成员校验，两种写法都收。
// ============================================================

const BTN = { serialize: false };

async function uploadBlob(blob, filename) {
    const fd = new FormData();
    fd.append('image', blob, filename);
    fd.append('overwrite', 'false');
    // 不传 subfolder：上传到 input 根目录，widget 值与遮罩编辑器自己保存时的
    // 格式一致，编辑器一定能加载它。
    fd.append('type', 'input');
    const r = await fetch('/upload/image', { method: 'POST', body: fd });
    if (!r.ok) throw new Error(`上传失败 HTTP ${r.status}: ${await r.text()}`);
    return r.json();   // { name, subfolder, type }
}

app.registerExtension({
    name: 'Comfy.PersonalUtilities.MaskPaintOutput',

    async nodeCreated(node) {
        if (node.comfyClass !== 'MaskPaintOutput') return;

        const imageW = node.widgets?.find((w) => w.name === 'image');
        if (!imageW) {
            console.error('[MaskPaintOutput] 找不到名为 image 的 widget');
            return;
        }
        // 这个 widget 是给遮罩编辑器 / 粘贴 / 拖放写文件名用的，不该让用户手填
        imageW.hidden = true;

        // Vue 模式下节点内预览读的是 output store（由运行结果填充），赋值 node.imgs
        // 不会刷新；所以写入后对单个节点做一次部分执行，让预览如实更新。
        // image_input 是可选输入，单节点 prompt 能过校验，且不会牵动下游。
        // 注意：onExecuted 回调发生在队列尚未排空时，此刻 queuePrompt 会因
        // processingQueue 直接返回 false（app.ts:1830-1832）——被丢弃的刷新要等队列
        // 空闲（status 事件 queue_remaining=0）再补，否则预览会停在旧图上。
        let pendingRefresh = false;
        api.addEventListener('status', ({ detail }) => {
            if (!pendingRefresh) return;
            if (!node.graph) { pendingRefresh = false; return; }   // 节点已删除就别再排队
            if ((detail?.status?.exec_info?.queue_remaining ?? 1) !== 0) return;
            pendingRefresh = false;
            refreshByRunning();
        });

        async function refreshByRunning() {
            try {
                // 重载签名：queuePrompt(number, batchCount, queueNodeIds[])，
                // 第三参传数组即部分执行，只跑本节点（app.ts:1796-1808）
                const queued = await app.queuePrompt(1, 1, [node.id]);
                if (queued === false) pendingRefresh = true;
                return queued;
            } catch (e) {
                console.warn('[MaskPaintOutput] 单节点刷新失败，预览将在下次运行时更新', e);
                node.imgs = node.imgs || [];
                node.graph?.setDirtyCanvas(true, true);
                return false;
            }
        }

        async function applyFile(file) {
            const info = await uploadBlob(file, file.name || `pasted-${Date.now()}.png`);
            const sub = info.subfolder || '';
            discardEditorSession(imageW.value);
            imageW.value = `${sub ? sub + '/' : ''}${info.name} [${info.type || 'input'}]`;
            // 经典画布模式下的预览靠 node.imgs；Vue 模式靠下面的单节点执行
            node.imgs = [{ filename: info.name, subfolder: sub, type: info.type || 'input' }];
            node.previewMediaType = 'image';
            // 手动传图视为「已对齐当前上游」：记下最近一次运行看到的上游指纹，
            // 否则紧接着的刷新运行会把它当成未接管的上游新图、又把刚写的 widget 清掉。
            node.properties[FP_BASE] = node.properties[FP_LAST] ?? null;
            node.graph?.setDirtyCanvas(true, true);
            await refreshByRunning();
        }

        // ---------- 取图模式：什么时候作废当前绘制、改用上游图 ----------
        // 遮罩编辑器优先读 image widget（useMaskEditorLoader.ts:106-111），widget 不清空
        // 就会在旧图上画；后端也以 widget 为准，输出同样停在旧图旧遮罩上。
        // 「上游是不是换了新图」用后端随 ui 发来的指纹 psl_fp（形状 + 抽样 crc32）判定：
        // 前端拿不到张量，而 app.nodeOutputs 里的文件名对很多上游节点根本不存在
        // （LoadImage、本包的 ImageContainer 都不往 ui 写图片）。
        // 两个触发点：
        //   A. stop_here 由关到开（用户明确要重新取图），或排队时指纹已对不上
        //   B. 停在取图模式期间上游出了新图 —— onExecuted 里比对本次运行的指纹
        // 记录都存在 node.properties 上，随工作流 JSON 保存，刷新/切换工作流不会误清。
        const stopW = node.widgets?.find((w) => w.name === 'stop_here');
        const FP_BASE = 'psl_adopted_fp';      // 当前 widget 对齐的上游指纹
        const FP_LAST = 'psl_last_fp';         // 最近一次运行看到的上游指纹
        const LAST_STOP = 'psl_last_stop';

        // ---- 旧会话作废：让遮罩编辑器不显示已被换掉的旧图 ----
        // 编辑器加载器优先读 Vue widget store 里的 image 值；那个 store 扩展写不进
        // （只有编辑器保存时核心才双写 store + LGraph widget），所以采纳上游或粘贴
        // 换图后，store 里残留的上一个会话文件名会让编辑器显示旧图+旧遮罩。
        // 加载器对 clipspace 会话有官方回退：图层文件加载失败就改用 node.images
        // （节点当前预览）。这里调用后端路由删掉旧会话的两个图层文件来触发回退；
        // clipspace-painted-masked 本体不动，其他节点可能还引用它。
        const SESSION_RE = /^clipspace-painted-masked-(\d+)\.png/;

        function discardEditorSession(value) {
            if (typeof value !== 'string' || !SESSION_RE.test(value.trim())) return;
            fetch('/psl/maskpaint/discard_session', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ name: value }),
            }).catch((e) => {
                console.warn('[MaskPaintOutput] 作废旧遮罩会话图层失败（编辑器若显示旧图，重画一次即恢复）', e);
            });
        }

        function readFp(output) {
            const fp = output?.psl_fp;
            return Array.isArray(fp) ? (fp[0] ?? null) : (fp ?? null);
        }

        // ---- 关→开切换即取图：不再要求用户再手动点一次运行 ----
        // 采纳只发生在运行钩子里（serializeValue / onExecuted）；用户切到取图模式后
        // 若不运行就打开遮罩编辑器，编辑器如实加载的还是节点画布上的旧图（编辑器
        // 内容 = 节点当前预览）。所以在开关 false→true 时立即单节点跑一轮：
        // serializeValue 里 justEntered 命中 → 作废当前图 → 本次运行直接取上游图。
        // 仅在上游确有连线时才跑；工作流加载直设 value 不走 callback，onConfigure 同步基准。
        let prevStop = stopW?.value;
        if (stopW) {
            const prevStopCb = stopW.callback;
            stopW.callback = function (value) {
                prevStopCb?.apply(this, arguments);
                const was = prevStop;
                prevStop = value;
                const hasUpstream = node.inputs?.some((inp) => inp.name === 'image_input' && inp.link != null);
                if (value === true && was === false && node.graph && hasUpstream) {
                    refreshByRunning();
                }
            };
        }

        function adoptUpstream(fp) {
            node.properties[FP_BASE] = fp;
            discardEditorSession(imageW.value);
            imageW.value = '';
            node.imgs = undefined;
        }

        // 触发点 A。放在 serializeValue 里是因为它是扩展能拿到的、构建 prompt 之前的
        // 唯一钩子（executionUtil.ts:134-136 每次排队都会调）：在这里清空 widget，
        // 同一次运行就直接改用上游图，不必像 onExecuted 那样再多跑一轮。
        imageW.serializeValue = () => {
            const taking = !!stopW?.value;
            const justEntered = taking && node.properties[LAST_STOP] === false;
            node.properties[LAST_STOP] = taking;
            const fp = node.properties[FP_LAST] ?? null;
            // 没连上游（或上游还没跑过）时 fp 为 null，无图可取，不作废当前绘制。
            if (taking && fp && (justEntered || fp !== node.properties[FP_BASE])) {
                adoptUpstream(fp);
            }
            return imageW.value;
        };

        // 触发点 B：排队时还看不到本次运行上游会出什么图，运行后补一次接管。
        node.onExecuted = (output) => {
            const fp = readFp(output);
            if (fp) node.properties[FP_LAST] = fp;
            if (!stopW?.value) return;             // 输出模式：当前图锁定，不动
            if (!fp || fp === node.properties[FP_BASE]) return;
            const hadWidget = !!imageW.value;
            adoptUpstream(fp);
            if (!hadWidget) return;
            // 刚才那次运行用的还是旧 widget，预览停在旧图上；再跑一次本节点让预览
            // 换成接管后的上游图。stop_here 开着，下游不会被牵动。
            refreshByRunning();
        };

        // 官方「粘贴图像」菜单项的判定条件：存在 pasteFiles 函数。
        // 注意核心会先调 pasteFile 再调 pasteFiles，两个都实现会重复上传，只实现后者。
        node.pasteFiles = async (files) => {
            const img = [...(files || [])].find((f) => f.type && f.type.startsWith('image/'));
            if (!img) return false;
            try {
                await applyFile(img);
            } catch (e) {
                console.error('[MaskPaintOutput] 粘贴上传失败', e);
                window.alert(`粘贴上传失败 —— ${e.message}`);
            }
            return true;
        };

        // 常驻粘贴按钮：右键再点三层菜单太慢。读剪贴板与官方菜单项同源
        // （navigator.clipboard.read()）；不可用时提示走菜单或 Ctrl+V。
        async function readImageFromClipboard() {
            if (!navigator.clipboard?.read) return null;
            for (const item of await navigator.clipboard.read()) {
                const type = item.types.find((t) => t.startsWith('image/'));
                if (type) return await item.getType(type);
            }
            return null;
        }

        node.addWidget('button', '📋 粘贴底图', null, async () => {
            let blob = null;
            try {
                blob = await readImageFromClipboard();
            } catch (e) {
                console.warn('[MaskPaintOutput] 剪贴板读取被拒绝', e);
            }
            if (!blob) {
                window.alert(
                    '剪贴板里没有图片，或浏览器拒绝了读取。\n'
                    + '（navigator.clipboard 需要安全上下文，局域网 IP + http 会被拒）\n\n'
                    + '请改用右键菜单的「粘贴图像」，或选中本节点后按 Ctrl+V。'
                );
                return;
            }
            try {
                await applyFile(new File([blob], `pasted-${Date.now()}.png`, { type: blob.type }));
            } catch (e) {
                console.error('[MaskPaintOutput] 粘贴上传失败', e);
                window.alert(`粘贴上传失败 —— ${e.message}`);
            }
        }, BTN);

        // 拖图片文件到节点上 = 换底图，与官方 LoadImage 行为一致；
        // 返回 true 以抑制全局的「每张图建一个 LoadImage」行为。
        node.onDragOver = (e) => {
            if (!e?.dataTransfer?.types?.includes('Files')) return false;
            return [...e.dataTransfer.items].some((it) => it.type.startsWith('image/'));
        };
        node.onDragDrop = async (e) => {
            const file = [...(e.dataTransfer?.files || [])].find((f) => f.type.startsWith('image/'));
            if (!file) return false;
            e.preventDefault();
            e.stopPropagation();
            try {
                await applyFile(file);
            } catch (err) {
                console.error('[MaskPaintOutput] 拖放上传失败', err);
            }
            return true;
        };

        // 工作流加载恢复 widget 值时不走 callback，prevStop 基准在这里同步，
        // 避免加载后第一次手动切换时把陈旧基准当成 false→true 误触发取图运行。
        const prevConfigure = node.onConfigure;
        node.onConfigure = function (info) {
            const r = prevConfigure?.apply(this, arguments);
            prevStop = stopW?.value;
            return r;
        };
    },
});
