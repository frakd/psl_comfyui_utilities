import { app } from '../../scripts/app.js';

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
        async function refreshByRunning() {
            try {
                // 重载签名：queuePrompt(number, batchCount, queueNodeIds[])，
                // 第三参传数组即部分执行，只跑本节点（app.ts:1796-1808）
                await app.queuePrompt(1, 1, [node.id]);
            } catch (e) {
                console.warn('[MaskPaintOutput] 单节点刷新失败，预览将在下次运行时更新', e);
                node.imgs = node.imgs || [];
                node.graph?.setDirtyCanvas(true, true);
            }
        }

        async function applyFile(file) {
            const info = await uploadBlob(file, file.name || `pasted-${Date.now()}.png`);
            const sub = info.subfolder || '';
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

        function readFp(output) {
            const fp = output?.psl_fp;
            return Array.isArray(fp) ? (fp[0] ?? null) : (fp ?? null);
        }

        function adoptUpstream(fp) {
            node.properties[FP_BASE] = fp;
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
    },
});
