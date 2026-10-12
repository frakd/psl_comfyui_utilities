import { app } from '../../scripts/app.js';

// ============================================================
//  Mask Paint Output — 底图上传 / 清除遮罩
//
//  粘贴走官方机制：Vue 节点菜单里「粘贴图像」的判定只有
//  typeof node.pasteFiles === 'function'（useImageMenuOptions.ts:18-20），
//  菜单项本身由核心提供、读 navigator.clipboard 后调用 node.pasteFiles([file])。
//  所以这里只实现 pasteFiles，不自己造按钮和剪贴板监听（否则会和核心重复上传）。
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

function viewUrl(info) {
    const q = new URLSearchParams({
        filename: info.filename || info.name,
        subfolder: info.subfolder || '',
        type: info.type || 'input',
    });
    return `/view?${q.toString()}`;
}

// widget 值形如 "[子目录/]文件名 [input]"，拆回 {name, subfolder, type}
function parseWidgetValue(v) {
    const type = (v.match(/\[(input|output|temp)\]$/) || [, 'input'])[1];
    const bare = v.replace(/\s*\[(input|output|temp)\]$/, '');
    const i = bare.lastIndexOf('/');
    return i >= 0
        ? { name: bare.slice(i + 1), subfolder: bare.slice(0, i), type }
        : { name: bare, subfolder: '', type };
}

// 把当前底图重传一份 alpha 全不透明的副本：RGB 保留、遮罩归零。
// 这样「清除遮罩」不会动底图 —— 遮罩与底图在编辑器保存的文件里是同一个
// PNG 的 alpha 与 RGB，没法只删其一，只能换一份没有 alpha 的文件。
async function stripAlphaToBlob(info) {
    const im = new Image();
    im.src = viewUrl(info);
    await im.decode();
    const c = document.createElement('canvas');
    c.width = im.naturalWidth;
    c.height = im.naturalHeight;
    const g = c.getContext('2d');
    g.drawImage(im, 0, 0);
    const d = g.getImageData(0, 0, c.width, c.height);
    for (let i = 3; i < d.data.length; i += 4) d.data[i] = 255;
    g.putImageData(d, 0, 0);
    return new Promise((res, rej) => c.toBlob((b) => (b ? res(b) : rej(new Error('toBlob 失败'))), 'image/png'));
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
            node.graph?.setDirtyCanvas(true, true);
            await refreshByRunning();
        }

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

        // 清除遮罩：只把 alpha 抹掉重传一份，底图 RGB 原样保留。
        node.addWidget('button', '🧹 清除遮罩', null, async () => {
            const cur = imageW.value;
            if (!cur) return;
            try {
                const blob = await stripAlphaToBlob(parseWidgetValue(cur));
                const info = await uploadBlob(blob, `base-${Date.now()}.png`);
                const sub = info.subfolder || '';
                imageW.value = `${sub ? sub + '/' : ''}${info.name} [${info.type || 'input'}]`;
                node.imgs = [{ filename: info.name, subfolder: sub, type: info.type || 'input' }];
                node.graph?.setDirtyCanvas(true, true);
                await refreshByRunning();
            } catch (e) {
                console.error('[MaskPaintOutput] 清除遮罩失败', e);
                window.alert(`清除遮罩失败 —— ${e.message}`);
            }
        }, BTN);
    },
});
