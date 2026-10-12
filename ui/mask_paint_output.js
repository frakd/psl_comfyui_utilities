import { app } from '../../scripts/app.js';

// ============================================================
//  Mask Paint Output — 粘贴底图按钮
//  把剪贴板里的图片上传后写进名为 image 的 widget，也就是官方遮罩编辑器
//  保存时写入的同一个槽，所以 Python 侧的 execute 不需要任何改动。
// ============================================================

// 按钮 widget 必须显式 serialize:false —— addWidget 不会自动加，
// 否则按钮的名字和值会漏进 prompt，白白参与运行缓存签名。
const BTN = { serialize: false };

async function uploadBlob(blob, filename) {
    const fd = new FormData();
    fd.append('image', blob, filename);
    fd.append('overwrite', 'false');
    // 故意不传 subfolder：上传到 input 根目录，widget 值就是 "<name> [input]"，
    // 与遮罩编辑器自己保存时写入的格式逐字一致，编辑器一定能加载它。
    fd.append('type', 'input');
    const r = await fetch('/upload/image', { method: 'POST', body: fd });
    if (!r.ok) throw new Error(`上传失败 HTTP ${r.status}: ${await r.text()}`);
    return r.json();   // { name, subfolder, type }
}

function applyToNode(node, imageW, info) {
    const sub = info.subfolder || '';
    imageW.value = `${sub ? sub + '/' : ''}${info.name} [${info.type || 'input'}]`;
    // 同步预览，否则要等下一次运行才会显示刚粘贴的图
    node.imgs = [{ filename: info.name, subfolder: sub, type: info.type || 'input' }];
    node.previewMediaType = 'image';
    node.setSize?.(node.computeSize?.());
}

async function readImageFromClipboard() {
    if (!navigator.clipboard?.read) return null;
    for (const item of await navigator.clipboard.read()) {
        const type = item.types.find((t) => t.startsWith('image/'));
        if (type) return { blob: await item.getType(type), ext: type.split('/')[1] || 'png' };
    }
    return null;
}

app.registerExtension({
    name: 'Comfy.PersonalUtilities.MaskPaintOutput',

    async nodeCreated(node) {
        if (node.comfyClass !== 'MaskPaintOutput') return;

        const imageW = node.widgets?.find((w) => w.name === 'image');
        if (!imageW) {
            console.error('[MaskPaintOutput] 找不到名为 image 的 widget，粘贴按钮不可用');
            return;
        }

        async function pasteBase(label) {
            let shot = null;
            try {
                shot = await readImageFromClipboard();
            } catch (e) {
                console.warn('[MaskPaintOutput] 剪贴板读取被拒绝，改用 Ctrl+V', e);
            }
            if (!shot) {
                window.alert(
                    `${label}：剪贴板里没有图片，或浏览器拒绝了读取。\n\n`
                    + 'navigator.clipboard 需要安全上下文（localhost 可用，局域网 IP + http 会被拒）。\n'
                    + '请改为选中本节点后按 Ctrl+V。'
                );
                return;
            }
            try {
                const info = await uploadBlob(shot.blob, `paste-${Date.now()}.${shot.ext}`);
                applyToNode(node, imageW, info);
            } catch (e) {
                console.error('[MaskPaintOutput] 粘贴上传失败', e);
                window.alert(`${label}：上传失败 —— ${e.message}`);
            }
        }

        node.addWidget('button', '📋 粘贴底图', null, () => pasteBase('粘贴底图'), BTN);
        node.addWidget('button', '🧹 清除底图', null, () => {
            imageW.value = '';
            node.imgs = null;
            node.setSize?.(node.computeSize?.());
        }, BTN);

        // Ctrl+V 兜底：仅当本节点被选中时生效，免得抢走别的节点的粘贴
        document.addEventListener('paste', async (e) => {
            const selected = app.canvas?.selected_nodes || {};
            const isTarget = node === app.canvas?.currentNode || Object.values(selected).includes(node);
            if (!isTarget) return;
            for (const it of e.clipboardData?.items || []) {
                if (it.kind !== 'file' || !it.type.startsWith('image/')) continue;
                const f = it.getAsFile();
                if (!f) continue;
                e.preventDefault();
                try {
                    const info = await uploadBlob(f, f.name || `paste-${Date.now()}.png`);
                    applyToNode(node, imageW, info);
                } catch (err) {
                    console.error('[MaskPaintOutput] 粘贴上传失败', err);
                }
                return;
            }
        });
    },
});
