import { app } from '../../scripts/app.js';

// ============================================================
//  Refreshable Text Box — 自刷新文本框
//  核心是 serializeValue：显示会变，但发给后端的值在未锁定时恒为常量，
//  于是 prompt 字节稳定、运行缓存不失效。
// ============================================================

const UNLOCKED_LABEL = '允许覆盖';
const LOCKED_LABEL = '禁止覆盖';

// 按钮 widget 必须显式 serialize:false —— addWidget 不会自动加，
// 否则按钮的名字和值会漏进 prompt，白白参与运行缓存签名。
const BTN = { serialize: false };

async function readClipboard() {
    try {
        return await navigator.clipboard.readText();
    } catch (e) {
        // navigator.clipboard 需要安全上下文：localhost/127.0.0.1 可用，
        // 局域网 IP + http 会被浏览器拒绝，这里退回手输。
        const v = window.prompt('浏览器限制了剪贴板读取，请手动粘贴：', '');
        return v === null ? null : v;
    }
}

async function writeClipboard(text) {
    try {
        await navigator.clipboard.writeText(text);
    } catch (e) {
        window.prompt('浏览器限制了剪贴板写入，请手动复制：', text);
    }
}

app.registerExtension({
    name: 'Comfy.RefreshableTextBox',

    async nodeCreated(node) {
        if (node.comfyClass !== 'RefreshableTextBox') return;

        const textW = node.widgets?.find((w) => w.name === 'locked_text');
        const lockW = node.widgets?.find((w) => w.name === 'allow_overwrite');
        if (!textW || !lockW) return;

        // 状态只通过下面的按钮展示和切换；布尔 widget 留着是给它当数据源和持久化载体，
        // 界面上再显示一份就重了。hidden 不影响它进 prompt 和存工作流。
        lockW.hidden = true;

        // 最近一次从上游收到的文本，供「还原上游」按钮使用。
        // 只在运行时存在，刷新页面后要等下一次运行才会重新填充。
        let upstream = null;

        // 未锁定（允许覆盖）时发送常量空串：prompt 对本节点保持字节稳定，
        // 缓存签名不变，本节点与下游都不会因为文本刷新而失效。
        // 锁定后发送真实文本，此时缓存从本节点起失效一次是预期行为。
        textW.serializeValue = () => (lockW.value ? '' : textW.value);

        function syncLockBtn() {
            if (lockBtn) lockBtn.name = lockW.value ? UNLOCKED_LABEL : LOCKED_LABEL;
        }

        let lockBtn = null;
        lockBtn = node.addWidget('button', UNLOCKED_LABEL, null, () => {
            lockW.value = !lockW.value;
            syncLockBtn();
        }, BTN);

        node.addWidget('button', '📋 粘贴覆盖', null, async () => {
            const t = await readClipboard();
            if (t !== null) textW.value = t;
        }, BTN);

        node.addWidget('button', '📄 复制全部', null, async () => {
            await writeClipboard(textW.value ?? '');
        }, BTN);

        node.addWidget('button', '↺ 还原上游', null, () => {
            // 没跑过（例如刚刷新页面）时 upstream 为 null，此时不动文本，
            // 免得一键把用户内容清成空。
            if (upstream !== null) textW.value = upstream;
        }, BTN);

        const prevOnExecuted = node.onExecuted;
        node.onExecuted = function (output) {
            prevOnExecuted?.apply(this, arguments);
            const t = output?.text;
            if (Array.isArray(t) && t.length > 0) {
                upstream = t[0];
                // 只有「允许覆盖」时才刷新显示；锁定状态下绝不覆盖用户改过的文本
                if (lockW.value) textW.value = upstream;
            }
            syncLockBtn();
        };

        const prevOnConfigure = node.onConfigure;
        node.onConfigure = function (info) {
            prevOnConfigure?.apply(this, arguments);
            syncLockBtn();
        };

        syncLockBtn();
    },
});
