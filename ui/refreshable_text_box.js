import { app } from '../../scripts/app.js';

// ============================================================
//  Refreshable Text Box — 自刷新文本框
//  核心是 serializeValue：显示会变，但发给后端的值在未锁定时恒为常量，
//  于是 prompt 字节稳定、运行缓存不失效。
//
//  按钮区是 DOM widget（2×2 网格，serialize:false，不进 prompt/工作流）：
//    第一行  锁定开关 + 还原上游（状态类操作）
//    第二行  复制 + 粘贴（剪贴板操作，按「复制粘贴」阅读顺序排列）
//  复制/粘贴的区分靠三重编码，扫一眼即可分辨：
//    颜色：蓝 = 内容出去（复制），绿 = 内容进来（粘贴）
//    图标：双方块 = 复制，写字板 = 粘贴
//    tooltip：明写「文本框 → 剪贴板 / 剪贴板 → 文本框」
//  锁定态：锁定按钮琥珀色填充 + 光晕，锁没锁一眼可见。
// ============================================================

let stylesInjected = false;

function injectStyles() {
    if (stylesInjected || document.getElementById('rtx-styles')) return;
    stylesInjected = true;
    const s = document.createElement('style');
    s.id = 'rtx-styles';
    s.textContent = `
        .rtx-btns {
            display: grid;
            grid-template-columns: 1fr 1fr;
            grid-template-rows: 1fr 1fr;
            gap: 6px;
            padding: 6px 8px;
            box-sizing: border-box;
            width: 100%;
            height: 100%;
        }
        .rtx-btn {
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 6px;
            border: 1px solid #555;
            border-radius: 5px;
            background: #3d3d3d;
            color: #e5e5e5;
            font-size: 12px;
            cursor: pointer;
            padding: 0 6px;
            min-width: 0;
            white-space: nowrap;
            overflow: hidden;
            user-select: none;
        }
        .rtx-btn:hover { filter: brightness(1.18); }
        .rtx-btn:active { filter: brightness(0.92); }
        .rtx-btn svg { width: 14px; height: 14px; flex: 0 0 auto; }
        /* 剪贴板行：蓝 = 内容出去（复制），绿 = 内容进来（粘贴） */
        .rtx-btn.copy { background: #2c5d8f; border-color: #3f79ad; }
        .rtx-btn.paste { background: #2c7a4b; border-color: #3f9a63; }
        /* 锁定态：琥珀色填充 + 光晕 */
        .rtx-btn.lock.locked {
            background: #b57a10;
            border-color: #f0b429;
            color: #fff;
            font-weight: 600;
            box-shadow: 0 0 0 1px rgba(240, 180, 41, 0.55), 0 0 10px rgba(240, 180, 41, 0.35);
        }
    `;
    document.head.appendChild(s);
}

// Feather 风格线性图标（stroke: currentColor，自动跟随按钮文字色）
const ICONS = {
    lockClosed: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="11" width="18" height="11" rx="2"/><path d="M7 11V7a5 5 0 0 1 10 0v4"/></svg>',
    lockOpen: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="11" width="18" height="11" rx="2"/><path d="M7 11V7a5 5 0 0 1 9.9-1"/></svg>',
    copy: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>',
    paste: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M16 4h2a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2h2"/><rect x="8" y="2" width="8" height="4" rx="1" ry="1"/></svg>',
    restore: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="1 4 1 10 7 10"/><path d="M3.51 15a9 9 0 1 0 2.13-9.36L1 10"/></svg>',
};

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
        injectStyles();

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

        // ---------- 按钮区（DOM widget，2×2） ----------
        // 高度 min=max 钉死：布局分配里 maxSize 缺省是 Infinity（会吃掉节点拉高的
        // 全部剩余空间，参考 ImageContainer），按钮条不该跟着长，所以两边都钉在 86px
        // （上下 padding 12 + 行距 6 + 两行各 34）。
        const root = document.createElement('div');
        root.className = 'rtx-btns';
        root.style.setProperty('--comfy-widget-min-height', '86px');
        root.style.setProperty('--comfy-widget-max-height', '86px');

        function makeBtn(cls, iconSvg, text, title, onClick) {
            const b = document.createElement('button');
            b.type = 'button';
            b.className = `rtx-btn ${cls}`;
            b.title = title;
            b.innerHTML = `${iconSvg}<span></span>`;
            b.querySelector('span').textContent = text;
            b.addEventListener('click', (e) => {
                e.preventDefault();
                e.stopPropagation();
                onClick();
            });
            return b;
        }

        const lockBtn = makeBtn('lock', ICONS.lockOpen, '允许覆盖',
            '未锁定：运行时自动用上游文本刷新显示，且不影响运行缓存（点击锁定）',
            () => { lockW.value = !lockW.value; syncLockUI(); });

        const restoreBtn = makeBtn('', ICONS.restore, '还原上游',
            '把文本框内容还原成最近一次运行从上游收到的文本（没跑过则不动）',
            () => {
                // 没跑过（例如刚刷新页面）时 upstream 为 null，此时不动文本，
                // 免得一键把用户内容清成空。
                if (upstream !== null) textW.value = upstream;
            });

        const copyBtn = makeBtn('copy', ICONS.copy, '复制全部',
            '复制全部：文本框 → 剪贴板',
            async () => { await writeClipboard(textW.value ?? ''); });

        const pasteBtn = makeBtn('paste', ICONS.paste, '粘贴覆盖',
            '粘贴覆盖：剪贴板 → 文本框（覆盖现有内容）',
            async () => {
                const t = await readClipboard();
                if (t !== null) textW.value = t;
            });

        // 网格按行填充：第一行 锁定/还原，第二行 复制/粘贴
        root.append(lockBtn, restoreBtn, copyBtn, pasteBtn);

        function syncLockUI() {
            const locked = !lockW.value;   // allow_overwrite=false 即锁定
            lockBtn.classList.toggle('locked', locked);
            const icon = lockBtn.querySelector('svg');
            if (icon) icon.outerHTML = locked ? ICONS.lockClosed : ICONS.lockOpen;
            lockBtn.querySelector('span').textContent = locked ? '禁止覆盖' : '允许覆盖';
            lockBtn.title = locked
                ? '已锁定：文本框内容生效并输出，运行不再刷新显示（点击解锁）'
                : '未锁定：运行时自动用上游文本刷新显示，且不影响运行缓存（点击锁定）';
        }

        // serialize:false：按钮条不进 prompt；hideOnZoom:false：缩小时不变成占位矩形
        node.addDOMWidget('rtx_buttons', 'HTML', root, { serialize: false, hideOnZoom: false });

        const prevOnExecuted = node.onExecuted;
        node.onExecuted = function (output) {
            prevOnExecuted?.apply(this, arguments);
            const t = output?.text;
            if (Array.isArray(t) && t.length > 0) {
                upstream = t[0];
                // 只有「允许覆盖」时才刷新显示；锁定状态下绝不覆盖用户改过的文本
                if (lockW.value) textW.value = upstream;
            }
            syncLockUI();
        };

        const prevOnConfigure = node.onConfigure;
        node.onConfigure = function (info) {
            prevOnConfigure?.apply(this, arguments);
            syncLockUI();
        };

        syncLockUI();
    },
});
