import { app } from '../../scripts/app.js';

// ============================================================
//  Image Container — 多图容器
//  上传 / 拖入 / 粘贴图片，点选决定输出槽，长按拖拽排序。
//
//  顺序规则：单击选中即按点击先后占用输出槽 1..6；一旦拖动排序，
//  输出顺序改为按容器内视觉顺序重新推导，保证「所见即所得」。
//  缩略图左上角的序号徽章始终等于它实际占用的输出槽。
// ============================================================

const SUBFOLDER = 'imageContainer';
const SLOT_COUNT = 6;
const ENTRY_CLS = 'ic-entry';
const DRAG_HOLD_MS = 200;   // 按住这么久即进入拖拽
const DRAG_MOVE_PX = 6;     // 或位移超过这么多像素即进入拖拽
const LAZY_ROOT_MARGIN = '100px';

let stylesInjected = false;

function injectStyles() {
    if (stylesInjected || document.getElementById('ic-styles')) return;
    stylesInjected = true;
    const s = document.createElement('style');
    s.id = 'ic-styles';
    s.textContent = `
        .ic-root {
            display: flex;
            flex-direction: column;
            width: 100%;
            background: #1f1f1f;
            border-radius: 6px;
            margin-top: 5px;
            box-sizing: border-box;
            overflow: hidden;
        }
        /* 头部固定：列表再长也不会把上传按钮顶走 */
        .ic-header {
            display: flex;
            align-items: center;
            gap: 6px;
            padding: 6px 8px;
            background: #2a2a2a;
            border-bottom: 1px solid #444;
            flex: 0 0 auto;
        }
        .ic-btn {
            background: rgba(47, 121, 191, 0.85);
            color: #fff;
            border: 0;
            border-radius: 4px;
            padding: 4px 8px;
            font-size: 12px;
            cursor: pointer;
            white-space: nowrap;
        }
        .ic-btn:hover { background: rgba(84, 153, 217, 0.9); }
        .ic-btn.danger { background: rgba(163, 54, 54, 0.85); }
        .ic-btn.danger:hover { background: rgba(210, 66, 66, 0.9); }
        .ic-btn.plain { background: #3d3d3d; border: 1px solid #555; padding: 3px 7px; }
        .ic-btn.plain:hover { background: #555; }
        .ic-spacer { flex: 1 1 auto; }
        .ic-cols {
            width: 40px;
            font-size: 13px;
            text-align: center;
            background: #111;
            color: #fff;
            border: 1px solid #444;
            border-radius: 3px;
        }
        .ic-scroll {
            overflow-y: auto;
            overflow-x: hidden;
            flex: 1 1 auto;
            min-height: 60px;
        }
        .ic-grid {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            padding: 8px;
            align-content: flex-start;
        }
        .ic-empty {
            color: #777;
            font-size: 12px;
            padding: 18px 10px;
            text-align: center;
            width: 100%;
        }
        .ic-entry {
            position: relative;
            border-radius: 4px;
            overflow: hidden;
            background: #161616;
            border: 2px solid transparent;
            box-sizing: border-box;
            cursor: pointer;
            user-select: none;
            -webkit-user-select: none;
            touch-action: none;
        }
        .ic-entry img {
            display: block;
            width: 100%;
            height: 100%;
            object-fit: cover;
            pointer-events: none;
        }
        .ic-entry.selected { border-color: #0095ff; }
        .ic-entry.dragging { opacity: 0.3; }
        .ic-ghost {
            position: fixed;
            opacity: 0.75;
            pointer-events: none;
            z-index: 10000;
            box-shadow: 0 8px 16px rgba(0,0,0,0.45);
            border-radius: 4px;
            overflow: hidden;
        }
        .ic-badge {
            position: absolute;
            top: 4px;
            left: 4px;
            min-width: 20px;
            height: 20px;
            line-height: 20px;
            padding: 0 4px;
            text-align: center;
            font-size: 12px;
            font-weight: bold;
            color: #fff;
            background: #0095ff;
            border-radius: 10px;
            z-index: 5;
            pointer-events: none;
        }
        .ic-del {
            position: absolute;
            top: 4px;
            right: 4px;
            width: 20px;
            height: 20px;
            line-height: 18px;
            text-align: center;
            font-size: 15px;
            color: #fff;
            background: rgba(75,75,75,0.9);
            border-radius: 4px;
            cursor: pointer;
            opacity: 0;
            transition: opacity 0.15s, background 0.15s;
            z-index: 6;
        }
        .ic-entry:hover .ic-del { opacity: 1; }
        .ic-del:hover { background: rgba(200,60,60,0.95); }
        .ic-status {
            padding: 4px 8px;
            font-size: 11px;
            color: #aaa;
            background: #2a2a2a;
            border-top: 1px solid #444;
            flex: 0 0 auto;
            font-family: monospace;
        }
        .ic-root.drop-active .ic-grid { outline: 2px dashed #0095ff; outline-offset: -4px; }
    `;
    document.head.appendChild(s);
}

function newId() {
    return `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 9)}`;
}

function viewUrl(entry) {
    const q = new URLSearchParams({
        filename: entry.name,
        subfolder: entry.subfolder || '',
        type: entry.type || 'input',
    });
    return `/view?${q.toString()}`;
}

async function uploadBlobs(items) {
    const uploaded = [];
    for (const it of items) {
        const blob = it.blob;
        const name = it.name || `paste-${Date.now()}-${uploaded.length}.png`;
        const fd = new FormData();
        fd.append('image', blob, name);
        fd.append('overwrite', 'false');
        fd.append('subfolder', SUBFOLDER);
        fd.append('type', 'input');
        try {
            const r = await fetch('/upload/image', { method: 'POST', body: fd });
            if (!r.ok) {
                console.error('[ImageContainer] 上传失败', name, r.status, await r.text());
                continue;
            }
            const j = await r.json();
            // 每个条目配唯一 id：服务器会对内容相同的文件去重返回同一个文件名，
            // 若用文件名当键，删一张会把同名的另一张一起删掉。
            uploaded.push({
                id: newId(),
                name: j.name,
                subfolder: j.subfolder || SUBFOLDER,
                type: j.type || 'input',
            });
        } catch (e) {
            console.error('[ImageContainer] 上传异常', name, e);
        }
    }
    return uploaded;
}

function imageFilesFromDataTransfer(dt) {
    const out = [];
    if (!dt) return out;
    for (const f of dt.files || []) {
        if (f.type && f.type.startsWith('image/')) out.push({ blob: f, name: f.name });
    }
    return out;
}

app.registerExtension({
    name: 'Comfy.PersonalUtilities.ImageContainer',

    async nodeCreated(node) {
        if (node.comfyClass !== 'ImageContainer') return;
        injectStyles();

        const selectedW = node.widgets?.find((w) => w.name === 'selected_json');
        const poolW = node.widgets?.find((w) => w.name === 'pool_json');
        if (!selectedW || !poolW) {
            console.error('[ImageContainer] 找不到 selected_json / pool_json widget');
            return;
        }
        selectedW.hidden = true;
        poolW.hidden = true;
        // 容器完整状态不进 prompt：否则上传/排序但没改选中集时，下游缓存也会白丢一遍。
        // widget.serialize（存工作流 json）与 options.serialize（进 prompt）是两个独立开关。
        poolW.options = poolW.options || {};
        poolW.options.serialize = false;

        // ---------- 状态 ----------
        let pool = [];          // [{id,name,subfolder,type}]，容器内显示顺序
        let selectedIds = [];   // 有序，最多 SLOT_COUNT 个
        let cols = 4;

        function readPoolState() {
            try {
                const st = JSON.parse(poolW.value || '{}');
                pool = Array.isArray(st.items) ? st.items.filter((x) => x && x.name) : [];
                cols = Number.isFinite(st.cols) ? Math.min(10, Math.max(1, st.cols | 0)) : 4;
            } catch (e) {
                pool = [];
                cols = 4;
            }
            // 选中列表按 id 过滤掉已不在容器里的项
            try {
                const sel = JSON.parse(selectedW.value || '[]');
                const ids = Array.isArray(sel) ? sel.map((x) => (x && x.id) || null).filter(Boolean) : [];
                const alive = new Set(pool.map((p) => p.id));
                selectedIds = ids.filter((id) => alive.has(id)).slice(0, SLOT_COUNT);
            } catch (e) {
                selectedIds = [];
            }
        }

        function writeState() {
            poolW.value = JSON.stringify({ cols, items: pool });
            const byId = new Map(pool.map((p) => [p.id, p]));
            selectedW.value = JSON.stringify(
                selectedIds.map((id) => byId.get(id)).filter(Boolean)
                    .map(({ name, subfolder, type }) => ({ name, subfolder, type }))
            );
        }

        // ---------- DOM ----------
        const root = document.createElement('div');
        root.className = 'ic-root';
        root.style.setProperty('--comfy-widget-min-height', '240px');
        root.style.setProperty('--comfy-widget-max-height', '620px');

        const header = document.createElement('div');
        header.className = 'ic-header';

        const uploadBtn = document.createElement('button');
        uploadBtn.className = 'ic-btn';
        uploadBtn.textContent = '📤 上传';
        const fileInput = document.createElement('input');
        fileInput.type = 'file';
        fileInput.multiple = true;
        fileInput.accept = 'image/*';
        fileInput.style.display = 'none';
        uploadBtn.onclick = () => fileInput.click();
        fileInput.onchange = async (e) => {
            const items = Array.from(e.target.files || []).map((f) => ({ blob: f, name: f.name }));
            e.target.value = '';   // 允许重复选同一个文件
            if (items.length) await addItems(items);
        };

        const pasteBtn = document.createElement('button');
        pasteBtn.className = 'ic-btn';
        pasteBtn.textContent = '📋 粘贴';
        pasteBtn.onclick = () => pasteFromClipboard(root);

        const clearBtn = document.createElement('button');
        clearBtn.className = 'ic-btn danger';
        clearBtn.textContent = '🗑 清空';
        clearBtn.onclick = () => {
            if (!pool.length) return;
            if (!confirm(`清空容器里的 ${pool.length} 张图？\n（只移除本节点的引用，不删除服务器上已上传的文件）`)) return;
            pool = [];
            selectedIds = [];
            writeState();
            render();
        };

        const spacer = document.createElement('div');
        spacer.className = 'ic-spacer';

        const minusBtn = document.createElement('button');
        minusBtn.className = 'ic-btn plain';
        minusBtn.textContent = '−';
        const colsInput = document.createElement('input');
        colsInput.className = 'ic-cols';
        colsInput.type = 'number';
        colsInput.min = '1';
        colsInput.max = '10';
        const plusBtn = document.createElement('button');
        plusBtn.className = 'ic-btn plain';
        plusBtn.textContent = '+';

        function setCols(v) {
            cols = Math.min(10, Math.max(1, v | 0));
            colsInput.value = cols;
            writeState();
            render();
        }
        minusBtn.onclick = () => setCols(cols - 1);
        plusBtn.onclick = () => setCols(cols + 1);
        colsInput.onchange = () => setCols(parseInt(colsInput.value, 10) || 4);

        header.append(uploadBtn, pasteBtn, clearBtn, spacer, minusBtn, colsInput, plusBtn);

        const scroll = document.createElement('div');
        scroll.className = 'ic-scroll';
        const grid = document.createElement('div');
        grid.className = 'ic-grid';
        scroll.appendChild(grid);
        scroll.addEventListener('wheel', (e) => e.stopPropagation());

        const status = document.createElement('div');
        status.className = 'ic-status';

        root.append(header, fileInput, scroll, status);

        // ---------- 渲染 ----------
        let observer = null;

        function render() {
            if (observer) { observer.disconnect(); observer = null; }
            grid.innerHTML = '';
            colsInput.value = cols;

            if (!pool.length) {
                const hint = document.createElement('div');
                hint.className = 'ic-empty';
                hint.textContent = '点「上传」、把图片文件拖到这里、或选中本节点后 Ctrl+V 粘贴';
                grid.appendChild(hint);
            } else {
                observer = new IntersectionObserver((entries) => {
                    for (const en of entries) {
                        if (!en.isIntersecting) continue;
                        const img = en.target;
                        if (img.dataset.src) {
                            img.src = img.dataset.src;
                            img.removeAttribute('data-src');
                        }
                        observer.unobserve(img);
                    }
                }, { root: scroll, rootMargin: LAZY_ROOT_MARGIN, threshold: 0.01 });

                const itemW = `calc(${100 / cols}% - ${8 - 8 / cols}px)`;
                pool.forEach((entry) => grid.appendChild(buildEntry(entry, itemW)));
            }
            updateStatus();
        }

        function updateStatus() {
            status.textContent = `容器 ${pool.length} 张 · 已选 ${selectedIds.length}/${SLOT_COUNT}`
                + (selectedIds.length < SLOT_COUNT ? ` · 未选中的输出槽会静默阻断下游` : '');
        }

        function buildEntry(entry, itemW) {
            const el = document.createElement('div');
            el.className = ENTRY_CLS;
            el.dataset.id = entry.id;
            el.style.width = itemW;
            el.style.aspectRatio = '1 / 1';

            const img = document.createElement('img');
            img.draggable = false;             // 禁掉浏览器原生的图片拖拽
            img.dataset.src = viewUrl(entry);  // 懒加载：进入视口才真正请求
            img.alt = entry.name;
            el.appendChild(img);
            observer.observe(img);

            const rank = selectedIds.indexOf(entry.id);
            if (rank >= 0) {
                el.classList.add('selected');
                const badge = document.createElement('div');
                badge.className = 'ic-badge';
                badge.textContent = String(rank + 1);
                el.appendChild(badge);
            }

            const del = document.createElement('div');
            del.className = 'ic-del';
            del.textContent = '×';
            del.title = '从容器移除（不删服务器文件）';
            del.addEventListener('pointerdown', (e) => { e.stopPropagation(); });
            del.addEventListener('click', (e) => {
                e.preventDefault();
                e.stopPropagation();
                removeEntry(entry.id);
            });
            el.appendChild(del);

            attachPointerDrag(el, entry.id);
            return el;
        }

        function removeEntry(id) {
            pool = pool.filter((p) => p.id !== id);
            selectedIds = selectedIds.filter((x) => x !== id);
            writeState();
            render();
        }

        async function addItems(items) {
            const uploaded = await uploadBlobs(items);
            if (!uploaded.length) return;
            pool = pool.concat(uploaded);
            writeState();
            render();
        }

        function toggleSelect(id) {
            const at = selectedIds.indexOf(id);
            if (at >= 0) {
                selectedIds.splice(at, 1);   // 取消输出，后面的自动前移
            } else {
                if (selectedIds.length >= SLOT_COUNT) {
                    status.textContent = `最多只能选 ${SLOT_COUNT} 张（输出槽只有 ${SLOT_COUNT} 个）`;
                    return;
                }
                selectedIds.push(id);      // 按点击先后占用下一个输出槽
            }
            writeState();
            render();
        }

        // ---------- 单击选中 / 长按拖拽排序 ----------
        function attachPointerDrag(el, id) {
            el.addEventListener('pointerdown', (e) => {
                if (e.button !== 0) return;
                if (e.target.closest('.ic-del')) return;
                e.preventDefault();
                e.stopPropagation();

                const startX = e.clientX, startY = e.clientY;
                const startedAt = Date.now();
                const rect = el.getBoundingClientRect();
                const offX = startX - rect.left, offY = startY - rect.top;
                let dragging = false, ghost = null, scrollTimer = null;

                const findEntry = (x, y) => {
                    const t = document.elementFromPoint(x, y);
                    const hit = t && t.closest ? t.closest('.' + ENTRY_CLS) : null;
                    return hit && grid.contains(hit) ? hit : null;
                };

                const startDrag = () => {
                    dragging = true;
                    el.classList.add('dragging');
                    ghost = el.cloneNode(true);
                    ghost.className = ENTRY_CLS + ' ic-ghost';
                    ghost.style.width = rect.width + 'px';
                    ghost.style.height = rect.height + 'px';
                    ghost.style.left = rect.left + 'px';
                    ghost.style.top = rect.top + 'px';
                    const gImg = ghost.querySelector('img');
                    if (gImg && gImg.dataset.src) { gImg.src = viewUrlById(id); gImg.removeAttribute('data-src'); }
                    document.body.appendChild(ghost);
                };

                const onMove = (ev) => {
                    const dist = Math.hypot(ev.clientX - startX, ev.clientY - startY);
                    if (!dragging) {
                        // 长按够久 或 位移够大 才进入拖拽，否则留给 click 语义
                        if (Date.now() - startedAt < DRAG_HOLD_MS && dist < DRAG_MOVE_PX) return;
                        startDrag();
                    }
                    ghost.style.left = (ev.clientX - offX) + 'px';
                    ghost.style.top = (ev.clientY - offY) + 'px';

                    // 靠近滚动区上下边缘时自动滚动
                    clearInterval(scrollTimer);
                    const sr = scroll.getBoundingClientRect();
                    const TH = 36, SP = 10;
                    if (ev.clientY < sr.top + TH) scrollTimer = setInterval(() => { scroll.scrollTop -= SP; }, 16);
                    else if (ev.clientY > sr.bottom - TH) scrollTimer = setInterval(() => { scroll.scrollTop += SP; }, 16);

                    const hit = findEntry(ev.clientX, ev.clientY);
                    if (!hit || hit === el) return;
                    const all = [...grid.querySelectorAll('.' + ENTRY_CLS)];
                    const from = all.indexOf(el), to = all.indexOf(hit);
                    if (from < to) grid.insertBefore(el, hit.nextSibling);
                    else grid.insertBefore(el, hit);
                };

                const onUp = (ev) => {
                    clearInterval(scrollTimer);
                    window.removeEventListener('pointermove', onMove);
                    window.removeEventListener('pointerup', onUp);
                    window.removeEventListener('pointercancel', onUp);
                    if (ghost) { ghost.remove(); ghost = null; }
                    el.classList.remove('dragging');
                    if (!dragging) return;          // 没进入拖拽 => 交给 click 处理
                    ev.stopPropagation();
                    commitDomOrder();
                };

                window.addEventListener('pointermove', onMove);
                window.addEventListener('pointerup', onUp);
                window.addEventListener('pointercancel', onUp);
            });

            el.addEventListener('click', (e) => {
                if (el.classList.contains('dragging')) return;
                e.preventDefault();
                e.stopPropagation();
                toggleSelect(id);
            });
        }

        function viewUrlById(id) {
            const e = pool.find((p) => p.id === id);
            return e ? viewUrl(e) : '';
        }

        // 拖动结束后：容器顺序以 DOM 为准，且输出顺序重新按视觉顺序推导
        function commitDomOrder() {
            const domIds = [...grid.querySelectorAll('.' + ENTRY_CLS)].map((el) => el.dataset.id);
            if (domIds.length !== pool.length) return;   // DOM 与状态不一致时不覆盖，避免丢数据
            const byId = new Map(pool.map((p) => [p.id, p]));
            pool = domIds.map((x) => byId.get(x)).filter(Boolean);
            const stillSelected = new Set(selectedIds);
            selectedIds = pool.filter((p) => stillSelected.has(p.id)).map((p) => p.id);
            writeState();
            render();
        }

        // ---------- 抑制 ComfyUI 原生拖放 ----------
        // 原生行为：单张图 -> 新建 LoadImage；多张图 -> 每张建一个 LoadImage 再加一个
        // BatchImages 节点（app.ts:797,810,2186-2202,2364-2384）。
        // 两条路都堵：litegraph 的 onDragDrop 返回 true 可完全接管（app.ts:766），
        // 而 app.ts:748 又会检查 event.defaultPrevented，所以 DOM 层 preventDefault 也生效。
        // Vue「Nodes 2.0」渲染器下 litegraph 钩子不覆盖，故 DOM 监听是必需的另一半。
        node.onDragOver = (e) => {
            if (!e?.dataTransfer?.types?.includes('Files')) return false;
            root.classList.add('drop-active');
            return true;
        };
        node.onDragDrop = async (e) => {
            root.classList.remove('drop-active');
            e.preventDefault();
            e.stopPropagation();
            const items = imageFilesFromDataTransfer(e.dataTransfer);
            if (items.length) await addItems(items);
            return true;
        };
        for (const type of ['dragenter', 'dragover']) {
            root.addEventListener(type, (e) => {
                e.preventDefault();
                e.stopPropagation();
                root.classList.add('drop-active');
            });
        }
        root.addEventListener('dragleave', (e) => {
            e.preventDefault();
            e.stopPropagation();
            if (!root.contains(e.relatedTarget)) root.classList.remove('drop-active');
        });
        root.addEventListener('drop', async (e) => {
            e.preventDefault();
            e.stopPropagation();
            root.classList.remove('drop-active');
            const items = imageFilesFromDataTransfer(e.dataTransfer);
            if (items.length) await addItems(items);
        });

        // ---------- 粘贴上传 ----------
        async function pasteFromClipboard(statusEl) {
            let items = [];
            try {
                if (navigator.clipboard?.read) {
                    for (const ci of await navigator.clipboard.read()) {
                        const t = ci.types.find((x) => x.startsWith('image/'));
                        if (t) items.push({ blob: await ci.getType(t), name: `paste-${Date.now()}.png` });
                    }
                }
            } catch (e) {
                console.warn('[ImageContainer] 剪贴板读取被拒绝，改用 Ctrl+V', e);
            }
            if (!items.length) {
                if (statusEl) statusEl.textContent = '剪贴板里没有图片，或浏览器拒绝了读取 —— 请选中本节点后按 Ctrl+V';
                return;
            }
            await addItems(items);
        }

        // Ctrl+V 兜底：navigator.clipboard.read() 需要安全上下文，
        // 局域网 IP + http 下会被浏览器拒绝，此时用 paste 事件仍能拿到图片。
        document.addEventListener('paste', async (e) => {
            const selected = app.canvas?.selected_nodes || {};
            const isTarget = node === app.canvas?.currentNode || Object.values(selected).includes(node);
            if (!isTarget) return;
            const items = [];
            for (const it of e.clipboardData?.items || []) {
                if (it.kind === 'file' && it.type.startsWith('image/')) {
                    const f = it.getAsFile();
                    if (f) items.push({ blob: f, name: f.name || `paste-${Date.now()}.png` });
                }
            }
            if (items.length) {
                e.preventDefault();
                await addItems(items);
            }
        });

        // ---------- 按住空格 / Alt 时让事件穿透，否则没法平移画布或拖动节点 ----------
        const setPassthrough = (on) => { root.style.pointerEvents = on ? 'none' : 'auto'; };
        window.addEventListener('keydown', (e) => { if (e.code === 'Space' || e.altKey) setPassthrough(true); });
        window.addEventListener('keyup', (e) => { if (e.code === 'Space' || !e.altKey) setPassthrough(false); });

        // hideOnZoom 默认 true 会导致画布缩小时整个容器变成占位矩形，这里关掉
        node.addDOMWidget('image_container_view', 'HTML', root, { serialize: false, hideOnZoom: false });

        readPoolState();
        render();

        const prevConfigure = node.onConfigure;
        node.onConfigure = function (info) {
            prevConfigure?.apply(this, arguments);
            readPoolState();
            render();
        };
    },
});
