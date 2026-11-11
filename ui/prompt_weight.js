// Frontend for MOSS-Image Prompt Weight (Presets).
// The editable list lives in the hidden `presets_data` widget so it is saved
// with the workflow and sent to Python unchanged.
// Visual language follows psl_comfyui_utilities/ui/any_switch.js: dark pill
// rows, circular enable mark, blue accent, and a panel height that fits content.

import { app } from '../../scripts/app.js';

const NODE_NAME = "MOSSImagePromptWeightPreset";
const MIN_WEIGHT = -2;
const MAX_WEIGHT = 2;
const WEIGHT_STEP = 0.05;
const ROW_GAP = 3;
const ROW_H = 27;
const ADD_H = 24;
const WIDGET_MIN_H = 40;
const WIDGET_SLACK = 24;
let styleReady = false;

function injectStyle() {
  if (styleReady || document.getElementById("moss-pw-style")) return;
  styleReady = true;
  const style = document.createElement("style");
  style.id = "moss-pw-style";
  style.textContent = `
    .moss-pw-root {
      display: flex; flex-direction: column; gap: 4px; padding: 4px 6px;
      box-sizing: border-box; width: 100%; height: 100%; position: relative;
    }
    .moss-pw-list {
      display: flex; flex-direction: column; gap: ${ROW_GAP}px;
      overflow-y: auto; flex: 1 1 auto; min-height: 0;
    }
    .moss-pw-row {
      display: flex; align-items: center; gap: 6px;
      background: #191919; border: 1px solid #333; border-radius: 4px;
      padding: 3px 7px; font-size: 12px; color: #ccc;
      user-select: none; min-height: 22px; box-sizing: border-box;
      flex: 0 0 auto;
    }
    .moss-pw-row:hover { border-color: #557; }
    .moss-pw-row.is-off { opacity: .45; }
    .moss-pw-switch {
      flex: 0 0 28px; width: 28px; height: 16px; border-radius: 8px;
      background: #3d3d3d; border: 1px solid #555; box-sizing: border-box;
      position: relative; cursor: pointer; outline: none;
      transition: background .15s ease, border-color .15s ease;
    }
    .moss-pw-switch::after {
      content: ""; position: absolute; top: 1px; left: 1px;
      width: 12px; height: 12px; border-radius: 50%; background: #999;
      transition: transform .15s ease, background .15s ease;
    }
    .moss-pw-switch.is-on { background: #0095ff; border-color: #0095ff; }
    .moss-pw-switch.is-on::after { transform: translateX(12px); background: #fff; }
    .moss-pw-switch:focus-visible { box-shadow: 0 0 0 2px rgba(0,149,255,.35); }
    .moss-pw-handle {
      flex: 0 0 14px; width: 14px; height: 20px; padding: 0; border: none;
      background: transparent; color: #999; display: block; line-height: 0;
      cursor: grab; opacity: .55; touch-action: none;
    }
    .moss-pw-handle svg {
      display: block; width: 14px; height: 20px;
      fill: currentColor; pointer-events: none;
    }
    .moss-pw-handle:hover { opacity: .9; }
    .moss-pw-handle:active { cursor: grabbing; }
    .moss-pw-handle:focus-visible { outline: 1px solid #0095ff; outline-offset: 2px; }
    .moss-pw-placeholder {
      flex: 0 0 auto; border: 1px dashed #0095ff; border-radius: 4px;
      background: rgba(0,149,255,.08); box-sizing: border-box;
    }
    .moss-pw-ghost {
      position: absolute; pointer-events: none; z-index: 9999; opacity: .85;
      box-shadow: 0 4px 14px rgba(0,0,0,.45); border-color: #0095ff;
    }
    .moss-pw-text {
      flex: 1 1 auto; min-width: 0; background: transparent; color: #ccc;
      border: 1px solid transparent; border-radius: 3px; padding: 1px 4px;
      font-size: 12px; font-family: inherit; line-height: 1.2; outline: none;
      user-select: text;
    }
    .moss-pw-text:hover { border-color: #555; }
    .moss-pw-text:focus { background: #101010; color: #fff; border-color: #0095ff; }
    .moss-pw-text::placeholder { color: #666; }
    .moss-pw-range {
      flex: 0 0 64px; width: 64px; height: 14px; margin: 0; padding: 0;
      background: transparent; -webkit-appearance: none; appearance: none;
      cursor: pointer;
    }
    .moss-pw-range::-webkit-slider-runnable-track {
      height: 4px; background: #333; border-radius: 2px;
    }
    .moss-pw-range::-webkit-slider-thumb {
      -webkit-appearance: none; width: 12px; height: 12px; margin-top: -4px;
      border-radius: 50%; background: #0095ff; border: none;
    }
    .moss-pw-range::-moz-range-track { height: 4px; background: #333; border-radius: 2px; }
    .moss-pw-range::-moz-range-thumb {
      width: 12px; height: 12px; border-radius: 50%; background: #0095ff; border: none;
    }
    .moss-pw-value {
      flex: 0 0 40px; text-align: right; color: #777; font-size: 11px;
      font-variant-numeric: tabular-nums; cursor: text;
    }
    .moss-pw-value:hover { color: #ccc; }
    .moss-pw-value-edit {
      flex: 0 0 40px; width: 40px; box-sizing: border-box;
      background: #101010; color: #fff; border: 1px solid #0095ff;
      border-radius: 3px; padding: 0 2px; font-size: 11px; font-family: inherit;
      text-align: right; outline: none; user-select: text;
    }
    .moss-pw-del {
      flex: 0 0 18px; width: 18px; height: 18px; padding: 0; line-height: 1;
      font-size: 14px; color: #777; background: transparent; border: none;
      cursor: pointer; opacity: .35;
    }
    .moss-pw-row:hover .moss-pw-del, .moss-pw-del:focus { opacity: 1; }
    .moss-pw-del:hover { color: #fff; }
    .moss-pw-add {
      flex: 0 0 auto; width: 100%; height: 24px; padding: 3px 0;
      background: #3d3d3d; color: #bbb; border: 1px solid #555; border-radius: 4px;
      font-size: 12px; cursor: pointer;
    }
    .moss-pw-add:hover { filter: brightness(1.15); color: #fff; border-color: #3f79ad; }
    .moss-pw-empty { color: #666; font-size: 11px; padding: 4px 2px; flex: 0 0 auto; }
    .moss-pw-list::-webkit-scrollbar { width: 6px; }
    .moss-pw-list::-webkit-scrollbar-thumb { background: rgba(127,127,127,.4); border-radius: 3px; }
  `;
  document.head.appendChild(style);
}

function parseItems(raw) {
  if (typeof raw !== "string" || !raw.trim()) return [];
  let data;
  try {
    data = JSON.parse(raw);
  } catch (error) {
    console.warn("[psl] invalid presets_data JSON; preset list ignored", error);
    return [];
  }
  if (!Array.isArray(data)) return [];
  const items = [];
  for (const item of data) {
    if (typeof item === "string") {
      items.push({ enabled: true, text: item, weight: 1 });
      continue;
    }
    if (!item || typeof item !== "object") continue;
    let weight = Number(item.weight);
    if (!Number.isFinite(weight)) weight = 1;
    items.push({
      enabled: item.enabled !== false,
      text: typeof item.text === "string" ? item.text : "",
      weight: Math.max(MIN_WEIGHT, Math.min(MAX_WEIGHT, weight)),
    });
  }
  return items;
}

function stopEvents(element) {
  for (const name of ["keydown", "keyup", "keypress", "input", "pointerdown",
                      "mousedown", "mouseup", "click", "dblclick", "wheel"]) {
    element.addEventListener(name, (event) => event.stopPropagation());
  }
}

function setup(node) {
  if (node._mossPresetUI) return;
  const hidden = node.widgets?.find((widget) => widget.name === "presets_data");
  if (!hidden) return;

  hidden.hidden = true;
  hidden.computeSize = () => [0, -4];
  const inputIndex = node.inputs?.findIndex((input) => input.name === "presets_data");
  if (inputIndex != null && inputIndex >= 0) node.removeInput(inputIndex);

  injectStyle();

  let items = parseItems(hidden.value);
  let lastFitH = 0;
  let fitScheduled = false;
  let fitGrow = false;
  let heightExtra = 0;

  const root = document.createElement("div");
  root.className = "moss-pw-root";
  root.style.setProperty("--comfy-widget-min-height", `${WIDGET_MIN_H}px`);
  const list = document.createElement("div");
  list.className = "moss-pw-list";
  const add = document.createElement("button");
  add.className = "moss-pw-add";
  add.type = "button";
  add.textContent = "+ Add preset entry";
  root.append(list, add);

  root.addEventListener("wheel", (event) => event.stopPropagation(), { passive: true });
  root.addEventListener("pointerdown", (event) => event.stopPropagation());

  function contentHeight() {
    const rootStyle = getComputedStyle(root);
    const listStyle = getComputedStyle(list);
    const padTop = parseFloat(rootStyle.paddingTop) || 4;
    const padBottom = parseFloat(rootStyle.paddingBottom) || 4;
    const rootGap = parseFloat(rootStyle.rowGap) || 4;
    const listGap = parseFloat(listStyle.rowGap) || ROW_GAP;
    let measured = 0;
    let children = 0;
    for (const child of list.children) {
      measured += child.offsetHeight;
      children++;
    }
    if (children > 1) measured += (children - 1) * listGap;
    // Before the DOM is laid out offsetHeight is 0; the estimate keeps the
    // first fit from locking the node to a clipped height.
    const estimated = items.length
      ? items.length * ROW_H + Math.max(0, items.length - 1) * ROW_GAP
      : measured;
    const addHeight = add.offsetHeight || ADD_H;
    const total = padTop + padBottom + Math.max(measured, estimated) + rootGap + addHeight;
    return Math.max(WIDGET_MIN_H, Math.round(total) + WIDGET_SLACK + heightExtra);
  }

  function fitNodeHeight() {
    const height = contentHeight();
    if (height === lastFitH) return;
    lastFitH = height;
    root.style.setProperty("--comfy-widget-min-height", `${height}px`);
    if (!node.graph) return;
    const width = node.size?.[0];
    if (!Number.isFinite(width) || width <= 0) return;
    try {
      const total = node.computeSize?.([width, 0]);
      const nodeHeight = Array.isArray(total) ? total[1] : null;
      if (Number.isFinite(nodeHeight) && nodeHeight > 0) node.size = [width, nodeHeight];
      node.graph.setDirtyCanvas(true, true);
    } catch (error) {
      console.warn("[psl] preset panel height fit failed; the list still scrolls", error);
    }
  }

  function correctOverflow() {
    if (!node.graph) return;
    const overflow = list.scrollHeight - list.clientHeight;
    if (overflow <= 0 || heightExtra >= 64) return;
    heightExtra = Math.min(64, heightExtra + overflow + 1);
    lastFitH = 0;
    fitNodeHeight();
    requestAnimationFrame(correctOverflow);
  }

  function scheduleFit(grow = false) {
    fitGrow = fitGrow || grow;
    if (fitScheduled) return;
    fitScheduled = true;
    requestAnimationFrame(() => {
      fitScheduled = false;
      if (!node.graph) return;
      fitNodeHeight();
      if (!fitGrow) return;
      fitGrow = false;
      requestAnimationFrame(correctOverflow);
    });
  }

  function commit() {
    hidden.value = JSON.stringify(items);
    node.setDirtyCanvas?.(true, true);
  }

  function makeRow(item, index) {
    const row = document.createElement("div");
    row.className = "moss-pw-row";
    row.dataset.index = String(index);
    row.classList.toggle("is-off", item.enabled === false);

    const handle = document.createElement("button");
    handle.type = "button";
    handle.className = "moss-pw-handle";
    handle.title = "Drag to reorder";
    handle.setAttribute("aria-label", "Drag to reorder");
    const icon = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    icon.setAttribute("viewBox", "0 0 14 20");
    icon.setAttribute("aria-hidden", "true");
    for (const [cx, cy] of [[4, 4], [10, 4], [4, 10], [10, 10], [4, 16], [10, 16]]) {
      const dot = document.createElementNS("http://www.w3.org/2000/svg", "circle");
      dot.setAttribute("cx", String(cx));
      dot.setAttribute("cy", String(cy));
      dot.setAttribute("r", "1.6");
      icon.appendChild(dot);
    }
    handle.append(icon);
    handle.addEventListener("pointerdown", startDrag);

    const toggle = document.createElement("span");
    toggle.className = "moss-pw-switch";
    toggle.classList.toggle("is-on", item.enabled !== false);
    toggle.setAttribute("role", "checkbox");
    toggle.setAttribute("aria-checked", String(item.enabled !== false));
    toggle.tabIndex = 0;
    toggle.title = "Enable or disable this entry";

    function flip() {
      item.enabled = !item.enabled;
      row.classList.toggle("is-off", !item.enabled);
      toggle.classList.toggle("is-on", item.enabled);
      toggle.setAttribute("aria-checked", String(item.enabled));
      commit();
    }

    toggle.addEventListener("click", (event) => {
      event.stopPropagation();
      flip();
    });
    toggle.addEventListener("keydown", (event) => {
      if (event.key === " " || event.key === "Enter") {
        event.preventDefault();
        event.stopPropagation();
        flip();
      }
    });

    const text = document.createElement("input");
    text.type = "text";
    text.className = "moss-pw-text";
    text.value = item.text;
    text.placeholder = "preset text";
    text.title = item.text;
    text.spellcheck = false;
    stopEvents(text);
    text.addEventListener("input", () => {
      item.text = text.value;
      text.title = text.value;
      commit();
    });

    const weight = document.createElement("input");
    weight.type = "range";
    weight.className = "moss-pw-range";
    weight.min = String(MIN_WEIGHT);
    weight.max = String(MAX_WEIGHT);
    weight.step = String(WEIGHT_STEP);
    weight.value = String(item.weight);
    weight.title = "1 = neutral, >1 emphasize, <1 weaken, 0 remove, <0 subtract the concept; scroll to adjust";
    stopEvents(weight);
    const value = document.createElement("span");
    value.className = "moss-pw-value";
    value.textContent = item.weight.toFixed(2);
    value.title = "Double-click to type a value";
    weight.addEventListener("input", () => {
      item.weight = Number(weight.value);
      value.textContent = item.weight.toFixed(2);
      commit();
    });

    weight.addEventListener("wheel", (event) => {
      if (event.deltaY === 0) return;
      event.preventDefault();
      event.stopPropagation();
      const direction = event.deltaY < 0 ? 1 : -1;
      const next = Math.max(MIN_WEIGHT, Math.min(MAX_WEIGHT,
        Math.round((item.weight + direction * WEIGHT_STEP) * 100) / 100));
      if (next === item.weight) return;
      item.weight = next;
      weight.value = String(next);
      value.textContent = next.toFixed(2);
      commit();
    }, { passive: false });

    let editingWeight = false;
    let weightEditor = null;

    function finishWeightEdit(save) {
      if (!editingWeight || !weightEditor) return;
      editingWeight = false;
      const editor = weightEditor;
      weightEditor = null;
      const raw = editor.value.trim();
      editor.remove();
      value.style.display = "";
      if (save && raw !== "") {
        const parsed = Number(raw);
        if (Number.isFinite(parsed)) {
          item.weight = Math.max(MIN_WEIGHT, Math.min(MAX_WEIGHT, Math.round(parsed * 100) / 100));
          weight.value = String(item.weight);
          value.textContent = item.weight.toFixed(2);
          commit();
          return;
        }
      }
      value.textContent = item.weight.toFixed(2);
    }

    value.addEventListener("dblclick", (event) => {
      event.preventDefault();
      event.stopPropagation();
      if (editingWeight) return;
      editingWeight = true;
      value.style.display = "none";
      const editor = document.createElement("input");
      editor.type = "text";
      editor.inputMode = "decimal";
      editor.className = "moss-pw-value-edit";
      editor.value = item.weight.toFixed(2);
      editor.title = "Enter to apply, Esc to cancel";
      editor.setAttribute("aria-label", "Weight");
      stopEvents(editor);
      editor.addEventListener("keydown", (event) => {
        if (event.key === "Enter") {
          event.preventDefault();
          event.stopPropagation();
          finishWeightEdit(true);
        } else if (event.key === "Escape") {
          event.preventDefault();
          event.stopPropagation();
          finishWeightEdit(false);
        }
      });
      editor.addEventListener("blur", () => finishWeightEdit(true));
      weightEditor = editor;
      value.after(editor);
      editor.focus();
      editor.select();
    });

    const remove = document.createElement("button");
    remove.className = "moss-pw-del";
    remove.type = "button";
    remove.textContent = "×";
    remove.title = "Remove this entry";
    stopEvents(remove);
    remove.addEventListener("click", (event) => {
      event.stopPropagation();
      items.splice(index, 1);
      commit();
      render();
    });

    function startDrag(event) {
      if (event.button !== 0 || event.isPrimary === false) return;
      event.preventDefault();
      event.stopPropagation();
      const pointerId = event.pointerId;
      const startY = event.clientY;
      const rowRect = row.getBoundingClientRect();
      const rootRect = root.getBoundingClientRect();
      const scale = root.offsetWidth ? rootRect.width / root.offsetWidth : 1;
      const ghostWidth = row.offsetWidth || rowRect.width;
      const ghostLeft = scale ? (rowRect.left - rootRect.left) / scale : 0;
      const ghostTop = scale ? (rowRect.top - rootRect.top) / scale : 0;
      let active = false;
      let ghost = null;
      let placeholder = null;
      let currentTargetRow;
      let placeholderTargetIndex = index;
      let cleaned = false;

      function visibleRows() {
        return [...list.querySelectorAll(".moss-pw-row")]
          .filter((candidate) => candidate !== row && candidate.style.display !== "none");
      }

      function nextVisibleRow() {
        let sibling = placeholder ? placeholder.nextElementSibling : null;
        while (sibling && (sibling === row || sibling.style.display === "none")) {
          sibling = sibling.nextElementSibling;
        }
        return sibling;
      }

      function movePlaceholder(targetRow) {
        if (targetRow === currentTargetRow) return;
        const before = new Map();
        for (const child of list.children) {
          if (child === row || child.style.display === "none") continue;
          before.set(child, child.getBoundingClientRect().top);
        }
        if (targetRow) list.insertBefore(placeholder, targetRow);
        else list.appendChild(placeholder);
        currentTargetRow = targetRow;
        const flipScale = scale || 1;
        for (const child of list.children) {
          if (child === row || child.style.display === "none") continue;
          const oldTop = before.get(child);
          if (oldTop == null) continue;
          const delta = (oldTop - child.getBoundingClientRect().top) / flipScale;
          if (Math.abs(delta) > 0.5 && typeof child.animate === "function") {
            child.animate(
              [{ transform: `translateY(${delta}px)` }, { transform: "translateY(0)" }],
              { duration: 140, easing: "ease" }
            );
          }
        }
      }

      function updatePlaceholder(clientY) {
        const rows = visibleRows();
        let targetRow = null;
        for (const candidate of rows) {
          const rect = candidate.getBoundingClientRect();
          if (clientY < rect.top + rect.height / 2) {
            targetRow = candidate;
            break;
          }
        }
        placeholderTargetIndex = targetRow ? Number(targetRow.dataset.index) : items.length;
        movePlaceholder(targetRow);
      }

      function cleanup() {
        if (cleaned) return;
        cleaned = true;
        handle.removeEventListener("pointermove", onMove);
        handle.removeEventListener("pointerup", onUp);
        handle.removeEventListener("pointercancel", onCancel);
        try {
          if (handle.hasPointerCapture?.(pointerId)) handle.releasePointerCapture(pointerId);
        } catch (error) { /* the pointer may already be gone */ }
        if (ghost) ghost.remove();
        ghost = null;
        if (placeholder) placeholder.remove();
        placeholder = null;
        row.style.display = "";
      }

      function onMove(moveEvent) {
        if (moveEvent.pointerId !== pointerId) return;
        moveEvent.preventDefault();
        moveEvent.stopPropagation();
        if (!active) {
          if (Math.abs(moveEvent.clientY - startY) < 4) return;
          active = true;
          ghost = row.cloneNode(true);
          const sources = row.querySelectorAll("input");
          ghost.querySelectorAll("input").forEach((input, i) => {
            const source = sources[i];
            if (!source) return;
            if (source.type === "checkbox") input.checked = source.checked;
            else input.value = source.value;
          });
          ghost.classList.add("moss-pw-ghost");
          ghost.style.width = `${ghostWidth}px`;
          ghost.style.left = `${ghostLeft}px`;
          ghost.style.top = `${ghostTop}px`;
          root.appendChild(ghost);

          placeholder = document.createElement("div");
          placeholder.className = "moss-pw-placeholder";
          placeholder.setAttribute("aria-hidden", "true");
          placeholder.style.height = `${row.offsetHeight}px`;
          row.style.display = "none";
          row.after(placeholder);
          currentTargetRow = nextVisibleRow();
          placeholderTargetIndex = currentTargetRow ? Number(currentTargetRow.dataset.index) : items.length;
        }
        ghost.style.top = `${ghostTop + (moveEvent.clientY - startY) / (scale || 1)}px`;
        const listRect = list.getBoundingClientRect();
        if (moveEvent.clientY < listRect.top + 20) list.scrollTop -= 8;
        else if (moveEvent.clientY > listRect.bottom - 20) list.scrollTop += 8;
        updatePlaceholder(moveEvent.clientY);
      }

      function finish(reorder) {
        const wasActive = active;
        cleanup();
        if (!reorder || !wasActive) return;
        let insertAt = placeholderTargetIndex;
        if (insertAt > index) insertAt -= 1;
        if (insertAt === index) return;
        const [moved] = items.splice(index, 1);
        items.splice(insertAt, 0, moved);
        commit();
        render();
      }

      function onUp(upEvent) {
        if (upEvent.pointerId !== pointerId) return;
        upEvent.preventDefault();
        upEvent.stopPropagation();
        finish(true);
      }

      function onCancel(cancelEvent) {
        if (cancelEvent.pointerId !== pointerId) return;
        cancelEvent.preventDefault();
        cancelEvent.stopPropagation();
        finish(false);
      }

      try {
        handle.setPointerCapture?.(pointerId);
      } catch (error) { /* synthetic pointers have no active capture */ }
      handle.addEventListener("pointermove", onMove);
      handle.addEventListener("pointerup", onUp);
      handle.addEventListener("pointercancel", onCancel);
    }

    row.append(handle, toggle, text, weight, value, remove);
    return row;
  }

  function render() {
    list.replaceChildren();
    if (!items.length) {
      const empty = document.createElement("div");
      empty.className = "moss-pw-empty";
      empty.textContent = "No preset entries. Enabled entries are merged into the prompt in list order.";
      list.append(empty);
    } else {
      items.forEach((item, index) => list.append(makeRow(item, index)));
    }
    heightExtra = 0;
    fitNodeHeight();
    scheduleFit(true);
  }

  stopEvents(add);
  add.addEventListener("click", () => {
    items.push({ enabled: true, text: "", weight: 1 });
    commit();
    render();
    const inputs = list.querySelectorAll(".moss-pw-text");
    inputs[inputs.length - 1]?.focus();
    list.scrollTop = list.scrollHeight;
  });

  node.addDOMWidget("prompt_weight_presets", "HTML", root, {
    serialize: false,
    hideOnZoom: false,
    getMinHeight: contentHeight,
  });

  hidden.serializeValue = () => JSON.stringify(items);

  function restore(info) {
    let raw = hidden.value;
    if (raw == null || raw === "") {
      for (const candidate of info?.widgets_values || []) {
        if (typeof candidate !== "string") continue;
        const parsed = parseItems(candidate);
        if (parsed.length) {
          raw = candidate;
          break;
        }
      }
    }
    items = parseItems(raw);
    render();
    commit();
    scheduleFit();
  }

  node._mossPresetUI = { restore };
  render();
  commit();
  if (node.size) node.setSize?.([Math.max(node.size[0], 380), node.size[1]]);

  if (typeof ResizeObserver === "function") {
    const observer = new ResizeObserver(() => scheduleFit());
    observer.observe(root);
    const previousRemoved = node.onRemoved;
    node.onRemoved = function () {
      observer.disconnect();
      document.querySelectorAll(".moss-pw-ghost").forEach((ghost) => ghost.remove());
      return previousRemoved?.apply(this, arguments);
    };
  }
  scheduleFit();
}

app.registerExtension({
  name: "PSL.PromptWeightPresets",
  beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData?.name !== NODE_NAME) return;

    const onCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const result = onCreated?.apply(this, arguments);
      setup(this);
      return result;
    };

    const onConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function () {
      const result = onConfigure?.apply(this, arguments);
      if (!this._mossPresetUI) setup(this);
      this._mossPresetUI?.restore(arguments[0]);
      if (this.size) this.setSize?.([Math.max(this.size[0], 380), this.size[1]]);
      return result;
    };
  },
});
