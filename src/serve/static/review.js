"use strict";

const el = id => document.getElementById(id);
const classes = {D00: "Longitudinal crack", D10: "Transverse crack", D20: "Alligator crack", D40: "Pothole"};
let items = [];
let selected;
let run;
let exampleNames = [];

function rankedItems() {
  // Stable sort preserves input order for ties. Pending and failed images come last.
  return [...items].sort((a, b) => (b.result?.severity ?? -1) - (a.result?.severity ?? -1));
}

function part(tag, text, className) {
  const node = document.createElement(tag);
  node.textContent = text;
  if (className) node.className = className;
  return node;
}

function showPhoto() {
  const result = selected?.result;
  el("filename").textContent = selected?.name ?? "No photo selected";
  el("size").textContent = result ? result.image_w + " × " + result.image_h + " px" : "";
  el("score").textContent = result ? result.severity.toFixed(3) : "—";
  el("band").textContent = result?.band ?? "—";
  el("count").textContent = result?.detections.length ?? "—";
  el("detections").replaceChildren();
  el("boxes").replaceChildren();
  el("frame").hidden = !result;
  el("empty").hidden = Boolean(result);
  if (!result) {
    el("photo").removeAttribute("src");
    el("empty").textContent = selected?.error || (selected ? "Waiting for inference…" : "Upload JPG or PNG photos, or choose an example.");
    return;
  }
  el("photo").src = selected.url;
  el("photo").alt = "Road inspection of " + selected.name;
  el("frame").style.width = "min(100%, " + (55 * result.image_w / result.image_h) + "vh)";
  for (const d of result.detections) {
    const conf = Math.round(d.conf * 100) + "%";
    const box = part("div", "", "box");
    Object.assign(box.style, {
      left: 100 * d.x1 / result.image_w + "%", top: 100 * d.y1 / result.image_h + "%",
      width: 100 * (d.x2 - d.x1) / result.image_w + "%", height: 100 * (d.y2 - d.y1) / result.image_h + "%"
    });
    box.append(part("span", d.cls + " " + conf));
    el("boxes").append(box);
    el("detections").append(part("li", classes[d.cls] + " (" + d.cls + "), " + conf));
  }
  if (!result.detections.length) el("detections").append(part("li", "No detections at this threshold."));
}

function updateQueue() {
  const queue = el("queue");
  const ordered = rankedItems();
  el("total").textContent = items.length + (items.length === 1 ? " photo" : " photos");
  for (const [index, item] of ordered.entries()) {
    if (!item.row) {
      item.row = document.createElement("li");
      item.button = document.createElement("button");
      item.button.append(part("span", "", "rank"), part("span", item.name, "name"), part("span", "", "value"));
      item.row.append(item.button);
      item.button.onclick = () => { selected = item; updateQueue(); showPhoto(); };
    }
    item.button.firstChild.textContent = item.result ? String(index + 1).padStart(2, "0") : "·";
    item.button.lastChild.textContent = item.result ? item.result.severity.toFixed(3) : (item.error ? "Error" : "…");
    item.button.setAttribute("aria-pressed", String(item === selected));
    if (queue.children[index] !== item.row) queue.insertBefore(item.row, queue.children[index] ?? null);
  }
}

async function analyze() {
  run?.abort();
  const controller = new AbortController();
  run = controller;
  const signal = controller.signal;
  const confidence = Number(el("conf").value) / 100;
  const batch = items;
  for (const item of batch) { item.result = null; item.error = ""; }
  updateQueue();
  showPhoto();
  for (const [index, item] of batch.entries()) {
    el("status").textContent = "Reviewing photo " + (index + 1) + " of " + batch.length + "…";
    try {
      if (!item.file) {
        const source = await fetch(item.url, {signal});
        if (!source.ok) throw new Error("Example unavailable.");
        item.file = await source.blob();
      }
      const form = new FormData();
      form.append("file", item.file, item.name);
      const response = await fetch("/detect?conf=" + confidence, {method: "POST", body: form, signal});
      const data = await response.json();
      if (!response.ok) throw new Error(typeof data.detail === "string" ? data.detail : "Could not review this photo.");
      if (signal.aborted) return;
      item.result = data;
    } catch (error) {
      if (signal.aborted) return;
      item.error = error.message;
    }
    updateQueue();
    if (item === selected) showPhoto();
  }
  const errors = batch.filter(item => item.error).length;
  el("status").textContent = batch.length ? (batch.length - errors) + " reviewed" + (errors ? " · " + errors + " failed" : "") : "Ready for photos.";
}

function loadBatch(next) {
  run?.abort();
  for (const item of items) if (item.local) URL.revokeObjectURL(item.url);
  items = next;
  selected = items[0];
  el("queue").replaceChildren();
  analyze();
}

function loadExamples(names) {
  loadBatch(names.map(name => ({name, url: "/examples/" + encodeURIComponent(name)})));
}

el("files").addEventListener("change", event => {
  const files = Array.from(event.target.files);
  if (files.length) {
    el("examples").value = "";
    loadBatch(files.map(file => ({name: file.name, file, local: true, url: URL.createObjectURL(file)})));
  }
  event.target.value = "";
});
el("conf").addEventListener("input", () => { el("confValue").textContent = el("conf").value + "%"; });
el("conf").addEventListener("change", analyze);
el("examples").addEventListener("change", event => {
  if (event.target.value === "*") loadExamples(exampleNames);
  else if (event.target.value) loadExamples([event.target.value]);
});

async function init() {
  try {
    const response = await fetch("/examples");
    if (!response.ok) throw new Error();
    exampleNames = await response.json();
    if (exampleNames.length) {
      el("examples").add(new Option("Review all examples", "*"));
      for (const name of exampleNames) el("examples").add(new Option(name, name));
      if (!items.length) {
        el("examples").value = "*";
        loadExamples(exampleNames);
      }
    } else {
      el("examples").disabled = true;
      if (!items.length) el("status").textContent = "Ready for uploads.";
    }
  } catch {
    if (!items.length) el("status").textContent = "Examples unavailable. Upload a photo to begin.";
  }
}
init();
