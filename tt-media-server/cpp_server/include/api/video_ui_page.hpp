// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

namespace tt::api {

/**
 * Single-page console for the video service, served at `/` and `/ui` when
 * MODEL_SERVICE=video. Plain HTML/CSS/JS, no external resources, so it works on
 * a machine without internet access. It calls the same-origin API from the
 * browser (the API key is typed into the page and kept in sessionStorage, so it is gone when the tab is closed).
 */
inline constexpr const char* kVideoUiHtml = R"VUI(<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Video service console</title>
<style>
:root{
  --bg:#f5f6f8;--panel:#ffffff;--text:#1b2330;--muted:#667085;--line:#e1e5ec;
  --accent:#2457d6;--accent-text:#ffffff;--code:#f0f2f6;
  --ok:#0f7a56;--okbg:#e0f5ec;--warn:#9a5b05;--warnbg:#fcefd6;
  --err:#b42318;--errbg:#fde7e4;--info:#1d4ed8;--infobg:#e3ecff;--idle:#475467;--idlebg:#eaecf0;
}
@media (prefers-color-scheme:dark){
  :root:not([data-theme="light"]){
    --bg:#0e131a;--panel:#161d27;--text:#e5e9f0;--muted:#97a1b2;--line:#283140;
    --accent:#5b8def;--accent-text:#0b1220;--code:#0f151d;
    --ok:#4cd6a0;--okbg:#10352a;--warn:#f2b35a;--warnbg:#3a2a0e;
    --err:#ff8a7d;--errbg:#3d1714;--info:#8db1ff;--infobg:#16294d;--idle:#b4bccb;--idlebg:#252e3b;
  }
}
:root[data-theme="dark"]{
  --bg:#0e131a;--panel:#161d27;--text:#e5e9f0;--muted:#97a1b2;--line:#283140;
  --accent:#5b8def;--accent-text:#0b1220;--code:#0f151d;
  --ok:#4cd6a0;--okbg:#10352a;--warn:#f2b35a;--warnbg:#3a2a0e;
  --err:#ff8a7d;--errbg:#3d1714;--info:#8db1ff;--infobg:#16294d;--idle:#b4bccb;--idlebg:#252e3b;
}
*{box-sizing:border-box}
html,body{margin:0}
body{background:var(--bg);color:var(--text);font:14px/1.45 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
button,input,select,textarea{font:inherit;color:inherit}
header{position:sticky;top:0;z-index:5;background:var(--panel);border-bottom:1px solid var(--line);
  display:flex;flex-wrap:wrap;align-items:center;gap:10px 16px;padding:10px 16px}
header h1{font-size:16px;margin:0;font-weight:650}
.spacer{flex:1}
.pill{display:inline-flex;align-items:center;gap:6px;padding:3px 10px;border-radius:999px;font-size:12px;font-weight:600;background:var(--idlebg);color:var(--idle)}
.pill::before{content:"";width:8px;height:8px;border-radius:50%;background:currentColor}
.pill.ok{background:var(--okbg);color:var(--ok)}
.pill.warn{background:var(--warnbg);color:var(--warn)}
.pill.err{background:var(--errbg);color:var(--err)}
.keybox{display:flex;align-items:center;gap:6px}
.keybox input{width:190px}
main{display:grid;grid-template-columns:minmax(300px,390px) minmax(0,1fr);gap:16px;padding:16px;max-width:1500px;margin:0 auto}
@media (max-width:960px){main{grid-template-columns:minmax(0,1fr)}}
.col{display:flex;flex-direction:column;gap:16px;min-width:0}
section.panel{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:14px;min-width:0}
section.panel>h2{font-size:13px;letter-spacing:.04em;text-transform:uppercase;color:var(--muted);margin:0 0 10px;display:flex;align-items:center;gap:8px}
section.panel>h2 .spacer{flex:1}
label{display:block;font-size:12px;color:var(--muted);margin:8px 0 3px}
input[type=text],input[type=password],input[type=number],select,textarea{width:100%;padding:7px 9px;border:1px solid var(--line);border-radius:7px;background:var(--bg)}
textarea{resize:vertical;min-height:64px}
input:focus,select:focus,textarea:focus,button:focus-visible{outline:2px solid var(--accent);outline-offset:1px}
.row{display:grid;grid-template-columns:1fr 1fr;gap:10px}
details{margin-top:8px}
summary{cursor:pointer;color:var(--muted);font-size:12px}
.btn{border:1px solid var(--line);background:var(--panel);border-radius:7px;padding:6px 11px;cursor:pointer;font-weight:550}
.btn:hover{border-color:var(--accent)}
.btn.primary{background:var(--accent);border-color:var(--accent);color:var(--accent-text)}
.btn.danger{color:var(--err)}
.btn.small{padding:2px 8px;font-size:12px}
.btn:disabled{opacity:.5;cursor:default}
.actions{display:flex;flex-wrap:wrap;gap:8px;margin-top:12px}
.seg{display:inline-flex}
.seg .btn:first-child{border-radius:7px 0 0 7px}
.seg .btn:last-child{border-radius:0 7px 7px 0;border-left:0;padding:2px 7px;color:var(--muted)}
.seg .btn:only-child{border-radius:7px}
.hint{font-size:12px;color:var(--muted);margin:4px 0 0}
.group{margin-top:10px}
.group h3{font-size:12px;color:var(--muted);margin:0 0 6px;font-weight:600}
.quick{display:flex;flex-wrap:wrap;gap:6px}
.badge{display:inline-block;padding:1px 8px;border-radius:999px;font-size:12px;font-weight:650;background:var(--idlebg);color:var(--idle)}
.badge.queued{background:var(--idlebg);color:var(--idle)}
.badge.in_progress,.badge.cancelling{background:var(--infobg);color:var(--info)}
.badge.cancelling{background:var(--warnbg);color:var(--warn)}
.badge.completed,.badge.s2{background:var(--okbg);color:var(--ok)}
.badge.failed,.badge.s5,.badge.err{background:var(--errbg);color:var(--err)}
.badge.cancelled{background:var(--idlebg);color:var(--muted)}
.badge.s4{background:var(--warnbg);color:var(--warn)}
.tablewrap{overflow-x:auto}
table{border-collapse:collapse;width:100%}
th,td{text-align:left;padding:7px 8px;border-bottom:1px solid var(--line);vertical-align:middle;white-space:nowrap}
th{font-size:12px;color:var(--muted);font-weight:600}
td.prompt{white-space:normal;min-width:160px;max-width:300px}
tr.sel td{background:var(--infobg)}
tr.clickable{cursor:pointer}
.mono{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:12px}
pre{margin:8px 0 0;padding:10px;background:var(--code);border-radius:8px;overflow:auto;max-height:340px;font-size:12px;line-height:1.45}
video{width:100%;max-height:420px;margin-top:8px;border-radius:8px;background:#000}
.empty{color:var(--muted);padding:14px 4px}
.steps{list-style:none;padding:0;margin:10px 0 0}
.steps li{padding:3px 0;display:flex;gap:8px;align-items:baseline}
.steps .mark{width:16px;text-align:center}
.hist{display:flex;flex-direction:column}
.hist .item{display:flex;align-items:center;gap:8px;padding:6px 4px;border-bottom:1px solid var(--line);cursor:pointer}
.hist .item:hover{background:var(--code)}
.hist .item.sel{background:var(--infobg)}
.hist .path{flex:1;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.hist .time{color:var(--muted);font-size:12px}
.toast{position:fixed;bottom:18px;left:50%;transform:translateX(-50%);background:var(--text);color:var(--bg);padding:8px 14px;border-radius:8px;font-size:13px;opacity:0;pointer-events:none;transition:opacity .15s}
.toast.show{opacity:1}
.chip{font-size:12px;color:var(--muted)}
</style>
</head>
<body>
<header>
  <h1>Video service console</h1>
  <span id="state" class="pill">connecting</span>
  <span id="stateInfo" class="chip"></span>
  <span class="spacer"></span>
  <div class="keybox">
    <label for="key" style="margin:0">API key</label>
    <input id="key" type="password" placeholder="Bearer token" autocomplete="off">
    <label style="margin:0;display:flex;gap:4px;align-items:center"><input id="mask" type="checkbox"> hide key in curl</label>
  </div>
  <button id="theme" class="btn small" title="Toggle light/dark">Theme</button>
</header>

<main>
  <div class="col">
    <section class="panel">
      <h2>New video request</h2>
      <label for="prompt">Prompt</label>
      <textarea id="prompt">A red fox running through fresh snow, cinematic, 4k</textarea>
      <div class="row">
        <div><label for="steps">Inference steps (4 to 50)</label><input id="steps" type="number" value="4" min="4" max="50"></div>
        <div><label for="mode">Mode</label>
          <select id="mode">
            <option value="">default (server setting)</option>
            <option value="true">async=true (job)</option>
            <option value="false">async=false (waits, returns mp4)</option>
          </select></div>
      </div>
      <details>
        <summary>More parameters</summary>
        <label for="neg">Negative prompt</label><input id="neg" type="text">
        <div class="row">
          <div><label for="seed">Seed</label><input id="seed" type="number"></div>
          <div><label for="fps">FPS</label><input id="fps" type="number"></div>
          <div><label for="width">Width</label><input id="width" type="number"></div>
          <div><label for="height">Height</label><input id="height" type="number"></div>
          <div><label for="frames">Num frames</label><input id="frames" type="number"></div>
          <div><label for="duration">Duration (s)</label><input id="duration" type="number"></div>
        </div>
        <label for="aspect">Aspect ratio</label><input id="aspect" type="text" placeholder="e.g. 16:9">
      </details>
      <div class="actions">
        <span class="seg"><button class="btn primary" id="send">Send request</button><button class="btn" id="sendCurl" title="Copy curl">curl</button></span>
        <button class="btn" id="flow" title="Create, wait, download, Range, delete, check 404">Run full flow</button>
      </div>
      <p class="hint" id="flowHint">The full flow sends a 4-step request, waits for it, downloads it, tests Range, deletes it and checks the 404.</p>
      <ul class="steps" id="flowSteps"></ul>
    </section>

    <section class="panel">
      <h2>Quick requests</h2>
      <p class="hint">Selected job: <span id="selLabel" class="mono">none</span>. Click a row in the jobs table to select one. Each button runs the request; the small button next to it copies the curl.</p>
      <div id="quick"></div>
    </section>
  </div>

  <div class="col">
    <section class="panel">
      <h2>Jobs on the server <span class="spacer"></span>
        <label style="margin:0;display:flex;gap:4px;align-items:center;font-weight:400;text-transform:none;letter-spacing:0"><input id="auto" type="checkbox" checked> auto refresh</label>
        <button class="btn small" id="refresh">Refresh</button></h2>
      <div class="tablewrap" id="jobs"><div class="empty">No data yet.</div></div>
    </section>

    <section class="panel" id="viewer">
      <h2>Response</h2>
      <div id="viewerBody" class="empty">Send a request to see the response here.</div>
    </section>

    <section class="panel">
      <h2>Requests sent from this page <span class="spacer"></span><button class="btn small danger" id="clearHist">Clear</button></h2>
      <div id="hist" class="hist"></div>
    </section>
  </div>
</main>
<div id="toast" class="toast"></div>

<script>
(() => {
'use strict';
const $ = (s, r = document) => r.querySelector(s);
const el = (tag, props = {}, ...kids) => {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(props)) {
    if (k === 'class') e.className = v;
    else if (k === 'onclick') e.onclick = v;
    else if (k === 'text') e.textContent = v;
    else if (v !== false && v != null) e.setAttribute(k, v === true ? '' : v);
  }
  for (const c of kids.flat()) if (c != null) e.append(c);
  return e;
};
const ls = {
  get(k, d) { try { const v = sessionStorage.getItem(k); return v == null ? d : JSON.parse(v); } catch (_) { return d; } },
  set(k, v) { try { sessionStorage.setItem(k, JSON.stringify(v)); } catch (_) {} },
};
const ORIGIN = location.origin;
let apiKey = ls.get('vui.key', '');
let selectedId = ls.get('vui.sel', '');
let history = ls.get('vui.hist', []);
let jobs = [];
let shownId = null;
let seq = Date.now();
const blobs = new Map();

function toast(msg) {
  const t = $('#toast'); t.textContent = msg; t.classList.add('show');
  clearTimeout(toast.h); toast.h = setTimeout(() => t.classList.remove('show'), 1600);
}
const q = s => "'" + String(s).replace(/'/g, "'\\''") + "'";

function curlOf(r) {
  const p = ['curl', '-sS'];
  if (r.dump) p.push('-D', '-');
  if (r.method !== 'GET') p.push('-X', r.method);
  p.push(q(ORIGIN + r.path));
  if (!r.noAuth && apiKey) p.push('-H', q('Authorization: Bearer ' + ($('#mask').checked ? '<token>' : apiKey)));
  const h = Object.assign({}, r.headers || {});
  if (r.body != null && !h['Content-Type']) h['Content-Type'] = 'application/json';
  for (const [k, v] of Object.entries(h)) p.push('-H', q(k + ': ' + v));
  if (r.body != null) p.push('-d', q(r.body));
  if (r.out) p.push('-o', q(r.out));
  return p.join(' ');
}
async function copy(text) {
  try { await navigator.clipboard.writeText(text); toast('Copied'); }
  catch (_) {
    const ta = el('textarea'); ta.value = text; document.body.append(ta); ta.select();
    try { document.execCommand('copy'); toast('Copied'); } catch (__) { toast('Copy failed'); }
    ta.remove();
  }
}

/* ---------- sending requests ---------- */
async function send(r) {
  const t0 = performance.now();
  const h = Object.assign({}, r.headers || {});
  if (!r.noAuth && apiKey) h['Authorization'] = 'Bearer ' + apiKey;
  if (r.body != null && !h['Content-Type']) h['Content-Type'] = 'application/json';
  const e = { id: ++seq, t: Date.now(), spec: r, status: 0, ms: 0, type: '', size: 0, text: '', error: '', hdrs: [] };
  try {
    const res = await fetch(ORIGIN + r.path, { method: r.method, headers: h, body: r.body == null ? undefined : r.body });
    e.status = res.status;
    e.type = res.headers.get('content-type') || '';
    e.hdrs = [...res.headers].filter(([k]) => /^(content-range|accept-ranges|retry-after|x-generation-time|content-length|content-type|www-authenticate)$/i.test(k));
    if (/^video\//i.test(e.type) || /octet-stream/i.test(e.type)) {
      const b = await res.blob();
      e.size = b.size;
      blobs.set(e.id, URL.createObjectURL(b));
      e.text = '[binary ' + e.type + ', ' + b.size + ' bytes]';
    } else {
      e.text = await res.text();
      e.size = e.text.length;
    }
  } catch (err) { e.error = String(err); }
  e.ms = Math.round(performance.now() - t0);
  addHistory(e);
  show(e);
  refreshJobs(true);
  return e;
}
function parseJson(s) { try { return JSON.parse(s); } catch (_) { return undefined; } }
const jsonBody = o => JSON.stringify(o);
async function quiet(path) {
  const h = {}; if (apiKey) h['Authorization'] = 'Bearer ' + apiKey;
  const res = await fetch(ORIGIN + path, { headers: h });
  return { status: res.status, json: parseJson(await res.text()) };
}

/* ---------- history ---------- */
function addHistory(e) {
  history.unshift(e);
  while (history.length > 100) { const o = history.pop(); if (blobs.has(o.id)) { URL.revokeObjectURL(blobs.get(o.id)); blobs.delete(o.id); } }
  persistHistory(); renderHistory();
}
function persistHistory() {
  ls.set('vui.hist', history.map(e => ({ ...e, text: e.text.length > 3000 ? e.text.slice(0, 3000) + '\n…(truncated)' : e.text })));
}
function statusBadge(e) {
  if (e.error) return el('span', { class: 'badge err', text: 'error' });
  return el('span', { class: 'badge s' + String(e.status)[0], text: String(e.status) });
}
function renderHistory() {
  const box = $('#hist'); box.replaceChildren();
  if (!history.length) { box.append(el('div', { class: 'empty', text: 'Nothing sent yet.' })); return; }
  for (const e of history) {
    const item = el('div', { class: 'item' + (e.id === shownId ? ' sel' : ''), onclick: () => show(e) },
      el('span', { class: 'time', text: new Date(e.t).toLocaleTimeString() }),
      statusBadge(e),
      el('span', { class: 'mono', text: e.spec.method }),
      el('span', { class: 'path mono', text: e.spec.path, title: e.spec.path }),
      el('span', { class: 'time', text: e.ms + ' ms' }),
      el('span', { class: 'seg' },
        el('button', { class: 'btn small', text: 'Run again', onclick: ev => { ev.stopPropagation(); send(e.spec); } }),
        el('button', { class: 'btn small', text: 'curl', onclick: ev => { ev.stopPropagation(); copy(curlOf(e.spec)); } })));
    box.append(item);
  }
}
$('#clearHist').onclick = () => {
  for (const u of blobs.values()) URL.revokeObjectURL(u);
  blobs.clear(); history = []; persistHistory(); renderHistory();
};

/* ---------- response viewer ---------- */
function show(e) {
  shownId = e.id;
  const body = $('#viewerBody'); body.className = ''; body.replaceChildren();
  const head = el('div', { style: 'display:flex;flex-wrap:wrap;gap:8px;align-items:center' },
    statusBadge(e),
    el('span', { class: 'mono', text: e.spec.method + ' ' + e.spec.path }),
    el('span', { class: 'chip', text: e.ms + ' ms' }),
    el('span', { class: 'spacer' }),
    el('span', { class: 'seg' },
      el('button', { class: 'btn small', text: 'Run again', onclick: () => send(e.spec) }),
      el('button', { class: 'btn small', text: 'curl', onclick: () => copy(curlOf(e.spec)) })));
  body.append(head);
  if (e.error) body.append(el('pre', { text: 'Request failed: ' + e.error + '\nIs the server running?' }));
  if (e.hdrs && e.hdrs.length) body.append(el('pre', { text: e.hdrs.map(([k, v]) => k + ': ' + v).join('\n') }));
  if (blobs.has(e.id) && /^video\//i.test(e.type)) {
    body.append(el('video', { src: blobs.get(e.id), controls: true }));
    body.append(el('p', { class: 'hint', text: e.text }));
  } else if (!e.error) {
    const j = parseJson(e.text);
    body.append(el('pre', { text: j !== undefined ? JSON.stringify(j, null, 2) : (e.text || '(empty body)') }));
    if (/^\[binary/.test(e.text)) body.append(el('p', { class: 'hint', text: 'The file is no longer in memory (page was reloaded). Run the request again to play it.' }));
  }
  renderHistory();
}

/* ---------- jobs table ---------- */
const ACTIVE = new Set(['queued', 'in_progress', 'cancelling']);
function jobPath(id, tail = '') { return '/v1/videos/generations/' + id + tail; }
const reqs = {
  get: id => ({ method: 'GET', path: jobPath(id) }),
  download: id => ({ method: 'GET', path: jobPath(id, '/download') }),
  range: id => ({ method: 'GET', path: jobPath(id, '/download'), headers: { Range: 'bytes=0-1023' }, out: 'part.mp4', dump: true }),
  cancel: id => ({ method: 'POST', path: jobPath(id, '/cancel') }),
  del: id => ({ method: 'DELETE', path: jobPath(id) }),
};
function pair(label, spec, cls = '') {
  return el('span', { class: 'seg' },
    el('button', { class: 'btn small ' + cls, text: label, onclick: ev => { ev.stopPropagation(); send(spec); } }),
    el('button', { class: 'btn small', text: '⧉', title: 'Copy curl', onclick: ev => { ev.stopPropagation(); copy(curlOf(spec)); } }));
}
function fmtTime(s) { return s ? new Date(s * 1000).toLocaleTimeString() : ''; }
function renderJobs() {
  const box = $('#jobs'); box.replaceChildren();
  if (!jobs.length) { box.append(el('div', { class: 'empty', text: 'No jobs. Send a request to create one.' })); return; }
  const rows = [...jobs].sort((a, b) => (b.created_at || 0) - (a.created_at || 0));
  const tb = el('tbody');
  for (const j of rows) {
    const p = j.request_parameters || {};
    const dur = j.completed_at ? (j.completed_at - j.created_at) + ' s' : '';
    const info = j.error ? (j.error.code + ': ' + j.error.message) : dur;
    const tr = el('tr', { class: 'clickable' + (j.id === selectedId ? ' sel' : ''), onclick: () => select(j.id) },
      el('td', {}, el('span', { class: 'badge ' + j.status, text: j.status })),
      el('td', { class: 'mono', title: j.id, text: j.id.slice(0, 8) }),
      el('td', { text: fmtTime(j.created_at) }),
      el('td', { class: 'prompt', title: p.prompt || '', text: (p.prompt || '').slice(0, 90) }),
      el('td', { text: p.num_inference_steps != null ? String(p.num_inference_steps) : '' }),
      el('td', { class: 'prompt', text: info }),
      el('td', {},
        el('div', { style: 'display:flex;gap:6px' },
          pair('Status', reqs.get(j.id)), pair('Download', reqs.download(j.id)), pair('Range', reqs.range(j.id)),
          pair('Cancel', reqs.cancel(j.id)), pair('Delete', reqs.del(j.id), 'danger'),
          el('button', { class: 'btn small', text: 'Copy id', onclick: ev => { ev.stopPropagation(); copy(j.id); } }))));
    tb.append(tr);
  }
  box.append(el('table', {}, el('thead', {}, el('tr', {}, ['Status', 'Id', 'Created', 'Prompt', 'Steps', 'Result', 'Requests'].map(h => el('th', { text: h })))), tb));
}
function select(id) {
  selectedId = id; ls.set('vui.sel', id);
  $('#selLabel').textContent = id || 'none';
  renderJobs();
}
let jobsBusy = false;
async function refreshJobs(silent) {
  if (jobsBusy) return; jobsBusy = true;
  try {
    const r = await quiet('/v1/videos/jobs');
    if (r.status === 200 && Array.isArray(r.json)) { jobs = r.json; renderJobs(); }
    else if (r.status === 401) { $('#jobs').replaceChildren(el('div', { class: 'empty', text: 'The server answered 401. Enter the API key at the top of the page.' })); }
    else if (!silent) toast('Jobs: HTTP ' + r.status);
  } catch (_) {
    $('#jobs').replaceChildren(el('div', { class: 'empty', text: 'Cannot reach the server.' }));
  } finally { jobsBusy = false; }
}
$('#refresh').onclick = () => refreshJobs(false);

/* ---------- liveness ---------- */
async function liveness() {
  const pill = $('#state'), info = $('#stateInfo');
  try {
    const res = await fetch(ORIGIN + '/tt-liveness');
    const j = parseJson(await res.text()) || {};
    if (j.model_ready) { pill.className = 'pill ok'; pill.textContent = 'model ready'; }
    else { pill.className = 'pill warn'; pill.textContent = 'warming up / not ready'; }
    const parts = [];
    if (j.runner_in_use) parts.push('runner: ' + j.runner_in_use);
    if (j.queue_size != null) parts.push('queue ' + j.queue_size + '/' + j.max_queue_size);
    if (j.error) parts.push(j.error);
    info.textContent = parts.join('  |  ');
  } catch (_) { pill.className = 'pill err'; pill.textContent = 'server unreachable'; info.textContent = ''; }
}

/* ---------- new request form ---------- */
function formSpec() {
  const body = { prompt: $('#prompt').value };
  const num = (id, key, int) => { const v = $(id).value; if (v !== '') body[key] = int ? parseInt(v, 10) : parseFloat(v); };
  num('#steps', 'num_inference_steps', true); num('#seed', 'seed', true); num('#fps', 'fps', false);
  num('#width', 'width', true); num('#height', 'height', true); num('#frames', 'num_frames', true); num('#duration', 'duration', false);
  if ($('#neg').value) body.negative_prompt = $('#neg').value;
  if ($('#aspect').value) body.aspect_ratio = $('#aspect').value;
  const mode = $('#mode').value;
  return { method: 'POST', path: '/v1/videos/generations' + (mode ? '?async=' + mode : ''), body: jsonBody(body) };
}
$('#send').onclick = async () => {
  const e = await send(formSpec());
  const j = parseJson(e.text);
  if (e.status === 202 && j && j.id) select(j.id);
};
$('#sendCurl').onclick = () => copy(curlOf(formSpec()));
$('#theme').onclick = () => {
  const cur = document.documentElement.getAttribute('data-theme');
  const dark = cur ? cur === 'dark' : matchMedia('(prefers-color-scheme:dark)').matches;
  const next = dark ? 'light' : 'dark';
  document.documentElement.setAttribute('data-theme', next); ls.set('vui.theme', next);
};
{ const t = ls.get('vui.theme', ''); if (t) document.documentElement.setAttribute('data-theme', t); }

/* ---------- quick requests ---------- */
const sel = () => selectedId || '00000000-0000-4000-8000-000000000000';
const smallBody = steps => jsonBody({ prompt: 'test', num_inference_steps: steps });
const QUICK = [
  ['Server', [
    ['Health', () => ({ method: 'GET', path: '/health' })],
    ['Liveness', () => ({ method: 'GET', path: '/tt-liveness' })],
    ['OpenAPI spec', () => ({ method: 'GET', path: '/openapi.json' })],
  ]],
  ['Jobs', [
    ['List jobs', () => ({ method: 'GET', path: '/v1/videos/jobs' })],
    ['Get job', () => reqs.get(sel())],
    ['Download', () => reqs.download(sel())],
    ['Download Range 0-1023', () => reqs.range(sel())],
    ['Cancel', () => reqs.cancel(sel())],
    ['Delete', () => reqs.del(sel())],
  ]],
  ['Create variants', [
    ['Create, 4 steps', () => ({ method: 'POST', path: '/v1/videos/generations', body: smallBody(4) })],
    ['Create, async=true', () => ({ method: 'POST', path: '/v1/videos/generations?async=true', body: smallBody(4) })],
    ['Create, sync (async=false)', () => ({ method: 'POST', path: '/v1/videos/generations?async=false', body: smallBody(4) })],
  ]],
  ['Expected errors', [
    ['No API key (401)', () => ({ method: 'GET', path: '/v1/videos/jobs', noAuth: true })],
    ['Invalid JSON (422)', () => ({ method: 'POST', path: '/v1/videos/generations', body: 'not json' })],
    ['Missing prompt (422)', () => ({ method: 'POST', path: '/v1/videos/generations', body: jsonBody({ num_inference_steps: 4 }) })],
    ['Steps = 3 (422)', () => ({ method: 'POST', path: '/v1/videos/generations', body: smallBody(3) })],
    ['Steps = 51 (422)', () => ({ method: 'POST', path: '/v1/videos/generations', body: smallBody(51) })],
    ['async=banana (422)', () => ({ method: 'POST', path: '/v1/videos/generations?async=banana', body: smallBody(4) })],
    ['Unknown job (404)', () => reqs.get('00000000-0000-4000-8000-000000000000')],
    ['Id is not a UUID (404)', () => reqs.get('12345')],
    ['Range out of file (416)', () => ({ method: 'GET', path: jobPath(sel(), '/download'), headers: { Range: 'bytes=999999999-' } })],
  ]],
];
function renderQuick() {
  const box = $('#quick'); box.replaceChildren();
  for (const [name, items] of QUICK) {
    const g = el('div', { class: 'group' }, el('h3', { text: name }));
    const row = el('div', { class: 'quick' });
    for (const [label, mk] of items) {
      row.append(el('span', { class: 'seg' },
        el('button', { class: 'btn small', text: label, onclick: () => send(mk()) }),
        el('button', { class: 'btn small', text: '⧉', title: 'Copy curl', onclick: () => copy(curlOf(mk())) })));
    }
    g.append(row); box.append(g);
  }
}

/* ---------- full flow ---------- */
let flowRunning = false;
async function runFlow() {
  if (flowRunning) { flowRunning = false; return; }
  flowRunning = true;
  const btn = $('#flow'); btn.textContent = 'Stop flow';
  const list = $('#flowSteps'); list.replaceChildren();
  const step = text => { const m = el('span', { class: 'mark', text: '…' }); const t = el('span', { text }); list.append(el('li', {}, m, t)); return (ok, msg) => { m.textContent = ok ? '✓' : '✗'; m.style.color = ok ? 'var(--ok)' : 'var(--err)'; if (msg) t.textContent = text + ' - ' + msg; }; };
  const stopped = () => !flowRunning;
  try {
    let done = step('Server is ready (/tt-liveness)');
    const lv = await fetch(ORIGIN + '/tt-liveness').then(r => r.json()).catch(() => ({}));
    if (!lv.model_ready) { done(false, 'model is not ready yet'); return; } done(true);

    done = step('POST create job (4 steps)');
    const c = await send({ method: 'POST', path: '/v1/videos/generations?async=true', body: smallBody(4) });
    const job = parseJson(c.text);
    if (c.status !== 202 || !job || !job.id) { done(false, 'HTTP ' + c.status); return; }
    done(true, 'id ' + job.id.slice(0, 8)); select(job.id);

    done = step('Wait for completed');
    let status = job.status, t0 = Date.now();
    while (!stopped() && ACTIVE.has(status) && Date.now() - t0 < 15 * 60 * 1000) {
      await new Promise(r => setTimeout(r, 2000));
      const g = await quiet(jobPath(job.id));
      status = g.json ? g.json.status : 'unknown';
      done(true, status + ' (' + Math.round((Date.now() - t0) / 1000) + ' s)'); list.lastChild.firstChild.textContent = '…';
    }
    if (stopped()) { done(false, 'stopped'); return; }
    if (status !== 'completed') { done(false, 'ended as ' + status); return; }
    done(true, 'completed in ' + Math.round((Date.now() - t0) / 1000) + ' s');

    done = step('GET download (200, video/mp4)');
    const d = await send(reqs.download(job.id));
    done(d.status === 200 && /video\/mp4/.test(d.type) && d.size > 0, 'HTTP ' + d.status + ', ' + d.size + ' bytes');

    done = step('GET download with Range (206)');
    const r = await send({ method: 'GET', path: jobPath(job.id, '/download'), headers: { Range: 'bytes=0-1023' }, out: 'part.mp4', dump: true });
    done(r.status === 206, 'HTTP ' + r.status + ', ' + r.size + ' bytes');

    done = step('DELETE job (200)');
    const x = await send(reqs.del(job.id));
    done(x.status === 200, 'HTTP ' + x.status);

    done = step('GET deleted job (404)');
    const n = await send(reqs.get(job.id));
    done(n.status === 404, 'HTTP ' + n.status);
  } catch (err) {
    step('Flow error').call(null, false, String(err));
  } finally { flowRunning = false; btn.textContent = 'Run full flow'; }
}
$('#flow').onclick = runFlow;

/* ---------- init ---------- */
$('#key').value = apiKey;
$('#mask').checked = ls.get('vui.mask', false);
$('#mask').onchange = () => ls.set('vui.mask', $('#mask').checked);
$('#key').oninput = () => { apiKey = $('#key').value.trim(); ls.set('vui.key', apiKey); };
$('#key').onchange = () => refreshJobs(false);
$('#selLabel').textContent = selectedId || 'none';
renderQuick(); renderHistory(); liveness(); refreshJobs(true);
setInterval(liveness, 4000);
setInterval(() => { if (!document.hidden && $('#auto').checked) refreshJobs(true); }, 2000);
})();
</script>
</body>
</html>
)VUI";

}  // namespace tt::api
