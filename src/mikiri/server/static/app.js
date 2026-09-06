/* Mikiri front end: plain JavaScript, no external libraries (works offline). */
'use strict';

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => Array.from(document.querySelectorAll(sel));

const COLORS = {
  left: '#3fc380', right: '#ff7f2a', center: '#e6c85a', corrected: '#ff5cff',
  low: '#8a8f9c', accent: '#4f8cff', cursor: '#ffffff', event: '#ffd166', compare: '#c9d1ff',
  // The other player: one flat colour, so the subject's coloured limbs stay the thing the eye lands on.
  other: '#d7dbe3',
};
const LEFT = new Set([1, 3, 5, 7, 9, 11, 13, 15, 20, 22, 24]);
const RIGHT = new Set([2, 4, 6, 8, 10, 12, 14, 16, 21, 23, 25]);
const sideOf = (i) => (LEFT.has(i) ? 'left' : RIGHT.has(i) ? 'right' : 'center');
const DEFAULT_SERIES = ['knee_L', 'knee_R', 'hip_L', 'hip_R', 'trunk_lean', 'hip_y'];

const S = {
  pid: null, state: null, pose: null, frame: 0, playing: false, timer: null, speed: 1,
  showSkeleton: true, showLabels: false, edit: false,
  imgs: new Map(), hoverJoint: null, drag: null, lastJoint: 13,
  series: new Set(DEFAULT_SERIES), plots: [],
  compare: null, calibPick: null, courtPick: null, settings: null, suggestions: null,
  timelineDrag: null, selectFrame: null, selectPeople: [],
  sidePage: 'movement', hideStepFor: null,
};

// ---------------------------------------------------------------- utilities

async function api(method, url, body, isForm) {
  const opts = { method, headers: {} };
  if (body !== undefined) {
    if (isForm) opts.body = body;
    else { opts.headers['Content-Type'] = 'application/json'; opts.body = JSON.stringify(body); }
  }
  const r = await fetch(url, opts);
  if (!r.ok) {
    let detail = r.statusText;
    try { const j = await r.json(); detail = typeof j.detail === 'string' ? j.detail : JSON.stringify(j.detail); } catch (e) { /* ignore */ }
    throw new Error(detail);
  }
  const ct = r.headers.get('content-type') || '';
  return ct.includes('json') ? r.json() : r;
}

function setStatus(msg, isError) {
  const el = $('#status');
  el.textContent = msg;
  $('#statusbar').classList.toggle('error', !!isError);
  if (isError) console.error(msg);
}

function fmtTime(frame, fps) { return (frame / fps).toFixed(3) + ' s'; }
const clamp = (v, a, b) => Math.min(Math.max(v, a), b);
function fps() { return S.state ? S.state.video.fps : 30; }
function nFrames() { return S.state ? S.state.video.n_frames : 0; }

function showModal(id) {
  $('#modal-root').classList.remove('hidden');
  $$('.modal').forEach((m) => m.classList.add('hidden'));
  $(id).classList.remove('hidden');
}
function closeModals() { $('#modal-root').classList.add('hidden'); $$('.modal').forEach((m) => m.classList.add('hidden')); }

async function pollJob(jobId, onProgress) {
  for (;;) {
    const j = await api('GET', `/api/jobs/${jobId}`);
    if (onProgress) onProgress(j);
    if (j.state === 'done') return j;
    if (j.state === 'error') throw new Error(j.error || 'job failed');
    if (j.state === 'cancelled') return j;
    await new Promise((res) => setTimeout(res, 150));
  }
}

// ---------------------------------------------------------------- project state

async function setProject(state) {
  stopPlayback();
  S.pid = state.id;
  S.state = state;
  S.pose = null;
  S.imgs.clear();
  S.suggestions = null;
  S.lastOptionsSynced = null;
  S.lastSmoothingSynced = null;
  S.lastTrimSynced = null;
  S.frame = clamp(state.trim_start, 0, Math.max(0, state.video.n_frames - 1));
  $('#main').classList.remove('no-project');
  if (state.has_pose) await loadPose();
  buildPlots();
  renderAll();
  refreshTabs();
  setStatus(`Opened ${state.video.name}: ${state.video.width}x${state.video.height}, ${state.video.fps.toFixed(2)} fps, ${state.video.n_frames} frames`);
}

function applyState(state) {
  S.state = state;
  renderAll();
}

async function loadPose() {
  if (!S.pid) return;
  const pose = await api('GET', `/api/projects/${S.pid}/pose`);
  S.pose = pose.has_pose ? pose : null;
  if (S.pose) S.suggestions = S.pose.suggestions;
}

async function refresh(refetchPose) {
  const state = await api('GET', `/api/projects/${S.pid}`);
  if (refetchPose) await loadPose();
  applyState(state);
}

function poseIndex(frame) {
  if (!S.pose) return -1;
  const i = frame - S.pose.start;
  return i >= 0 && i < S.pose.keypoints.length ? i : -1;
}
function keypointsAt(frame) {
  const i = poseIndex(frame);
  return i < 0 ? null : S.pose.keypoints[i];
}

// ---------------------------------------------------------------- frames

function getImage(idx, pid) {
  const key = (pid || S.pid) + ':' + idx;
  let e = S.imgs.get(key);
  if (e) return e;
  e = { img: new Image(), ready: false, failed: false };
  e.img.onload = () => { e.ready = true; if (idx === S.frame || (S.compare && idx === compareFrame())) renderVideo(); };
  e.img.onerror = () => { e.failed = true; };
  e.img.src = `/api/projects/${pid || S.pid}/frame/${idx}`;
  S.imgs.set(key, e);
  if (S.imgs.size > 600) { const first = S.imgs.keys().next().value; S.imgs.delete(first); }
  return e;
}
function prefetch(center) {
  for (let i = center - 2; i <= center + 12; i++) if (i >= 0 && i < nFrames()) getImage(i);
}

function fitCanvas(canvas, wrap, vw, vh) {
  const W = wrap.clientWidth, H = wrap.clientHeight;
  if (!W || !H) return { w: 0, h: 0, scale: 1 };
  const s = Math.min(W / vw, H / vh);
  const w = Math.max(1, Math.floor(vw * s)), h = Math.max(1, Math.floor(vh * s));
  const dpr = window.devicePixelRatio || 1;
  if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
    canvas.width = Math.round(w * dpr); canvas.height = Math.round(h * dpr);
    canvas.style.width = w + 'px'; canvas.style.height = h + 'px';
  }
  return { w, h, scale: w / vw, dpr };
}

function drawSkeleton(ctx, kp, scale, opts) {
  const thr = opts.thr, corrected = opts.corrected || [], muted = !!opts.muted;
  const edges = S.state.meta.skeleton_edges;
  const lw = Math.max(2, scale * S.state.video.width / 360) * (muted ? 0.7 : 1);
  ctx.lineWidth = lw; ctx.lineCap = 'round';
  for (const [a, b] of edges) {
    if (a < 5 || b < 5) continue;
    if (kp[a][2] < thr || kp[b][2] < thr) continue;
    const sides = new Set([sideOf(a), sideOf(b)]); sides.delete('center');
    ctx.strokeStyle = muted ? COLORS.other : (sides.size === 1 ? COLORS[[...sides][0]] : COLORS.center);
    ctx.beginPath(); ctx.moveTo(kp[a][0] * scale, kp[a][1] * scale); ctx.lineTo(kp[b][0] * scale, kp[b][1] * scale); ctx.stroke();
  }
  const r = lw * 1.3;
  for (let i = 0; i < kp.length; i++) {
    if (i < 5 && i !== 0) continue;
    const [x, y, c] = kp[i];
    const cx = x * scale, cy = y * scale;
    if (corrected.includes(i)) {
      ctx.fillStyle = COLORS.corrected; ctx.beginPath(); ctx.arc(cx, cy, r, 0, Math.PI * 2); ctx.fill();
      ctx.strokeStyle = COLORS.corrected; ctx.lineWidth = 2; ctx.beginPath(); ctx.arc(cx, cy, r + 4, 0, Math.PI * 2); ctx.stroke();
    } else if (c >= thr) {
      ctx.fillStyle = muted ? COLORS.other : COLORS[sideOf(i)]; ctx.beginPath(); ctx.arc(cx, cy, r, 0, Math.PI * 2); ctx.fill();
    } else if (c > 0) {
      ctx.strokeStyle = COLORS.low; ctx.lineWidth = 1; ctx.beginPath(); ctx.arc(cx, cy, Math.max(2, r - 1), 0, Math.PI * 2); ctx.stroke();
    }
    if (opts.highlight === i) {
      ctx.strokeStyle = '#fff'; ctx.lineWidth = 2; ctx.beginPath(); ctx.arc(cx, cy, r + 7, 0, Math.PI * 2); ctx.stroke();
    }
    if (opts.labels && (c >= thr || corrected.includes(i))) {
      const name = S.state.meta.keypoint_names[i].replace('left_', 'L ').replace('right_', 'R ').replace('_', ' ');
      ctx.font = '11px sans-serif'; ctx.fillStyle = '#000'; ctx.fillText(name, cx + 7, cy - 5); ctx.fillStyle = '#fff'; ctx.fillText(name, cx + 6, cy - 6);
    }
  }
}

function renderVideo() {
  if (!S.state) return;
  const canvas = $('#video-canvas'), wrap = $('#video-wrap');
  const { w, h, scale, dpr } = fitCanvas(canvas, wrap, S.state.video.width, S.state.video.height);
  if (!w) return;
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.fillStyle = '#000'; ctx.fillRect(0, 0, w, h);
  const e = getImage(S.frame);
  if (e.ready) ctx.drawImage(e.img, 0, 0, w, h);
  prefetch(S.frame);
  const msg = $('#video-overlay-msg');
  if (e.failed) { msg.textContent = 'Frame not available'; msg.classList.remove('hidden'); } else msg.classList.add('hidden');
  // A frame that is not here yet is being decoded: on a long clip that can
  // be a second or two, and a black picture with no word is a broken app.
  clearTimeout(S.loadingTimer);
  if (!e.ready && !e.failed) {
    S.loadingTimer = setTimeout(() => {
      const cur = getImage(S.frame);
      if (!cur.ready && !cur.failed) { msg.textContent = `Loading frame ${S.frame}...`; msg.classList.remove('hidden'); }
    }, 250);
  }
  const kp = keypointsAt(S.frame);
  if (kp && S.showSkeleton) {
    let drawn = kp;
    if (S.drag) { drawn = kp.map((p) => p.slice()); drawn[S.drag.joint] = [S.drag.x, S.drag.y, 1]; }
    const corrected = (S.pose.corrected[poseIndex(S.frame)] || []).slice();
    if (S.drag) corrected.push(S.drag.joint);
    drawSkeleton(ctx, drawn, scale, { thr: S.state.conf_threshold, corrected, highlight: S.edit ? (S.drag ? S.drag.joint : S.hoverJoint) : null, labels: S.showLabels });
  }
  // The other player, when one was tracked: muted, and never editable.
  if (S.pose && S.showSkeleton && S.pose.others) {
    const i = poseIndex(S.frame);
    for (const o of S.pose.others) {
      const okp = i >= 0 ? o.keypoints[i] : null;
      if (okp) drawSkeleton(ctx, okp, scale, { thr: S.state.conf_threshold, muted: true });
    }
  }
  if (S.calibPick) {
    ctx.strokeStyle = '#00e5ff'; ctx.lineWidth = 2;
    for (const [x, y] of S.calibPick.points) { ctx.beginPath(); ctx.moveTo(x * scale - 8, y * scale); ctx.lineTo(x * scale + 8, y * scale); ctx.moveTo(x * scale, y * scale - 8); ctx.lineTo(x * scale, y * scale + 8); ctx.stroke(); }
    if (S.calibPick.points.length === 2) { const [a, b] = S.calibPick.points; ctx.beginPath(); ctx.moveTo(a[0] * scale, a[1] * scale); ctx.lineTo(b[0] * scale, b[1] * scale); ctx.stroke(); }
  }
  // racket-head estimate
  if (S.pose && S.pose.racket && $('#show-racket').checked) {
    const i = poseIndex(S.frame);
    const head = i >= 0 ? S.pose.racket.head[i] : null;
    const kpNow = keypointsAt(S.frame);
    const wi = S.state.racket_hand === 'left' ? 9 : 10;
    if (head && kpNow && kpNow[wi][2] >= S.state.conf_threshold) {
      ctx.strokeStyle = '#ffffff'; ctx.lineWidth = 2; ctx.beginPath(); ctx.moveTo(kpNow[wi][0] * scale, kpNow[wi][1] * scale); ctx.lineTo(head[0] * scale, head[1] * scale); ctx.stroke();
      ctx.beginPath(); ctx.arc(head[0] * scale, head[1] * scale, 5, 0, Math.PI * 2); ctx.stroke();
      const sp = S.pose.racket.speed_mps ? S.pose.racket.speed_mps[i] : null; const spb = S.pose.racket.speed_bh[i];
      const txt = sp != null ? `${sp.toFixed(1)} m/s` : (spb != null ? `${spb.toFixed(1)} bh/s` : '');
      if (txt) { ctx.font = '11px sans-serif'; ctx.fillStyle = '#000'; ctx.fillText(txt, head[0] * scale + 8, head[1] * scale + 4); ctx.fillStyle = '#fff'; ctx.fillText(txt, head[0] * scale + 7, head[1] * scale + 3); }
    }
  }
  // court calibration points (stored, and the ones being clicked)
  const showStored = $('#show-court').checked && S.state.court && S.state.court.image_points;
  const courtPts = S.courtPick ? S.courtPick.points : (showStored ? S.state.court.image_points : []);
  // How many of those are floor points; anything beyond is a net-post top.
  const nFloor = S.courtPick ? S.courtPick.courtNeeded : (courtPts ? courtPts.length : 0);
  const postPts = S.courtPick ? S.courtPick.points.slice(nFloor) : (showStored ? (S.state.court.post_tops_image || []) : []);
  if (courtPts && courtPts.length) {
    ctx.strokeStyle = S.courtPick ? '#00e5ff' : '#ffd166'; ctx.fillStyle = ctx.strokeStyle; ctx.lineWidth = 2; ctx.font = 'bold 12px sans-serif';
    courtPts.slice(0, nFloor).forEach(([x, y], i) => {
      const cx = x * scale, cy = y * scale;
      ctx.beginPath(); ctx.moveTo(cx - 9, cy); ctx.lineTo(cx + 9, cy); ctx.moveTo(cx, cy - 9); ctx.lineTo(cx, cy + 9); ctx.stroke();
      ctx.fillText(String(i + 1), cx + 6, cy - 6);
    });
    const floorPts = courtPts.slice(0, nFloor);
    if (floorPts.length >= 2) { ctx.setLineDash([4, 4]); ctx.beginPath(); floorPts.forEach(([x, y], i) => (i ? ctx.lineTo(x * scale, y * scale) : ctx.moveTo(x * scale, y * scale))); if (floorPts.length >= 4) ctx.closePath(); ctx.stroke(); ctx.setLineDash([]); }
  }
  if (postPts && postPts.length) {
    ctx.strokeStyle = '#b388ff'; ctx.fillStyle = ctx.strokeStyle; ctx.lineWidth = 2; ctx.font = 'bold 12px sans-serif';
    postPts.forEach(([x, y], i) => {
      const cx = x * scale, cy = y * scale;
      ctx.beginPath(); ctx.arc(cx, cy, 6, 0, Math.PI * 2); ctx.stroke();
      ctx.fillText(i === 0 ? 'L post' : 'R post', cx + 8, cy - 6);
    });
  }
  // event badge
  const ev = S.state.events.find((x) => x.frame === S.frame);
  if (ev) { ctx.font = 'bold 14px sans-serif'; ctx.fillStyle = 'rgba(0,0,0,0.6)'; ctx.fillRect(6, 6, ctx.measureText(ev.name).width + 14, 24); ctx.fillStyle = COLORS.event; ctx.fillText(ev.name, 13, 23); }
  // The rally, or the rest, the frame belongs to: the timeline's label for
  // it, captioned top right so scrubbing a match reads as a match.
  const caption = S.pose ? timelineCaptionAt(S.frame) : '';
  if (caption) {
    ctx.font = '12px sans-serif';
    const cw = ctx.measureText(caption).width + 14;
    ctx.fillStyle = 'rgba(0,0,0,0.55)'; ctx.fillRect(w - cw - 6, 6, cw, 22);
    ctx.fillStyle = 'rgba(255,255,255,0.9)'; ctx.fillText(caption, w - cw + 1, 21);
  }
  if (S.state.camera_side === 'unknown' || true) {
    ctx.font = '11px sans-serif'; ctx.fillStyle = 'rgba(255,255,255,0.65)'; ctx.fillText('2D image-plane angles (side-on tool)', 8, h - 8);
  }
  renderCompare();
  $('#frame-number').textContent = S.frame;
  $('#frame-total').textContent = Math.max(0, nFrames() - 1);
  $('#frame-time').textContent = fmtTime(S.frame, fps());
}

function compareFrame() { return S.compare ? S.frame + S.compare.offset : null; }

function renderCompare() {
  const wrap = $('#compare-wrap');
  if (!S.compare) { wrap.classList.add('hidden'); return; }
  wrap.classList.remove('hidden');
  const c = S.compare, canvas = $('#compare-canvas');
  const { w, h, scale, dpr } = fitCanvas(canvas, wrap, c.state.video.width, c.state.video.height);
  if (!w) return;
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.fillStyle = '#000'; ctx.fillRect(0, 0, w, h);
  const f = compareFrame();
  if (f >= 0 && f < c.state.video.n_frames) {
    const e = getImage(f, c.pid);
    if (e.ready) ctx.drawImage(e.img, 0, 0, w, h);
    if (c.pose && S.showSkeleton) {
      const i = f - c.pose.start;
      if (i >= 0 && i < c.pose.keypoints.length) drawSkeleton(ctx, c.pose.keypoints[i], scale, { thr: c.state.conf_threshold, corrected: c.pose.corrected[i] });
    }
  }
  $('#compare-label').textContent = `${c.state.name}  frame ${f}  (offset ${c.offset >= 0 ? '+' : ''}${c.offset}, aligned on ${c.alignEvent || 'start'})`;
}

// ---------------------------------------------------------------- timeline

function timelineGeom() {
  const canvas = $('#timeline-canvas');
  const w = canvas.clientWidth, h = canvas.clientHeight;
  const pad = 10;
  const n = Math.max(1, nFrames() - 1);
  return { w, h, pad, n, x: (f) => pad + (f / n) * (w - 2 * pad), f: (x) => Math.round(clamp((x - pad) / (w - 2 * pad), 0, 1) * n) };
}

function renderTimeline() {
  if (!S.state) return;
  const canvas = $('#timeline-canvas');
  const dpr = window.devicePixelRatio || 1;
  const g = timelineGeom();
  if (canvas.width !== Math.round(g.w * dpr) || canvas.height !== Math.round(g.h * dpr)) { canvas.width = Math.round(g.w * dpr); canvas.height = Math.round(g.h * dpr); }
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, g.w, g.h);
  const y0 = 36, y1 = g.h - 8;
  ctx.fillStyle = '#2a2e38'; ctx.fillRect(g.pad, y0, g.w - 2 * g.pad, y1 - y0);
  // trimmed range
  ctx.fillStyle = 'rgba(79,140,255,0.28)'; ctx.fillRect(g.x(S.state.trim_start), y0, g.x(S.state.trim_end) - g.x(S.state.trim_start), y1 - y0);
  // What each stretch of the clip is, along the top of the bar: play in
  // green, rest in grey, another scene in red. Read straight off the
  // segmenter's runs, so a twelve-minute match shows its points at a glance.
  const runs = (S.state.tracking && S.state.tracking.segments) || [];
  for (const r of runs) {
    ctx.fillStyle = TIMELINE_STATE[r.state] || TIMELINE_STATE.idle;
    ctx.fillRect(g.x(r.start), y0, Math.max(1, g.x(r.end + 1) - g.x(r.start)), 4);
  }
  // Rallies as bands, tinted by who won where the chain knows: near in the
  // left-hand colour, far in the right-hand one, open in grey. Without a
  // chain the bouts of play stand in, untinted.
  for (const band of timelineBands()) {
    ctx.fillStyle = band.fill;
    ctx.fillRect(g.x(band.start), y0 + 6, Math.max(1, g.x(band.end + 1) - g.x(band.start)), y1 - 6 - (y0 + 6));
  }
  // tracked range
  if (S.pose) { ctx.fillStyle = 'rgba(63,195,128,0.35)'; ctx.fillRect(g.x(S.pose.start), y1 - 5, g.x(S.pose.end) - g.x(S.pose.start), 5); }
  // lost frames
  if (S.pose && S.pose.meta) { ctx.fillStyle = 'rgba(255,92,92,0.7)'; for (const m of S.pose.meta) if (m.lost) ctx.fillRect(g.x(m.frame) - 1, y1 - 5, 2, 5); }
  // detected contacts, as short ticks so they never crowd the event labels
  if (S.pose && S.pose.strokes && S.pose.strokes.strokes.length) {
    ctx.fillStyle = 'rgba(179,136,255,0.9)';
    for (const st of S.pose.strokes.strokes) ctx.fillRect(g.x(st.frame + S.pose.start) - 1, y0 + 6, 2, 8);
  }
  // handles
  for (const f of [S.state.trim_start, S.state.trim_end]) { ctx.fillStyle = COLORS.accent; ctx.fillRect(g.x(f) - 3, y0 - 4, 6, y1 - y0 + 8); }
  // events (labels use up to three rows so close events stay readable)
  ctx.font = '11px sans-serif';
  const lastLabelEnd = [-Infinity, -Infinity, -Infinity];
  for (const ev of [...S.state.events].sort((a, b) => a.frame - b.frame)) {
    const x = g.x(ev.frame);
    ctx.strokeStyle = COLORS.event; ctx.lineWidth = 1.5; ctx.beginPath(); ctx.moveTo(x, y0 - 2); ctx.lineTo(x, y1); ctx.stroke();
    const w = ctx.measureText(ev.name).width;
    let row = lastLabelEnd.findIndex((end) => x + 3 >= end);
    if (row < 0) row = lastLabelEnd.indexOf(Math.min(...lastLabelEnd));
    lastLabelEnd[row] = x + 3 + w + 6;
    ctx.fillStyle = COLORS.event; ctx.fillText(ev.name, x + 3, 10 + row * 11);
  }
  // cursor
  const xc = g.x(S.frame);
  ctx.strokeStyle = COLORS.cursor; ctx.lineWidth = 2; ctx.beginPath(); ctx.moveTo(xc, y0 - 6); ctx.lineTo(xc, y1 + 4); ctx.stroke();
  ctx.fillStyle = '#ccc'; ctx.font = '10px sans-serif'; ctx.fillText('0', g.pad, g.h - 0); ctx.fillText(String(nFrames() - 1), g.w - g.pad - 24, g.h - 0);
}

const TIMELINE_STATE = {
  play: 'rgba(63,195,128,0.55)', idle: 'rgba(154,160,173,0.28)', interval: 'rgba(154,160,173,0.28)',
  warm_up: 'rgba(230,200,90,0.45)', not_badminton: 'rgba(255,92,92,0.45)',
};
const TIMELINE_WINNER = { near: 'rgba(63,195,128,0.22)', far: 'rgba(255,127,42,0.22)' };

// The bands the timeline draws and names: the chain's rallies with their
// outcome where there is a chain, else the bouts of play. Frames absolute.
function timelineBands() {
  if (!S.pose) return [];
  const off = S.pose.start;
  const m = S.state.match;
  if (m && m.rallies && m.rallies.length) {
    return m.rallies.map((r, i) => {
      const w = r.winner && r.winner.value;
      const label = `rally ${i + 1}, ${r.duration_s} s, ${r.n_contacts} contacts, ${w ? `${w} won (${r.winner.how})` : 'open'}`;
      return { start: r.start + off, end: r.end + off, fill: TIMELINE_WINNER[w] || 'rgba(154,160,173,0.18)', label };
    });
  }
  const b = S.state.bouts;
  if (b && b.bouts && b.bouts.length) {
    return b.bouts.map((x, i) => ({ start: x.start + off, end: x.end + off, fill: 'rgba(154,160,173,0.18)', label: `bout ${i + 1}, ${x.duration_s} s` }));
  }
  return [];
}

function timelineCaptionAt(f) {
  const band = timelineBands().find((b) => f >= b.start && f <= b.end);
  const run = ((S.state.tracking && S.state.tracking.segments) || []).find((r) => f >= r.start && f <= r.end);
  const parts = [];
  if (band) parts.push(band.label);
  if (run && run.state !== 'play') parts.push(run.state.replace('_', ' '));
  return parts.join('; ');
}

function timelineLabelAt(x) {
  const g = timelineGeom(); const f = g.f(x);
  return timelineCaptionAt(f);
}

function bindTimeline() {
  const canvas = $('#timeline-canvas');
  const pos = (ev) => { const r = canvas.getBoundingClientRect(); return ev.clientX - r.left; };
  // Hovering a band names it: the rally, its length, who won.
  canvas.addEventListener('mousemove', (ev) => { if (S.state && !S.timelineDrag) canvas.title = timelineLabelAt(pos(ev)); });
  canvas.addEventListener('mousedown', (ev) => {
    if (!S.state) return;
    const g = timelineGeom(); const x = pos(ev);
    const dIn = Math.abs(x - g.x(S.state.trim_start)), dOut = Math.abs(x - g.x(S.state.trim_end));
    if (dIn < 7 && dIn <= dOut) S.timelineDrag = 'in';
    else if (dOut < 7) S.timelineDrag = 'out';
    else { S.timelineDrag = 'seek'; seek(g.f(x)); }
  });
  window.addEventListener('mousemove', (ev) => {
    if (!S.timelineDrag || !S.state) return;
    const g = timelineGeom(); const f = g.f(pos(ev));
    if (S.timelineDrag === 'seek') seek(f);
    else if (S.timelineDrag === 'in') { S.state.trim_start = Math.min(f, S.state.trim_end); renderTimeline(); renderTrim(); }
    else { S.state.trim_end = Math.max(f, S.state.trim_start); renderTimeline(); renderTrim(); }
  });
  window.addEventListener('mouseup', async () => {
    if (S.timelineDrag === 'in' || S.timelineDrag === 'out') await setTrim(S.state.trim_start, S.state.trim_end);
    S.timelineDrag = null;
  });
}

// ---------------------------------------------------------------- plots

function seriesColor(name) {
  if (name.endsWith('_L') || name.includes('_L_')) return COLORS.left;
  if (name.endsWith('_R') || name.includes('_R_')) return COLORS.right;
  if (name === 'hip_y') return COLORS.accent;
  return COLORS.center;
}
function seriesLabel(name) {
  if (name === 'hip_y') return 'hip height (px, up = higher)';
  if (name === 'ankle_L_y') return 'ankle L height (px)';
  if (name === 'ankle_R_y') return 'ankle R height (px)';
  if (name === 'stance_width') return 'stance width (px)';
  if (name === 'racket_speed') return (S.pose && S.pose.racket && S.pose.racket.speed_mps) ? 'racket head speed (m/s, estimate)' : 'racket head speed (body heights/s, estimate)';
  if (name.endsWith('_seg')) return name.replace('_seg', ' segment vs ground (deg)');
  return name.replace('_', ' ') + ' (deg)';
}
function seriesValues(pose, state, name) {
  if (name === 'racket_speed') return pose.racket ? (pose.racket.speed_mps || pose.racket.speed_bh) : null;
  const v = pose.angles[name];
  if (!v) return null;
  if (name === 'hip_y' || name === 'ankle_L_y' || name === 'ankle_R_y') return v.map((x) => (x == null ? null : state.video.height - x));
  return v;
}

function buildPlots() {
  const holder = $('#plots'); holder.innerHTML = '';
  S.plots = [];
  const toggles = $('#angle-toggles'); toggles.innerHTML = '';
  if (!S.state) return;
  const names = [...S.state.meta.angle_names, ...S.state.meta.segment_names, ...S.state.meta.position_names, 'racket_speed'];
  for (const name of names) {
    const lab = document.createElement('label');
    const cb = document.createElement('input'); cb.type = 'checkbox'; cb.checked = S.series.has(name); cb.dataset.series = name;
    cb.addEventListener('change', () => { if (cb.checked) S.series.add(name); else S.series.delete(name); buildPlots(); renderPlots(); });
    const sw = document.createElement('span'); sw.className = 'sw'; sw.style.background = seriesColor(name);
    lab.appendChild(cb); lab.appendChild(sw); lab.appendChild(document.createTextNode(name.replace('_seg', ' seg')));
    toggles.appendChild(lab);
  }
  for (const name of names) {
    if (!S.series.has(name)) continue;
    const div = document.createElement('div'); div.className = 'plot'; div.dataset.series = name;
    const canvas = document.createElement('canvas'); div.appendChild(canvas); holder.appendChild(div);
    canvas.addEventListener('mousedown', (ev) => { const f = plotFrameAt(canvas, ev); if (f != null) seek(f); S.timelineDrag = 'seek-plot'; });
    canvas.addEventListener('mousemove', (ev) => { if (S.timelineDrag === 'seek-plot') { const f = plotFrameAt(canvas, ev); if (f != null) seek(f); } });
    S.plots.push({ name, canvas });
  }
}
const PLOT_PAD = { l: 44, r: 8, t: 14, b: 4 };
function plotFrameAt(canvas, ev) {
  if (!S.pose) return null;
  const r = canvas.getBoundingClientRect(); const x = ev.clientX - r.left;
  const w = canvas.clientWidth; const n = S.pose.keypoints.length;
  const t = clamp((x - PLOT_PAD.l) / (w - PLOT_PAD.l - PLOT_PAD.r), 0, 1);
  return S.pose.start + Math.round(t * (n - 1));
}

function renderPlots() {
  if (!S.state) return;
  for (const p of S.plots) {
    const canvas = p.canvas, dpr = window.devicePixelRatio || 1;
    const w = canvas.clientWidth, h = canvas.clientHeight;
    if (!w) continue;
    if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) { canvas.width = Math.round(w * dpr); canvas.height = Math.round(h * dpr); }
    const ctx = canvas.getContext('2d'); ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);
    ctx.font = '11px sans-serif'; ctx.fillStyle = '#c8ccd6'; ctx.fillText(seriesLabel(p.name), PLOT_PAD.l + 4, 10);
    if (!S.pose) { ctx.fillStyle = '#666'; ctx.fillText('run tracking to see angles', PLOT_PAD.l + 4, h / 2); continue; }
    const vals = seriesValues(S.pose, S.state, p.name);
    if (!vals) { ctx.fillStyle = '#666'; ctx.fillText('not available', PLOT_PAD.l + 4, h / 2); continue; }
    const n = vals.length; const x0 = PLOT_PAD.l, x1 = w - PLOT_PAD.r, y0 = PLOT_PAD.t, y1 = h - PLOT_PAD.b;
    const finite = vals.filter((v) => v != null);
    let cmpVals = null;
    if (S.compare && S.compare.pose) cmpVals = seriesValues(S.compare.pose, S.compare.state, p.name);
    const allFinite = cmpVals ? finite.concat(cmpVals.filter((v) => v != null)) : finite;
    let lo = allFinite.length ? Math.min(...allFinite) : 0, hi = allFinite.length ? Math.max(...allFinite) : 1;
    if (hi - lo < 5) { lo -= 2.5; hi += 2.5; }
    const padv = (hi - lo) * 0.08; lo -= padv; hi += padv;
    const X = (i) => x0 + (i / Math.max(1, n - 1)) * (x1 - x0);
    const Y = (v) => y1 - ((v - lo) / (hi - lo)) * (y1 - y0);
    // grid
    ctx.strokeStyle = '#2c313d'; ctx.lineWidth = 1; ctx.fillStyle = '#8a8f9c'; ctx.font = '10px sans-serif';
    for (let k = 0; k <= 2; k++) { const v = lo + ((hi - lo) * k) / 2; const y = Y(v); ctx.beginPath(); ctx.moveTo(x0, y); ctx.lineTo(x1, y); ctx.stroke(); ctx.fillText(v.toFixed(0), 4, y + 3); }
    // events
    for (const ev of S.state.events) {
      const i = ev.frame - S.pose.start; if (i < 0 || i >= n) continue;
      ctx.strokeStyle = 'rgba(255,209,102,0.7)'; ctx.beginPath(); ctx.moveTo(X(i), y0); ctx.lineTo(X(i), y1); ctx.stroke();
    }
    // compare series
    if (cmpVals) {
      ctx.strokeStyle = COLORS.compare; ctx.lineWidth = 1; ctx.setLineDash([4, 3]); ctx.beginPath(); let pen = false;
      for (let i = 0; i < n; i++) {
        const j = i + S.pose.start + S.compare.offset - S.compare.pose.start;
        const v = j >= 0 && j < cmpVals.length ? cmpVals[j] : null;
        if (v == null) { pen = false; continue; }
        if (!pen) { ctx.moveTo(X(i), Y(v)); pen = true; } else ctx.lineTo(X(i), Y(v));
      }
      ctx.stroke(); ctx.setLineDash([]);
    }
    // main series, dim where confidence is low
    const conf = S.pose.confidence[p.name];
    ctx.lineWidth = 1.6; let pen = false; ctx.strokeStyle = seriesColor(p.name); ctx.beginPath();
    for (let i = 0; i < n; i++) {
      const v = vals[i];
      if (v == null) { pen = false; continue; }
      if (!pen) { ctx.moveTo(X(i), Y(v)); pen = true; } else ctx.lineTo(X(i), Y(v));
    }
    ctx.stroke();
    if (conf) { ctx.fillStyle = 'rgba(255,92,92,0.9)'; for (let i = 0; i < n; i++) if (conf[i] < S.state.conf_threshold + 0.15 && vals[i] != null) ctx.fillRect(X(i) - 1, Y(vals[i]) - 1, 2, 2); }
    // corrected markers
    for (let i = 0; i < n; i++) if (S.pose.corrected[i] && S.pose.corrected[i].length && vals[i] != null) { ctx.fillStyle = COLORS.corrected; ctx.fillRect(X(i) - 1.5, y1 - 3, 3, 3); }
    // cursor
    const ci = S.frame - S.pose.start;
    if (ci >= 0 && ci < n) {
      ctx.strokeStyle = COLORS.cursor; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(X(ci), y0); ctx.lineTo(X(ci), y1); ctx.stroke();
      const v = vals[ci];
      if (v != null) { ctx.fillStyle = '#fff'; ctx.font = '11px sans-serif'; const txt = v.toFixed(1); const tx = Math.min(X(ci) + 4, x1 - 34); ctx.fillText(txt, tx, Math.max(y0 + 10, Math.min(Y(v) - 4, y1 - 2))); }
    }
  }
  renderCurrentValues();
}

function renderCurrentValues() {
  const el = $('#current-values'); el.innerHTML = '';
  if (!S.pose) return;
  const i = poseIndex(S.frame); if (i < 0) return;
  const names = [...S.state.meta.angle_names, ...S.state.meta.position_names];
  for (const name of names) {
    const v = S.pose.angles[name][i];
    const span = document.createElement('span');
    span.innerHTML = `<span class="lname">${name}</span> <b>${v == null ? '-' : v.toFixed(1)}</b>`;
    el.appendChild(span);
  }
}

// ---------------------------------------------------------------- panels

function renderInfo() {
  const v = S.state.video;
  $('#video-info').innerHTML = `<span><b>${v.name}</b></span><span>${v.width} x ${v.height}</span><span>${v.fps.toFixed(3)} fps</span><span>${v.n_frames} frames, ${v.duration_s.toFixed(2)} s</span><span>rotation ${v.rotation} deg</span>${S.state.project_path ? `<span>project: ${S.state.project_path}</span>` : ''}`;
}

function renderTrim() {
  // Never overwrite an input the user is editing, and only sync when the trim changed.
  const key = S.state.trim_start + ':' + S.state.trim_end;
  if (key !== S.lastTrimSynced) {
    for (const [id, v] of [['#trim-in', S.state.trim_start], ['#trim-out', S.state.trim_end]]) {
      if (document.activeElement !== $(id)) $(id).value = v;
    }
    S.lastTrimSynced = key;
  }
  $('#trim-in').max = nFrames() - 1; $('#trim-out').max = nFrames() - 1;
  const n = S.state.trim_end - S.state.trim_start + 1;
  $('#trim-summary').textContent = `${n} frames, ${(n / fps()).toFixed(2)} s selected`;
}

function renderEvents() {
  const presets = $('#event-presets'); presets.innerHTML = '';
  for (const name of S.state.meta.preset_events) {
    const b = document.createElement('button'); b.className = 'btn small'; b.textContent = name; b.id = 'event-preset-' + name.replace(/\s+/g, '-');
    b.addEventListener('click', () => addEvent(name, S.frame)); presets.appendChild(b);
  }
  const list = $('#events-list'); list.innerHTML = '';
  S.state.events.forEach((ev, idx) => {
    const li = document.createElement('li');
    li.innerHTML = `<span class="name">${ev.name}</span><span class="frame">frame ${ev.frame}, ${fmtTime(ev.frame, fps())}</span>` +
      (ev.source === 'auto' ? `<span class="auto">auto ${ev.confidence != null ? Math.round(ev.confidence * 100) + '%' : ''}</span>` : '<span></span>') +
      `<span class="spacer"></span><button class="btn go">go</button><button class="btn move">move here</button><button class="btn del" title="delete">x</button>`;
    li.querySelector('.go').addEventListener('click', () => seek(ev.frame));
    li.querySelector('.move').addEventListener('click', async () => { applyState(await api('PUT', `/api/projects/${S.pid}/events/${idx}`, { frame: S.frame })); });
    li.querySelector('.del').addEventListener('click', async () => { applyState(await api('DELETE', `/api/projects/${S.pid}/events/${idx}`)); });
    list.appendChild(li);
  });
  const m = S.state.metrics || {};
  const parts = [];
  if (m.flight_time_s != null) parts.push(`flight time <b>${m.flight_time_s.toFixed(3)} s</b>`);
  if (m.flight_height_cm != null) parts.push(`height from flight time <b>${m.flight_height_cm.toFixed(1)} cm</b>`);
  if (m.hip_rise_px != null) parts.push(`hip rise <b>${m.hip_rise_px.toFixed(0)} px${m.hip_rise_cm != null ? ' = ' + m.hip_rise_cm.toFixed(1) + ' cm' : ''}</b>`);
  if (m.load_depth_px != null) parts.push(`loading depth <b>${m.load_depth_px.toFixed(0)} px${m.load_depth_cm != null ? ' = ' + m.load_depth_cm.toFixed(1) + ' cm' : ''}</b>`);
  $('#metrics').innerHTML = parts.map((p) => `<span>${p}</span>`).join('');
  renderSuggestions();
  renderAssessment();
}

function renderAssessment() {
  const table = $('#assessment-table'); const cues = $('#assessment-cues');
  const a = S.state.assessment;
  const prof = (S.state.meta.movements || []).find((m) => m.id === S.state.movement);
  $('#assessment-sub').textContent = prof ? `against the ${prof.name} reference bands` : '';
  if (!a) {
    table.innerHTML = '';
    cues.textContent = S.state.has_pose ? '' : 'Run tracking, then mark the events (or accept the suggestions) to see the assessment.';
    return;
  }
  const rows = a.metrics.map((m) => {
    const value = m.value == null ? '' : `${m.value}${m.unit ? ' ' + m.unit : ''}`;
    const status = m.status === 'missing' ? `missing: ${m.missing.join(', ')}` : m.status;
    return `<tr><td>${m.label}<div class="muted">${m.explain}</div></td><td class="val">${value}</td><td class="val">${m.band_text}</td><td><span class="st ${m.status}">${status}</span></td><td class="cue">${m.cue || ''}</td></tr>`;
  });
  table.innerHTML = '<tr><th>Metric</th><th>Value</th><th>Band</th><th>Status</th><th>Cue</th></tr>' + rows.join('');
  const s = a.summary || {};
  let text = `${s.ideal || 0} in band, ${(s.low || 0) + (s.high || 0)} outside, ${s.missing || 0} waiting for an event.`;
  if (a.missing_events && a.missing_events.length) text += ` Mark: ${a.missing_events.join(', ')}.`;
  if (a.cues && a.cues.length) text += ' Cues: ' + a.cues.join(' ');
  text += ' Bands are coaching heuristics; edit them in the movements.json file in the Mikiri folder.';
  cues.textContent = text;
}

function renderSuggestions() {
  const box = $('#suggestions');
  if (!S.suggestions || !S.suggestions.length || !S.pose) { box.classList.add('hidden'); return; }
  box.classList.remove('hidden');
  box.innerHTML = '<div class="row"><b>Suggested events</b> <span class="muted small-text">(from smoothed ankle height; confirm or move)</span><span class="spacer"></span><button class="btn small" id="accept-all-btn">Accept all</button><button class="btn small" id="dismiss-sug-btn">Dismiss</button></div>';
  for (const s of S.suggestions) {
    const d = document.createElement('div'); d.className = 'sug';
    d.innerHTML = `<span class="name" style="min-width:100px">${s.name}</span><span>frame ${s.frame}</span><span class="conf">confidence ${Math.round(s.confidence * 100)}%</span><span class="muted small-text">${s.note || ''}</span><span class="spacer"></span><button class="btn small go">go</button><button class="btn small acc">Accept</button>`;
    d.querySelector('.go').addEventListener('click', () => seek(s.frame));
    d.querySelector('.acc').addEventListener('click', async () => { applyState(await api('POST', `/api/projects/${S.pid}/events/accept`, { suggestions: [s] })); });
    box.appendChild(d);
  }
  $('#accept-all-btn').addEventListener('click', async () => { applyState(await api('POST', `/api/projects/${S.pid}/events/accept`, { suggestions: S.suggestions })); S.suggestions = null; renderSuggestions(); });
  $('#dismiss-sug-btn').addEventListener('click', () => { S.suggestions = null; renderSuggestions(); });
}

function renderOptions() {
  // Form inputs are only synced when the corresponding state changed, so a
  // background refresh never clobbers something the user is typing.
  const optKey = JSON.stringify([S.state.swap_lr, S.state.camera_side, S.state.conf_threshold, S.state.notes, S.state.movement, S.state.racket_hand, S.state.lead_leg]);
  if (optKey !== S.lastOptionsSynced) {
    $('#swap-lr').checked = S.state.swap_lr;
    $('#camera-side').value = S.state.camera_side;
    const mv = $('#movement');
    if (!mv.options.length) for (const m of S.state.meta.movements || []) { const o = document.createElement('option'); o.value = m.id; o.textContent = m.name; mv.appendChild(o); }
    mv.value = S.state.movement;
    const prof = (S.state.meta.movements || []).find((m) => m.id === S.state.movement);
    $('#movement-desc').textContent = prof ? prof.description : '';
    $('#racket-hand').value = S.state.racket_hand || 'right';
    $('#lead-leg').value = S.state.lead_leg || 'auto';
    $('#conf-threshold').value = S.state.conf_threshold;
    $('#notes').value = S.state.notes || '';
    $('#player').value = S.state.player || '';
    S.lastOptionsSynced = optKey;
  }
  const sm = S.state.smoothing;
  const smKey = JSON.stringify(sm);
  if (smKey !== S.lastSmoothingSynced) {
    $('#smooth-method').value = sm.method; $('#smooth-window').value = sm.window; $('#smooth-poly').value = sm.polyorder; $('#smooth-cutoff').value = sm.cutoff_hz; $('#smooth-order').value = sm.order;
    $('#smooth-savgol-opts').classList.toggle('hidden', sm.method !== 'savgol');
    $('#smooth-butter-opts').classList.toggle('hidden', sm.method !== 'butterworth');
    S.lastSmoothingSynced = smKey;
  }
  $('#undo-btn').disabled = !S.state.can_undo; $('#redo-btn').disabled = !S.state.can_redo;
  $('#save-btn').disabled = false; $('#export-btn').disabled = !S.state.has_pose; $('#compare-btn').disabled = false;
  const t = S.state.tracking;
  const share = (p) => (p.found_share == null ? '' : ` (found in ${Math.round(p.found_share * 100)}% of the frames looked at)`);
  const others = (S.state.players || []).filter((p) => !p.subject).map((p) => `the ${p.side || 'other'} player${share(p)}`);
  const who = t && t.side ? ` The subject is on the ${t.side} side${others.length ? ', and ' + others.join(', ') + ' was tracked too' : ''}.` : '';
  $('#tracking-info').textContent = t ? `Tracked frames ${t.start}-${t.end} with ${t.model} in ${t.seconds ?? '?'} s (${t.processed_at}).${who}` : 'Not tracked yet. Set the in/out points, then Run tracking.';
  const cs = S.state.corrections;
  $('#corrections-summary').textContent = cs.length ? `${cs.length} corrected keypoint${cs.length > 1 ? 's' : ''} (${cs.filter((c) => c.interpolated).length} interpolated), frames ${Math.min(...cs.map((c) => c.frame))}-${Math.max(...cs.map((c) => c.frame))}` : 'No corrections yet.';
  const sel = $('#interp-joint');
  if (!sel.options.length) S.state.meta.keypoint_names.forEach((n, i) => { const o = document.createElement('option'); o.value = i; o.textContent = n.replace(/_/g, ' '); sel.appendChild(o); });
  sel.value = S.lastJoint;
  const cal = S.state.calibration;
  $('#calib-status').textContent = cal.px_per_cm ? `Current: ${cal.px_per_cm.toFixed(3)} px/cm (${cal.method}, ${cal.reference_cm} cm). ${cal.note || ''}` : 'No calibration: distances stay in pixels.';
}

function renderAll() {
  if (!S.state) return;
  renderInfo(); renderTrim(); renderEvents(); renderOptions();
  renderVideo(); renderTimeline(); renderPlots(); renderCourt();
  renderStrokes(); renderRallies(); renderMatch(); renderReview(); renderTactics(); renderDoubles(); renderTrends(); renderCoaching(); renderCameraView();
  renderSideTabs(); renderNextStep();
}

// ---------------------------------------------------------------- folding

function foldedPanels() {
  try { return new Set(JSON.parse(localStorage.getItem('mikiri.folded') || '[]')); } catch (e) { return new Set(); }
}

function setFolded(panel, key, folded) {
  panel.classList.toggle('folded', folded);
  const btn = panel.querySelector(':scope > .panel-title > .fold');
  if (btn) { btn.textContent = folded ? '+' : '-'; btn.title = folded ? 'Unfold this panel' : 'Fold this panel to its title'; }
  const set = foldedPanels();
  if (folded) set.add(key); else set.delete(key);
  try { localStorage.setItem('mikiri.folded', JSON.stringify([...set])); } catch (e) { /* private mode */ }
}

// A fold button in front of every panel title. The title's own text is
// wrapped so it stays visible when the rest of the title row is hidden.
function bindFolding() {
  const folded = foldedPanels();
  for (const title of $$('.panel > .panel-title')) {
    const panel = title.parentElement;
    const text = title.firstChild;
    if (!text || text.nodeType !== Node.TEXT_NODE) continue;
    const key = panel.id || text.textContent.trim();
    const ttl = document.createElement('span'); ttl.className = 'ttl'; ttl.textContent = text.textContent.trim();
    title.replaceChild(ttl, text);
    const btn = document.createElement('button'); btn.className = 'btn small fold'; btn.type = 'button';
    btn.addEventListener('click', (e) => { e.stopPropagation(); setFolded(panel, key, !panel.classList.contains('folded')); if (S.state) { renderPlots(); renderCourt(); } });
    title.prepend(btn);
    setFolded(panel, key, folded.has(key));
  }
}

// ---------------------------------------------------------------- side pages

const SIDE_PAGES = ['movement', 'match', 'time', 'project'];

// A page has content when any panel on it is showing; the tab for an empty
// page is dimmed and the page explains what would fill it.
function pageHasContent(page) {
  const el = $(`.tab-page[data-page="${page}"]`);
  return [...el.children].some((c) => (c.classList.contains('panel') || c.classList.contains('view-badge')) && !c.classList.contains('hidden') && !(c.classList.contains('view-badge') && !c.textContent));
}

function showSidePage(page, remember = true) {
  if (!SIDE_PAGES.includes(page)) page = 'movement';
  S.sidePage = page;
  if (remember) { try { localStorage.setItem('mikiri.sidePage', page); } catch (e) { /* private mode */ } }
  for (const p of SIDE_PAGES) {
    $(`.tab-page[data-page="${p}"]`).classList.toggle('hidden', p !== page);
    $(`#side-tab-${p}`).classList.toggle('active', p === page);
  }
  // Canvases drawn while their page was display:none measured zero wide.
  if (page === 'movement' && S.state) { renderPlots(); renderCourt(); }
}

function renderSideTabs() {
  const q = S.state.review, t = S.state.trends;
  const counts = { match: q && q.n_open ? String(q.n_open) : '', time: t && t.n_sessions > 1 ? String(t.n_sessions) : '' };
  for (const page of SIDE_PAGES) {
    const has = pageHasContent(page);
    $(`#side-tab-${page}`).classList.toggle('empty', !has);
    const c = $(`#side-count-${page}`); if (c) c.textContent = counts[page] || '';
    const empty = $(`.tab-page[data-page="${page}"] .tab-empty`); if (empty) empty.classList.toggle('hidden', has);
  }
  showSidePage(S.sidePage, false);
}

// What to do next, from what the project has: one line and one button, in
// the order the analysis is built -- tracking, the court, the other player,
// the review, the save -- and gone when there is nothing left to suggest.
function nextStep() {
  const st = S.state;
  if (!st || S.hideStepFor === S.pid) return null;
  const court = st.court && st.court.image_points && st.court.image_points.length >= 4;
  const players = st.players || [];
  const q = st.review;
  // The court comes first: with it, one tracking run follows everyone on
  // it, and a second run to add the other player is a whole clip decoded
  // and tracked again.
  if (!court && !st.has_pose) return { text: 'Calibrate the court first: four clicks on its corners, then one tracking run follows everyone on it. Skip it for a clip that does not show the court.', btn: 'Court calibration', act: openCourtModal };
  if (!st.has_pose) return { text: 'Set the in and out points, then run tracking; with the court in, tick the other end too.', btn: 'Run tracking', act: openRunModal };
  if (!court) return { text: 'Calibrate the court, four clicks on its corners, for footwork in metres, rallies and the match figures.', btn: 'Court calibration', act: openCourtModal };
  if (players.length < 2) return { text: 'Only one player was tracked. Run tracking again with the other end ticked for serves, outcomes and the score.', btn: 'Run tracking', act: openRunModal };
  if (q && q.n_open) return { text: `${q.n_open} question${q.n_open === 1 ? '' : 's'} the rally chain could not settle ${q.n_open === 1 ? 'is' : 'are'} waiting in Review; a keypress answers each.`, btn: 'Open Review', act: () => showSidePage('match') };
  if (!st.project_path) return { text: 'Save the project to keep this session; the next clip of the same player is read against it in Over time.', btn: 'Save', act: () => saveProject(false) };
  return null;
}

function renderNextStep() {
  const s = nextStep();
  $('#next-step').classList.toggle('hidden', !s);
  if (!s) return;
  $('#next-step-text').textContent = s.text;
  const b = $('#next-step-btn'); b.textContent = s.btn; b.onclick = s.act;
}

async function refreshTabs() {
  const list = await api('GET', '/api/projects');
  const tabs = $('#project-tabs'); tabs.innerHTML = '';
  for (const p of list) {
    const t = document.createElement('div'); t.className = 'tab' + (p.id === S.pid ? ' active' : ''); t.textContent = p.name || p.video.name;
    const x = document.createElement('span'); x.className = 'close'; x.textContent = 'x'; x.title = 'close';
    x.addEventListener('click', async (ev) => { ev.stopPropagation(); await api('DELETE', `/api/projects/${p.id}`); if (p.id === S.pid) { S.pid = null; S.state = null; S.pose = null; $('#main').classList.add('no-project'); const rest = await api('GET', '/api/projects'); if (rest.length) await setProject(await api('GET', `/api/projects/${rest[rest.length - 1].id}`)); } if (S.compare && S.compare.pid === p.id) { S.compare = null; } refreshTabs(); renderAll(); });
    t.appendChild(x);
    t.addEventListener('click', async () => { if (p.id !== S.pid) await setProject(await api('GET', `/api/projects/${p.id}`)); });
    tabs.appendChild(t);
  }
}

// ---------------------------------------------------------------- actions

function seek(f) {
  if (!S.state) return;
  S.frame = clamp(Math.round(f), 0, Math.max(0, nFrames() - 1));
  renderVideo(); renderTimeline(); renderPlots(); renderCourt();
}
function step(d) { seek(S.frame + d); }

function stopPlayback() { if (S.timer) clearInterval(S.timer); S.timer = null; S.playing = false; $('#play-btn').textContent = 'Play'; }
function startPlayback() {
  if (!S.state) return;
  stopPlayback(); S.playing = true; $('#play-btn').textContent = 'Pause';
  const interval = 1000 / (fps() * S.speed);
  S.timer = setInterval(() => {
    let f = S.frame + 1;
    if (f > S.state.trim_end || f >= nFrames()) f = S.state.trim_start;
    seek(f);
  }, Math.max(8, interval));
}
function togglePlay() { if (S.playing) stopPlayback(); else startPlayback(); }

async function setTrim(start, end) {
  // Requests can overlap (two inputs changed back to back); only the newest response may win.
  const seq = (S.trimSeq = (S.trimSeq || 0) + 1);
  const state = await api('PUT', `/api/projects/${S.pid}/trim`, { start, end });
  if (seq === S.trimSeq) applyState(state);
}

async function addEvent(name, frame) {
  try { applyState(await api('POST', `/api/projects/${S.pid}/events`, { name, frame })); setStatus(`Marked "${name}" at frame ${frame}`); }
  catch (e) { setStatus(e.message, true); }
}

async function undo() { applyState(await api('POST', `/api/projects/${S.pid}/undo`)); await loadPose(); renderAll(); }
async function redo() { applyState(await api('POST', `/api/projects/${S.pid}/redo`)); await loadPose(); renderAll(); }

async function saveProject(saveAs) {
  let path = null;
  if (saveAs) { path = window.prompt('Save project as (full path ending in .mikiri.json):', S.state.project_path || ''); if (!path) return; }
  try { const st = await api('POST', `/api/projects/${S.pid}/save`, { path }); applyState(st); setStatus(`Saved ${st.saved_path}`); renderRecent(); }
  catch (e) { setStatus(e.message, true); }
}

// keypoint editing on the video canvas
function canvasToVideo(ev) {
  const canvas = $('#video-canvas'); const r = canvas.getBoundingClientRect();
  const scale = r.width / S.state.video.width;
  return { x: (ev.clientX - r.left) / scale, y: (ev.clientY - r.top) / scale, scale };
}
function nearestJoint(vx, vy, scale) {
  const kp = keypointsAt(S.frame); if (!kp) return null;
  let best = null, bd = 14 / scale;
  for (let i = 0; i < kp.length; i++) {
    if (i < 5 && i !== 0) continue;
    if (kp[i][2] <= 0) continue;
    const d = Math.hypot(kp[i][0] - vx, kp[i][1] - vy);
    if (d < bd) { bd = d; best = i; }
  }
  return best;
}
function bindVideoCanvas() {
  const canvas = $('#video-canvas');
  canvas.addEventListener('mousemove', (ev) => {
    if (!S.state) return;
    if (S.drag) { const p = canvasToVideo(ev); S.drag.x = clamp(p.x, 0, S.state.video.width); S.drag.y = clamp(p.y, 0, S.state.video.height); renderVideo(); return; }
    if (!S.edit) return;
    const p = canvasToVideo(ev); const j = nearestJoint(p.x, p.y, p.scale);
    if (j !== S.hoverJoint) { S.hoverJoint = j; renderVideo(); }
    canvas.style.cursor = j != null ? 'grab' : 'crosshair';
  });
  canvas.addEventListener('mousedown', (ev) => {
    if (!S.state || ev.button !== 0) return;
    const p = canvasToVideo(ev);
    if (S.calibPick) { S.calibPick.points.push([p.x, p.y]); renderVideo(); if (S.calibPick.points.length === 2) finishCalibPoints(); return; }
    if (S.courtPick) { S.courtPick.points.push([p.x, p.y]); renderVideo(); if (S.courtPick.points.length >= S.courtPick.needed) { finishCourtPick(); } else { setStatus(courtPickPrompt()); renderCourt(); } return; }
    if (!S.edit || !S.pose) return;
    const j = nearestJoint(p.x, p.y, p.scale);
    if (j == null) return;
    S.drag = { joint: j, x: p.x, y: p.y }; ev.preventDefault();
  });
  window.addEventListener('mouseup', async (ev) => {
    if (!S.drag) return;
    const d = S.drag; S.drag = null;
    try {
      // Fetch the corrected pose before showing the correction as applied.
      // Rendering the new state first left a window in which the panel said
      // "1 corrected keypoint" while the canvas still drew the old joint, and
      // anything reading the joint in that window -- the UI smoke test did --
      // saw it unmoved.
      const state = await api('POST', `/api/projects/${S.pid}/corrections`, { frame: S.frame, joint: d.joint, x: d.x, y: d.y });
      await loadPose();
      applyState(state);
      S.lastJoint = d.joint; $('#interp-joint').value = d.joint;
      if (!$('#interp-start').value) $('#interp-start').value = S.frame; else $('#interp-end').value = S.frame;
      setStatus(`Corrected ${S.state.meta.keypoint_names[d.joint]} on frame ${S.frame}`);
    } catch (e) { setStatus(e.message, true); }
  });
  canvas.addEventListener('contextmenu', async (ev) => {
    if (!S.edit || !S.pose) return;
    ev.preventDefault();
    const p = canvasToVideo(ev); const j = nearestJoint(p.x, p.y, p.scale);
    if (j == null) return;
    const i = poseIndex(S.frame);
    if (!S.pose.corrected[i] || !S.pose.corrected[i].includes(j)) return;
    applyState(await api('DELETE', `/api/projects/${S.pid}/corrections?frame=${S.frame}&joint=${j}`));
    await loadPose(); renderAll();
  });
}

// run tracking
async function openRunModal() {
  if (!S.state) return;
  showModal('#modal-run');
  $('#job-box').classList.add('hidden'); $('#run-cancel-job').classList.add('hidden'); $('#run-largest').disabled = false;
  $('#run-hint').textContent = 'Loading first frame and detecting people (the model downloads on first use)...';
  const canvas = $('#select-canvas'); const ctx = canvas.getContext('2d');
  S.selectFrame = S.state.trim_start; S.selectPeople = [];
  // The other player is told apart by which side of the net they stand on, so without a court there is nothing to track them by.
  const hasCourt = !!(S.state.court && S.state.court.image_points && S.state.court.image_points.length >= 4);
  const both = $('#run-both'); both.disabled = !hasCourt; both.checked = hasCourt;
  const doubles = $('#run-doubles'); doubles.disabled = !hasCourt; doubles.checked = false;
  both.onchange = () => { doubles.disabled = !both.checked; if (!both.checked) doubles.checked = false; };
  $('#run-both-hint').textContent = hasCourt ? 'The other player is told apart by which side of the net they stand on.' : 'Needs the court calibration: the other player is told apart by which side of the net they stand on.';
  try {
    const ms = await api('GET', '/api/models/status');
    $('#run-model-status').textContent = ms.all_present ? `Models ready (${ms.backend}) in ${ms.models_dir}` : (ms.allow_download ? `Some models are missing and will be downloaded once from ${ms.models.filter((m) => !m.present).map((m) => m.url).join(', ')}` : 'Models missing and downloads are disabled: set local model paths in Settings.');
    const img = new Image();
    img.src = `/api/projects/${S.pid}/frame/${S.selectFrame}`;
    await new Promise((res, rej) => { img.onload = res; img.onerror = rej; });
    const maxW = Math.min(window.innerWidth * 0.8, 900), maxH = window.innerHeight * 0.6;
    const s = Math.min(maxW / S.state.video.width, maxH / S.state.video.height);
    canvas.width = Math.round(S.state.video.width * s); canvas.height = Math.round(S.state.video.height * s);
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
    const det = await api('GET', `/api/projects/${S.pid}/detections?frame=${S.selectFrame}`);
    S.selectPeople = det.people;
    ctx.lineWidth = 3; ctx.strokeStyle = COLORS.left; ctx.font = '14px sans-serif'; ctx.fillStyle = COLORS.left;
    det.people.forEach((p, i) => { const [x0, y0, x1, y1] = p.bbox; ctx.strokeRect(x0 * s, y0 * s, (x1 - x0) * s, (y1 - y0) * s); ctx.fillText(`person ${i + 1}`, x0 * s + 4, y0 * s + 16); });
    $('#run-hint').textContent = det.people.length ? `${det.people.length} person(s) found on frame ${S.selectFrame}. Click the athlete to track, or track the largest.` : 'No person detected on the first frame. Click roughly on the athlete to help the tracker, or track the largest.';
    canvas.onclick = (ev) => {
      const r = canvas.getBoundingClientRect(); const x = (ev.clientX - r.left) / s * (canvas.width / r.width), y = (ev.clientY - r.top) / s * (canvas.height / r.height);
      startRun([x, y]);
    };
  } catch (e) { $('#run-hint').textContent = 'Could not prepare person selection: ' + e.message; setStatus(e.message, true); }
}

async function startRun(selectPoint) {
  $('#job-box').classList.remove('hidden'); $('#run-cancel-job').classList.remove('hidden'); $('#run-largest').disabled = true;
  $('#select-canvas').onclick = null;
  try {
    const { job_id } = await api('POST', `/api/projects/${S.pid}/run`, { select_point: selectPoint, players: $('#run-both').checked ? ($('#run-doubles').checked ? 4 : 2) : 1 });
    $('#run-cancel-job').onclick = () => api('POST', `/api/jobs/${job_id}/cancel`);
    const job = await pollJob(job_id, (j) => {
      const bar = $('#job-progress');
      if (j.progress < 0) bar.classList.add('indeterminate'); else { bar.classList.remove('indeterminate'); bar.style.width = Math.round(j.progress * 100) + '%'; }
      $('#job-message').textContent = `${j.message || j.state} (${j.elapsed_s} s)`;
    });
    if (job.state === 'cancelled') { setStatus('Tracking cancelled'); }
    else { setStatus(`Tracking done: ${job.result && job.result.frames} frames${job.result && job.result.cached ? ' (from cache)' : ''}`); }
    await refresh(true);
    S.suggestions = S.pose ? S.pose.suggestions : null;
    renderAll();
    if (job.state === 'done') closeModals();
  } catch (e) { $('#job-message').textContent = 'Error: ' + e.message; setStatus(e.message, true); $('#run-largest').disabled = false; }
}

// export
function openExportModal() {
  showModal('#modal-export'); $('#export-job').classList.add('hidden'); $('#export-result').textContent = '';
}
async function runExport() {
  const body = { include_videos: $('#exp-videos').checked, slowmo: +$('#exp-slowmo').value || 1, every_n: +$('#exp-every').value || 5, make_zip: $('#exp-zip').checked, out_dir: $('#exp-out').value || null };
  $('#export-job').classList.remove('hidden'); $('#export-result').textContent = ''; $('#export-run').disabled = true;
  try {
    const { job_id } = await api('POST', `/api/projects/${S.pid}/export`, body);
    const job = await pollJob(job_id, (j) => { $('#export-progress').style.width = Math.round(Math.max(0, j.progress) * 100) + '%'; $('#export-message').textContent = j.message; });
    const r = job.result;
    $('#export-result').innerHTML = `<b>Written to</b> ${r.out_dir}${r.zip ? `<br><b>Zip</b> ${r.zip} (${(r.zip_bytes / 1e6).toFixed(1)} MB)` : ''}<br>${r.files.join(', ')}`;
    setStatus(`Export complete: ${r.out_dir}`);
    await refresh(false);
  } catch (e) { $('#export-result').textContent = 'Error: ' + e.message; setStatus(e.message, true); }
  $('#export-run').disabled = false;
}

// settings
async function openSettings() {
  showModal('#modal-settings');
  const { settings, models } = await api('GET', '/api/settings');
  S.settings = settings;
  $('#set-backend').value = settings.backend; $('#set-rtmlib-mode').value = settings.rtmlib_mode; $('#set-mp-model').value = settings.mediapipe_model;
  $('#set-det-path').value = settings.rtmlib_det_model || ''; $('#set-pose-path').value = settings.rtmlib_pose_model || ''; $('#set-mp-path').value = settings.mediapipe_model_path || '';
  $('#set-models-dir').value = settings.models_dir || models.models_dir; $('#set-allow-download').checked = settings.allow_model_download;
  $('#set-downscale').value = settings.downscale_longest; $('#set-det-every').value = settings.det_every; $('#set-det-thr').value = settings.det_score_threshold; $('#set-conf').value = settings.conf_threshold; $('#set-frame-longest').value = settings.frame_cache_longest; $('#set-ort-threads').value = settings.ort_threads || 0;
  renderModelStatus(models);
}
function renderModelStatus(models) {
  $('#model-status').innerHTML = `<div><b>Model files</b> (${models.backend}, folder ${models.models_dir})</div>` + models.models.map((m) => `<div class="model"><span class="${m.present ? 'ok' : 'missing'}">${m.present ? 'present' : 'missing'}</span><span>${m.label}</span><span class="muted">${m.path}</span>${m.present ? '' : `<span class="muted">will download from ${m.url}</span>`}</div>`).join('') + '<div class="muted">Mikiri makes no other network calls.</div>';
}
async function saveSettings() {
  const body = {
    backend: $('#set-backend').value, rtmlib_mode: $('#set-rtmlib-mode').value, mediapipe_model: $('#set-mp-model').value,
    rtmlib_det_model: $('#set-det-path').value || null, rtmlib_pose_model: $('#set-pose-path').value || null, mediapipe_model_path: $('#set-mp-path').value || null,
    models_dir: $('#set-models-dir').value || null, allow_model_download: $('#set-allow-download').checked,
    downscale_longest: +$('#set-downscale').value, det_every: +$('#set-det-every').value, det_score_threshold: +$('#set-det-thr').value, conf_threshold: +$('#set-conf').value, frame_cache_longest: +$('#set-frame-longest').value, ort_threads: +$('#set-ort-threads').value || 0,
  };
  try { const { models } = await api('PUT', '/api/settings', body); renderModelStatus(models); setStatus('Settings saved'); if (S.pid) { S.imgs.clear(); await refresh(false); } }
  catch (e) { setStatus(e.message, true); }
}

// calibration
async function calibHeight() {
  try { applyState(await api('PUT', `/api/projects/${S.pid}/calibration`, { method: 'height', reference_cm: +$('#calib-height').value, frame: S.frame })); setStatus('Calibrated from athlete height'); }
  catch (e) { $('#calib-status').textContent = e.message; setStatus(e.message, true); }
}
function calibPointsStart() {
  S.calibPick = { points: [], distance: +$('#calib-dist').value };
  closeModals(); setStatus('Click two points on the video that are ' + S.calibPick.distance + ' cm apart'); renderVideo();
}
async function finishCalibPoints() {
  const pick = S.calibPick; S.calibPick = null;
  try { applyState(await api('PUT', `/api/projects/${S.pid}/calibration`, { method: 'two_points', reference_cm: pick.distance, points: pick.points, frame: S.frame })); setStatus('Calibrated from two points'); }
  catch (e) { setStatus(e.message, true); }
  renderVideo();
}

// court calibration and footwork map
function openCourtModal() {
  showModal('#modal-court');
  const sel = $('#court-preset'); const presets = (S.state.meta.court && S.state.meta.court.presets) || [];
  if (!sel.options.length) for (const p of presets) { const o = document.createElement('option'); o.value = p.id; o.textContent = p.label; sel.appendChild(o); }
  if (S.state.court) sel.value = S.state.court.preset;
  const updateHint = () => { const p = presets.find((x) => x.id === sel.value); $('#court-hint').textContent = p ? p.hint : ''; };
  sel.onchange = updateHint; updateHint();
  const c = S.state.court;
  const bits = [];
  if (c && c.image_points && c.image_points.length) {
    bits.push(`Calibrated with ${c.image_points.length} points (${c.preset}), reprojection error ${c.reprojection_error_m} m.`);
    if (c.camera) bits.push(`Camera ${c.camera.height_m.toFixed(2)} m high, ${describeCameraSpot(c.camera)}.`);
    else if (S.state.court_camera_error) bits.push(`Net posts not usable: ${S.state.court_camera_error}`);
    else bits.push('No camera geometry: far-court positions carry an uncorrected bias.');
  } else {
    bits.push('Not calibrated.');
  }
  $('#court-status').textContent = bits.join(' ');
}
function describeCameraSpot(cam) {
  const [x, y] = cam.ground_point;
  const behind = y < 0 ? `${(-y).toFixed(1)} m behind the near baseline` : `${y.toFixed(1)} m up the court`;
  return `${behind}, ${Math.abs(x - 3.05) < 0.6 ? 'near the centre line' : (x < 3.05 ? 'left of centre' : 'right of centre')}`;
}
function startCourtPick() {
  const presets = (S.state.meta.court && S.state.meta.court.presets) || [];
  const p = presets.find((x) => x.id === $('#court-preset').value) || presets[0];
  const posts = $('#court-posts').checked;
  S.courtPick = { preset: p.id, needed: p.n_points + (posts ? 2 : 0), courtNeeded: p.n_points, posts, points: [] };
  closeModals(); setStatus(`Court calibration: ${p.hint}`); renderVideo(); renderCourt();
}
function courtPickPrompt() {
  const k = S.courtPick;
  if (!k) return '';
  const i = k.points.length;
  if (i < k.courtNeeded) return `Court point ${i + 1} of ${k.courtNeeded}`;
  return i === k.courtNeeded ? 'Now click the top of the LEFT net post' : 'Now click the top of the RIGHT net post';
}
async function finishCourtPick() {
  const pick = S.courtPick; S.courtPick = null;
  const body = { preset: pick.preset, image_points: pick.points.slice(0, pick.courtNeeded), frame: S.frame };
  if (pick.posts) body.post_tops = pick.points.slice(pick.courtNeeded);
  try {
    const st = await api('PUT', `/api/projects/${S.pid}/court`, body);
    applyState(st); await loadPose(); renderAll();
    const cam = st.court && st.court.camera;
    setStatus(cam
      ? `Court calibrated (reprojection error ${st.court.reprojection_error_m} m); camera ${cam.height_m.toFixed(2)} m high`
      : `Court calibrated (reprojection error ${st.court.reprojection_error_m} m)${st.court_camera_error ? ` -- net posts not usable: ${st.court_camera_error}` : ''}`);
  } catch (e) { setStatus(e.message, true); renderVideo(); }
}
function renderCameraView() {
  const el = $('#camera-view');
  if (!el) return;
  const v = S.state && S.state.camera_view;
  if (!v) {
    const cal = S.state && S.state.court;
    el.className = 'view-badge muted';
    el.textContent = cal ? 'Camera position unknown -- click the net-post tops to unlock view-gated metrics' : '';
    el.title = '';
    return;
  }
  const n = v.available.length, total = n + v.hidden.length;
  el.className = 'view-badge ' + (v.view === 'oblique' ? 'warn' : 'ok');
  el.textContent = `${v.view} view, ${v.elevation_deg} deg above the court -- ${n} of ${total} view-dependent metrics available`;
  const NL = String.fromCharCode(10);
  el.title = v.description + NL + NL + v.hidden.map((h) => `${h.metric}: ${h.why} ${h.fix}`).join(NL);
}

function renderStrokes() {
  const panel = $('#strokes-panel');
  const sv = S.pose && S.pose.strokes;
  const has = !!(sv && sv.strokes && sv.strokes.length);
  panel.classList.toggle('hidden', !has);
  if (!has) return;
  const sum = (S.state && S.state.strokes) || {};
  const kv = (k, v) => `<span class="k">${k}</span><span>${v}</span>`;
  const dist = (obj) => Object.entries(obj || {}).map(([k, v]) => `${k} ${v}`).join(', ');
  const rows = [kv('contacts', String(sum.n_strokes ?? sv.strokes.length))];
  if (sum.by_height) rows.push(kv('contact height', dist(sum.by_height)));
  if (sum.by_side) rows.push(kv('side', dist(sum.by_side)));
  if (sum.by_zone) rows.push(kv('from', dist(sum.by_zone)));
  if (sum.n_rallies) rows.push(kv('rallies', `${sum.n_rallies}, mean ${sum.mean_rally_s ?? '-'} s`));
  if (sum.peak_racket_speed_bh) rows.push(kv('racket speed', `peak ${sum.peak_racket_speed_bh}, mean ${sum.mean_racket_speed_bh} body heights/s`));
  if (sum.max_contact_height_m) rows.push(kv('contact height', `mean ${sum.mean_contact_height_m} m, max ${sum.max_contact_height_m} m`));
  $('#strokes-summary').innerHTML = rows.join('');

  // What the shuttle search did and did not give, read off each contact's
  // own evidence, so "found the shuttle, named nothing" says why.
  const ev = sv.strokes.map((x) => x.evidence || '');
  const named = sv.strokes.filter((x) => x.shot).length;
  const searched = ev.some((e) => !e.startsWith('no shuttle evidence'));
  if (!searched) {
    $('#strokes-note').textContent = 'Shot names (smash, clear, drop) need the shuttle: Find shuttle looks for it in the frames after each contact. Until then these are honest descriptions, not shot types.';
  } else {
    const noLaunch = ev.filter((e) => e.startsWith('no shuttle launch')).length;
    const weak = ev.filter((e) => e.startsWith('shuttle launch too weak'));
    const best = Math.max(0, ...weak.map((e) => Number((e.match(/confidence ([0-9.]+)/) || [])[1] || 0)));
    const need = weak.length ? (weak[0].match(/< ([0-9.]+)/) || [])[1] : null;
    const view = ev.filter((e) => e.includes('foreshortens')).length;
    const other = sv.strokes.length - named - noLaunch - weak.length - view;
    const parts = [`Shuttle launch recovered after ${sv.strokes.length - noLaunch} of ${sv.strokes.length} contacts; ${named} named.`];
    const why = [];
    if (weak.length) why.push(`${weak.length} launch${weak.length === 1 ? '' : 'es'} too faint to name a shot (best ${best.toFixed(2)}, ${need} needed)`);
    if (view) why.push(`${view} refused because this camera view foreshortens the shuttle's path (shot names need a side-on view)`);
    if (noLaunch) why.push(`${noLaunch} with no launch found`);
    if (other > 0) why.push(`${other} whose launch fits no shot the shuttle alone can prove`);
    if (why.length) parts.push('Not named: ' + why.join('; ') + '.');
    $('#strokes-note').textContent = parts.join(' ');
  }

  const cur = S.pose ? S.frame - S.pose.start : -1;
  $('#strokes-list').innerHTML = sv.strokes.map((st) => {
    const t = ((st.frame + S.pose.start) / (S.state.fps || 30)).toFixed(2);
    const conf = st.shot ? ` <span class="muted">${Math.round(st.shot_confidence * 100)}%</span>` : '';
    const here = Math.abs(st.frame - cur) <= 2 ? ' style="outline:1px solid var(--accent,#4f8cff)"' : '';
    return `<li data-frame="${st.frame + S.pose.start}"${here}><span class="mono">${t}s</span> <b>${st.label}</b>${conf}`
      + `<span class="muted small-text"> ${st.racket_speed_bh} bh/s${st.contact_height_m != null ? `, ${st.contact_height_m} m high` : ''}</span></li>`;
  }).join('');
  for (const li of $('#strokes-list').querySelectorAll('li')) {
    li.style.cursor = 'pointer';
    li.addEventListener('click', () => seek(Number(li.dataset.frame)));
  }
}

function renderRallies() {
  const panel = $('#rally-panel');
  const b = S.state && S.state.bouts;
  const has = !!(b && b.n_bouts);
  panel.classList.toggle('hidden', !has);
  if (!has) return;
  const kv = (k, v) => `<span class="k">${k}</span><span>${v}</span>`;
  const rows = [kv('bouts', `${b.n_bouts}${b.mean_bout_s != null ? `, mean ${b.mean_bout_s} s, longest ${b.longest_bout_s} s` : ''}`)];
  if (b.distance_per_bout_m != null) rows.push(kv('distance per bout', `${b.distance_per_bout_m} m`));
  if (b.play_seconds != null) rows.push(kv('in play', `${b.play_seconds} s${b.metres_per_minute_in_play != null ? `, ${b.metres_per_minute_in_play} m/min` : ''}`));
  if (b.mean_rest_s != null) rows.push(kv('rest between', `mean ${b.mean_rest_s} s`));
  if (b.contacts_per_bout != null) rows.push(kv('contacts per bout', String(b.contacts_per_bout)));
  if (b.effective_playing_time_pct != null) rows.push(kv('playing time', `${b.effective_playing_time_pct}% of the session, work-to-rest ${b.work_density}`));
  const sp = S.state.split_steps;
  if (sp && sp.n_moves) {
    rows.push(kv('moves', String(sp.n_moves)));
    rows.push(kv('split step before', sp.split_presence_rate != null
      ? `${sp.n_moves_with_split} of ${sp.n_moves} moves (${Math.round(sp.split_presence_rate * 100)}%)`
      : `${sp.n_moves_with_split} of ${sp.n_moves} moves -- too few to give a rate`));
    if (sp.median_split_to_step_s != null) rows.push(kv('split to first step', `${sp.median_split_to_step_s} s (median)`));
  }
  $('#rally-summary').innerHTML = rows.join('');

  const fps = (S.state && S.state.fps) || 30;
  const start = S.pose ? S.pose.start : 0;
  $('#rally-list').innerHTML = (b.bouts || []).map((x, i) => {
    const t = ((x.start + start) / fps).toFixed(1);
    return `<li data-frame="${x.start + start}"><span class="mono">${t}s</span> <b>bout ${i + 1}</b>`
      + `<span class="muted small-text"> ${x.duration_s} s, ${x.distance_m} m, peak ${x.peak_speed_mps} m/s`
      + `${x.n_contacts ? `, ${x.n_contacts} contacts` : ''}</span></li>`;
  }).join('');
  for (const li of $('#rally-list').querySelectorAll('li')) {
    li.style.cursor = 'pointer';
    li.addEventListener('click', () => seek(Number(li.dataset.frame)));
  }

  const notes = [];
  if (b.note) notes.push(b.note.charAt(0).toUpperCase() + b.note.slice(1) + '.');
  if (b.not_segmented) notes.push('No separate bout could be found, so these are whole-clip figures rather than per-rally ones.');
  else if (b.trimmed_to_play) notes.push('The clip is already cut to the play, so work-to-rest would measure the editing rather than you, and is not reported.');
  if (sp && sp.n_splits) notes.push('A split step here means a brief two-footed hop before a move, which a recovery push-off can also look like: read it as preparation of some kind, and compare it against your own earlier sessions rather than a published figure.');
  $('#rally-note').textContent = notes.join(' ');
}

function renderMatch() {
  const panel = $('#match-panel');
  const m = S.state && S.state.match;
  const has = !!(m && m.n_rallies);
  panel.classList.toggle('hidden', !has);
  if (!has) return;
  const kv = (k, v) => `<span class="k">${k}</span><span>${v}</span>`;
  const fact = (f) => (f && f.value != null) ? `${Array.isArray(f.value) ? f.value.join('-') : f.value} <span class="muted small-text">(${f.how})</span>` : '<span class="muted">withheld</span>';
  const you = S.state.tracking && S.state.tracking.side;
  const who = (side) => side == null ? '?' : (side === you ? `you (${side})` : side);
  const rows = [kv('rallies', String(m.n_rallies))];
  rows.push(kv(m.score_is_absolute ? 'score (near-far)' : 'points in clip (near-far)', fact(m.score)));
  const won = m.rallies_won || {};
  rows.push(kv('rallies won', `near ${won.near ?? 0}, far ${won.far ?? 0}${won.open ? `, ${won.open} open` : ''}`));
  const sv = m.serves || {};
  rows.push(kv('serves', `${sv.detected} found; ${sv.consistent} of ${sv.checked} tested stand where the count puts them`));
  const prov = m.by_provenance || {};
  rows.push(kv('outcomes by', `confirmed ${prov.confirmed ?? 0}, inferred ${prov.inferred ?? 0}, auto ${prov.auto ?? 0}, withheld ${prov.withheld ?? 0}`));
  $('#match-summary').innerHTML = rows.join('');

  const rate = fps();
  const start = S.pose ? S.pose.start : (S.state.tracking ? S.state.tracking.start : 0);
  $('#match-list').innerHTML = (m.rallies || []).map((r) => {
    const t = ((r.start + start) / rate).toFixed(1);
    const server = r.server.value ? `${who(r.server.value)} served${r.service_court.value ? ` from the ${r.service_court.value}` : ''}` : 'server unknown';
    const winner = r.winner.value ? `${who(r.winner.value)} won` : 'open';
    const how = r.winner.how || 'withheld';
    const n = r.n_contacts ? `${r.n_contacts} contacts` : 'no contacts';
    return `<li data-frame="${r.start + start}"><span class="mono">${t}s</span> <b>rally ${r.index + 1}</b>`
      + `<span class="muted small-text"> ${r.duration_s} s, ${n}; ${server}; ${winner} <i>(${how})</i>${r.issues.length ? `; ${r.issues.length} question${r.issues.length === 1 ? '' : 's'}` : ''}</span></li>`;
  }).join('');
  $('#match-issues').innerHTML = (m.issues || []).map((i) =>
    `<li data-frame="${i.frame + start}"><span class="mono">${((i.frame + start) / rate).toFixed(1)}s</span> ${i.question}</li>`
  ).join('');
  for (const li of panel.querySelectorAll('li')) {
    li.style.cursor = 'pointer';
    li.addEventListener('click', () => seek(Number(li.dataset.frame)));
  }

  const notes = [];
  notes.push(m.headline ? m.headline.charAt(0).toUpperCase() + m.headline.slice(1) + '.' : `No result is stated: ${m.headline_withheld}.`);
  notes.push('Who won a rally is read off who serves next; the serve is the first detected contact of a bout. Confirmed means you answered it, inferred means the rules of the game derived it from confirmed facts, auto means unreviewed detection.');
  if (m.n_issues) notes.push(`${m.n_issues} question${m.n_issues === 1 ? '' : 's'} the chain could not settle, listed below and asked one at a time in the Review panel.`);
  if (m.note) notes.push(m.note.charAt(0).toUpperCase() + m.note.slice(1) + '.');
  $('#match-note').textContent = notes.join(' ');
}

// The review queue: one question at a time, the video already at its frame,
// answered by a number key. Only the first open question is shown; answering
// or leaving it brings the next. The panel seeks once per question, so the
// user can scrub away to find a hit for "at the frame I am on".
function renderReview() {
  const panel = $('#review-panel');
  const q = S.state && S.state.review;
  const tg = S.state && S.state.tagging;
  const hasQueue = !!(q && q.n_open);
  const hasContacts = !!(tg && tg.n_contacts);
  panel.classList.toggle('hidden', !hasQueue && !hasContacts);
  $('#tag-toggle').classList.toggle('hidden', !hasContacts);
  $('#tag-toggle').textContent = S.tagging ? 'Back to questions' : 'Tag shots';
  $('#review-undo').disabled = !(q && q.can_undo);
  renderClassifier(tg);
  if (!hasQueue && !hasContacts) { S.review = null; S.tag = null; return; }
  if (S.tagging && hasContacts) { S.review = null; renderTagging(tg); return; }
  S.tag = null;
  if (!hasQueue) {
    S.review = null;
    $('#review-progress').textContent = `nothing open, ${q ? q.answered : 0} answered`;
    $('#review-question').textContent = 'Every question is answered or left. Label shots to teach the classifier.';
    $('#review-detail').textContent = '';
    $('#review-options').innerHTML = '';
    $('#review-help').textContent = '';
    return;
  }
  const item = q.open[0];
  S.review = item;
  if (S.reviewSeekId !== item.id) { S.reviewSeekId = item.id; seek(item.frame); }
  $('#review-progress').textContent = `${q.n_open} open, ${q.answered} answered`;
  $('#review-question').textContent = item.question;
  $('#review-detail').textContent = item.detail;
  $('#review-options').innerHTML = item.options.map((o, i) =>
    `<button class="btn small" data-i="${i}"><b>${i + 1}</b> ${o.label}</button>`).join(' ')
    + ' <button class="btn small" data-i="d"><b>d</b> leave it</button>';
  $('#review-help').innerHTML = 'Press the number to answer, <b>d</b> to leave this one, <b>u</b> to take the last answer back. The video is at the frame in question; for "at the frame I am on", scrub to the hit first.';
  for (const b of $('#review-options').querySelectorAll('button')) {
    b.addEventListener('click', () => answerReview(b.dataset.i === 'd' ? 'd' : Number(b.dataset.i)));
  }
}

// The tactical summary: what each side played and whether it worked. Counts
// are always shown; shares only above the minimum the module withholds
// under, with the note saying which; every block names how it was known.
function renderTactics() {
  const panel = $('#tactics-panel');
  const t = S.state && S.state.tactics;
  const has = !!(t && t.n_rallies);
  panel.classList.toggle('hidden', !has);
  if (!has) return;
  const kv = (k, v) => `<span class="k">${k}</span><span>${v}</span>`;
  const pct = (x) => `${Math.round(100 * x)}%`;
  const how = (h) => h ? ` <span class="muted small-text">(${h})</span>` : ' <span class="muted small-text">(withheld)</span>';
  const counts = (obj) => Object.entries(obj || {}).map(([k, v]) => `${k} ${v}`).join(', ') || 'none';
  const r = t.rallies || {};
  const rows = [];
  rows.push(kv('rallies', `${r.n}; median ${r.shots_median} shots (spread ${r.shots_iqr}), ${r.seconds_median} s (spread ${r.seconds_iqr})`));
  rows.push(kv('by length', Object.entries(r.buckets || {}).map(([b, n]) => `${b}: ${n}`).join(', ')));
  if (r.long_share != null) rows.push(kv('11+ shots', pct(r.long_share)));
  $('#tactics-rallies').innerHTML = rows.join('');
  const you = S.state.tracking && S.state.tracking.side;
  const blocks = [];
  for (const [side, s] of Object.entries(t.sides || {})) {
    const label = side === you ? `${side} (you)` : side;
    const sh = s.shots || {}, p = s.points || {}, sv = s.serve || {};
    const lines = [];
    lines.push(kv('shots', `${sh.contacts} contacts, ${sh.named} named${sh.unnamed ? `, ${sh.unnamed} unnamed` : ''}`));
    lines.push(kv('mix', (sh.mix_share ? Object.entries(sh.mix_share).map(([k, v]) => `${k} ${pct(v)} (${sh.mix[k]})`).join(', ') : counts(sh.mix)) + how(sh.how)));
    if (sh.note) lines.push(kv('', `<span class="muted small-text">${sh.note}</span>`));
    lines.push(kv('points', `won ${p.won} (own winners ${p.won_by_own_winner}, opponent errors ${p.won_by_opponent_error}), lost ${p.lost} (own errors ${p.lost_by_own_error}, opponent winners ${p.lost_to_opponent_winner})${p.undecided ? `; ${p.undecided} undecided` : ''}` + how(p.how)));
    lines.push(kv('winners by shot', counts(p.winners_by_shot)));
    lines.push(kv('errors by shot', counts(p.errors_by_shot)));
    if (p.unnamed_endings) lines.push(kv('', `<span class="muted small-text">${p.unnamed_endings} point-ending shots unnamed; label them to see which</span>`));
    lines.push(kv('serve', `${sv.served} served, ${sv.won_serving} won${sv.won_serving_share != null ? ` (${pct(sv.won_serving_share)})` : ''}` + how(sv.how)));
    if (sv.note) lines.push(kv('', `<span class="muted small-text">${sv.note}</span>`));
    blocks.push(`<div class="panel-title">${label}</div><div class="footwork">${lines.join('')}</div>`);
  }
  $('#tactics-sides').innerHTML = blocks.join('');
  const notes = [];
  if (!t.score_is_absolute) notes.push('Points are those won in this clip; the score at its start is not known.');
  notes.push('Not measured: ' + (t.not_measured || []).join('; ') + '.');
  $('#tactics-note').textContent = notes.join(' ');
}

// Formation, for each side with two tracked partners: the shape shares,
// the seam between the partners, and how the pair answered its own shots.
// Shares are withheld under the module's minimums, with the note saying so.
function renderDoubles() {
  const panel = $('#doubles-panel');
  const d = S.state && S.state.doubles;
  const has = !!(d && d.sides && Object.keys(d.sides).length);
  panel.classList.toggle('hidden', !has);
  if (!has) return;
  const kv = (k, v) => `<span class="k">${k}</span><span>${v}</span>`;
  const pct = (x) => `${Math.round(100 * x)}%`;
  const blocks = [];
  for (const [side, s] of Object.entries(d.sides)) {
    const label = side === d.you ? `${side} (you)` : side;
    const lines = [];
    if (s.note && !s.counts) { blocks.push(`<div class="panel-title">${label}</div><div class="footwork">${kv('', s.note)}</div>`); continue; }
    lines.push(kv('located', `${s.located_s} s of ${s.play_s} s of play`));
    if (s.shares) lines.push(kv('shape', Object.entries(s.shares).map(([k, v]) => `${k} ${pct(v)}`).join(', ')));
    else lines.push(kv('shape', `${Object.entries(s.counts || {}).map(([k, v]) => `${k} ${v}`).join(', ')} frames <span class="muted small-text">${s.note || ''}</span>`));
    if (s.separation_median_m != null) lines.push(kv('apart', `${s.separation_median_m} m median; within 1.5 m of each other ${pct(s.crowded_share || 0)} of the time`));
    const r = s.response || {};
    lines.push(kv('own shots', `${r.n_judged || 0} judged (${(r.by_basis || {}).name || 0} by name, ${(r.by_basis || {}).height || 0} by contact height), ${r.n_unjudged || 0} not`));
    if (r.wrong_shape_share != null) {
      lines.push(kv('wrong shape after', `${pct(r.wrong_shape_share)} of the time <span class="muted small-text">(${r.how || 'withheld'})</span>`));
      if (r.rotation_median_s != null) lines.push(kv('time to right shape', `${r.rotation_median_s} s median, reached after ${r.reached} of ${r.n_judged}`));
    } else if (r.note) {
      lines.push(kv('', `<span class="muted small-text">${r.note}</span>`));
    }
    blocks.push(`<div class="panel-title">${label}</div><div class="footwork">${lines.join('')}</div>`);
  }
  $('#doubles-sides').innerHTML = blocks.join('');
  $('#doubles-note').textContent = 'After a lift the pair should be side-by-side, after a downward shot front-and-back; the direction comes from the shot name where there is one and from an underarm contact otherwise. Not measured: ' + (d.not_measured || []).join('; ') + '.';
}

// Over time: this clip against the same player's kept sessions. A metric
// with fewer than two readable sessions is listed as withheld with the
// count; a movement under the metric's own minimum reads as unchanged.
function renderTrends() {
  const panel = $('#trends-panel');
  const t = S.state && S.state.trends;
  const has = !!(t && t.n_sessions);
  panel.classList.toggle('hidden', !has);
  if (!has) return;
  const kv = (k, v) => `<span class="k">${k}</span><span>${v}</span>`;
  const fmt = (id, x) => (id === 'shots_per_rally' ? String(x) : `${Math.round(100 * x)}%`);
  const day = (iso) => (iso || '').slice(0, 10) || 'undated';
  $('#trends-sessions').innerHTML = (t.sessions || []).map((s) => kv(day(s.recorded_at),
    `${s.name || ''}: ${s.n_rallies} rallies${s.points ? `, points ${s.points[0]}-${s.points[1]}` : ''}${s.current ? ' <span class="muted small-text">(this clip)</span>' : ''}`)).join('');
  const rows = [];
  for (const [id, m] of Object.entries(t.metrics || {})) {
    const arrow = m.direction === 'unchanged' ? 'unchanged' : m.direction;
    rows.push(kv(m.label, `${fmt(id, m.first)} to ${fmt(id, m.last)}, ${arrow}${m.n < t.n_sessions ? ` (${m.n} of ${t.n_sessions} sessions)` : ''} <span class="muted small-text">(${m.how || 'withheld'})</span>`));
  }
  // The withheld metrics usually share one reason (too few rallies for a
  // share in most sessions), so they go on one line per reason.
  const byReason = new Map();
  for (const w of t.withheld || []) byReason.set(w.reason, [...(byReason.get(w.reason) || []), w.label]);
  for (const [reason, labels] of byReason) rows.push(kv('withheld', `<span class="muted small-text">${labels.join(', ')}: ${reason}</span>`));
  // The movement comparator on the same sessions: its metrics carry units
  // and a severity, and its advice ids say what persisted or went away.
  const c = t.coaching;
  if (c && c.n_sessions >= 2) {
    rows.push(kv('<span class="muted small-text">movement</span>', ''));
    // Only the movements that cleared their metric's own minimum are rows;
    // the rest are a count, so the panel reads as what changed.
    const unit = (m) => (m.unit ? ` ${m.unit}` : '');
    const num = (x) => (Math.abs(x) >= 100 ? Math.round(x) : Math.round(x * 100) / 100);
    const all = Object.values(c.metrics || {});
    for (const m of all.filter((m) => m.meaningful)) {
      rows.push(kv(m.label, `${num(m.first)}${unit(m)} to ${num(m.last)}${unit(m)}, ${m.direction}${m.series && m.series.length < c.n_sessions ? ` (${m.series.length} of ${c.n_sessions} sessions)` : ''}`));
    }
    const still = all.filter((m) => !m.meaningful);
    if (still.length) rows.push(kv('', `<span class="muted small-text">${still.length} other movement number${still.length === 1 ? '' : 's'} unchanged: ${still.map((m) => m.label.toLowerCase()).join(', ')}</span>`));
    const ids = (xs) => (xs || []).map((x) => x.replace(/_/g, ' ')).join(', ');
    if ((c.persistent || []).length) rows.push(kv('still to work on', ids(c.persistent)));
    if ((c.resolved || []).length) rows.push(kv('no longer flagged', ids(c.resolved)));
    if ((c.new || []).length) rows.push(kv('newly flagged', ids(c.new)));
    if (!Object.keys(c.metrics || {}).length) rows.push(kv('', '<span class="muted small-text">no movement number was measured in two of these sessions</span>'));
  }
  $('#trends-metrics').innerHTML = rows.join('');
  const notes = (t.notes || []).concat(c && c.n_sessions >= 2 ? (c.notes || []).filter((n) => !(t.notes || []).includes(n)) : []);
  $('#trends-note').textContent = notes.map((n) => n.charAt(0).toUpperCase() + n.slice(1) + '.').join(' ');
}

// The tagger: the labelling pass the per-player classifier trains on. It walks
// the chain's contacts in order, skipping the ones already labelled and the
// ones the user skipped this session, and a number key names the shot.
function renderTagging(tg) {
  S.tagSkip = S.tagSkip || new Set();
  const next = tg.contacts.find((c) => c.how !== 'labelled' && !S.tagSkip.has(`${c.side}:${c.frame}`));
  $('#review-progress').textContent = `${tg.n_labelled} of ${tg.n_contacts} contacts labelled`;
  if (!next) {
    S.tag = null;
    $('#review-question').textContent = 'Every contact is labelled or skipped.';
    $('#review-detail').textContent = '';
    $('#review-options').innerHTML = '';
    $('#review-help').textContent = '';
    return;
  }
  S.tag = next;
  const key = `${next.side}:${next.frame}`;
  if (S.tagSeekKey !== key) { S.tagSeekKey = key; seek(next.frame); }
  const guess = next.shot ? ` The app would call it a ${next.shot} (${next.how}, ${Math.round(100 * next.confidence)}%).` : '';
  $('#review-question').textContent = `What shot did the ${next.side} side play at ${(next.frame / fps()).toFixed(1)} s? It reads as ${next.descriptor}.${guess}`;
  $('#review-detail').textContent = 'Your labels train a classifier for this player alone. It names a shot only when your labels for that shot separate from every other shot you have labelled, and it says how well.';
  $('#review-options').innerHTML = tg.shots.map((sh, i) =>
    `<button class="btn small" data-i="${i}"><b>${i + 1}</b> ${sh}</button>`).join(' ')
    + ' <button class="btn small" data-i="s"><b>s</b> skip</button>';
  $('#review-help').innerHTML = 'Press the number for the shot, <b>s</b> to skip this contact, <b>t</b> to go back to the questions.';
  for (const b of $('#review-options').querySelectorAll('button')) {
    b.addEventListener('click', () => tagShot(b.dataset.i === 's' ? 's' : Number(b.dataset.i)));
  }
}

function renderClassifier(tg) {
  const el = $('#classifier-report');
  const rep = tg && tg.classifier;
  if (!rep || !Object.keys(rep).length) { el.innerHTML = ''; return; }
  const parts = [];
  for (const [side, r] of Object.entries(rep)) {
    const classes = Object.entries(r.classes || {}).map(([k, v]) => `${k} ${v}`).join(', ') || 'none';
    let line = `<b>${side}</b>: ${r.n_labelled} labelled (${classes}).`;
    if (r.pairs && r.pairs.length) {
      line += ' Pairs: ' + r.pairs.map((p) => `${p.a}/${p.b} ${Math.round(100 * p.accuracy)}% of ${p.n}${p.separable ? '' : ' (not separated)'}`).join('; ') + '.';
      line += ` Named ${r.named}, withheld ${r.withheld}.`;
    } else if (r.note) line += ' ' + r.note.charAt(0).toUpperCase() + r.note.slice(1) + '.';
    parts.push(line);
  }
  el.innerHTML = parts.join('<br>');
}

async function tagShot(choice) {
  const c = S.tag;
  if (!c) return;
  if (choice === 's') { S.tagSkip.add(`${c.side}:${c.frame}`); renderReview(); return; }
  const shot = S.state.tagging.shots[choice];
  if (!shot) return;
  try {
    applyState(await api('PUT', `/api/projects/${S.pid}/review`, { action: 'label_shot', frame: c.frame, side: c.side, value: shot }));
    setStatus(`Labelled ${shot}`);
  } catch (e) { setStatus(e.message, true); }
}

async function undoReview() {
  try { applyState(await api('PUT', `/api/projects/${S.pid}/review`, { action: 'undo', frame: S.frame })); setStatus('Took the last answer back'); }
  catch (e) { setStatus(e.message, true); }
}

async function answerReview(choice) {
  const item = S.review;
  if (!item) return;
  let body;
  if (choice === 'd') body = { action: 'dismiss', frame: item.frame, side: item.side, kind: item.kind };
  else {
    const o = item.options[choice];
    if (!o) return;
    body = { action: o.action, frame: o.frame == null ? S.frame : o.frame, side: o.side, kind: item.kind };
    // A move keeps the contact's own frame and takes the user's as the destination.
    if (o.moves) body.value = String(S.frame);
  }
  try { applyState(await api('PUT', `/api/projects/${S.pid}/review`, body)); setStatus('Answered'); }
  catch (e) { setStatus(e.message, true); }
}

function renderCoaching() {
  const panel = $('#coaching-panel');
  if (S.state && S.state.coach_level) $('#coach-level').value = S.state.coach_level;
  const advice = S.state && S.state.coaching;
  const has = !!(advice && advice.length);
  panel.classList.toggle('hidden', !has);
  if (!has) return;
  const badge = { good: 'ok', info: 'info', 'work-on': 'warn', unknown: 'muted' };
  $('#coaching-list').innerHTML = advice.map((a) => `
    <div class="advice ${badge[a.severity] || 'info'}">
      <div class="advice-title">${a.title}</div>
      <div class="advice-detail">${a.detail}</div>
      <div class="muted small-text">${a.evidence}${a.drill ? ` &middot; try: ${a.drill}` : ''}</div>
    </div>`).join('');
}

function renderCourt() {
  const panel = $('#court-panel'); const canvas = $('#court-canvas');
  if (!S.state) return;
  const meta = S.state.meta.court; const cal = S.state.court;
  const has = !!(cal && cal.image_points && cal.image_points.length >= 4);
  panel.classList.toggle('hidden', !has && !S.courtPick);
  if (!has && !S.courtPick) return;
  const margin = 1.0; const W = 220; const Hh = Math.round(W * (meta.length + 2 * margin) / (meta.width + 2 * margin));
  const dpr = window.devicePixelRatio || 1;
  if (canvas.width !== Math.round(W * dpr) || canvas.height !== Math.round(Hh * dpr)) { canvas.width = Math.round(W * dpr); canvas.height = Math.round(Hh * dpr); canvas.style.height = Hh + 'px'; }
  const ctx = canvas.getContext('2d'); ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  const scale = W / (meta.width + 2 * margin);
  const P = (x, y) => [(x + margin) * scale, Hh - (y + margin) * scale];
  ctx.fillStyle = '#3c6e3c'; ctx.fillRect(0, 0, W, Hh);
  ctx.fillStyle = '#3ca0c8'; const [cx0, cy0] = P(0, meta.length); ctx.fillRect(cx0, cy0, meta.width * scale, meta.length * scale);
  ctx.strokeStyle = '#fafafa'; ctx.lineWidth = 1.5;
  const alley = (meta.width - meta.singles_width) / 2;
  const line = (x1, y1, x2, y2) => { const a = P(x1, y1), b = P(x2, y2); ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]); ctx.stroke(); };
  for (const x of [0, alley, meta.width - alley, meta.width]) line(x, 0, x, meta.length);
  for (const y of [0, meta.long_service, meta.net_y - meta.short_service, meta.net_y + meta.short_service, meta.length - meta.long_service, meta.length]) line(0, y, meta.width, y);
  line(meta.width / 2, 0, meta.width / 2, meta.net_y - meta.short_service); line(meta.width / 2, meta.net_y + meta.short_service, meta.width / 2, meta.length);
  ctx.strokeStyle = '#222'; ctx.lineWidth = 2.5; line(-0.3, meta.net_y, meta.width + 0.3, meta.net_y);
  const pos = S.pose && S.pose.positions;
  if (pos) {
    // Heat: each sample is smeared by its own measured uncertainty, so the far
    // court reads as the smear it really is rather than a crisp dot.
    const sig = S.pose.position_sigma;
    ctx.fillStyle = 'rgba(230,30,30,0.10)';
    pos.forEach((p, i) => {
      if (!p) return;
      const [x, y] = P(p[0], p[1]);
      const s = sig && sig[i] ? sig[i] : null;
      const r = Math.max(0.25, s ? s[0] : 0.35) * scale;
      ctx.beginPath(); ctx.arc(x, y, r, 0, Math.PI * 2); ctx.fill();
    });
    ctx.strokeStyle = '#fff'; ctx.lineWidth = 1.5; ctx.beginPath(); let pen = false;
    for (const p of pos) { if (!p) { pen = false; continue; } const [x, y] = P(p[0], p[1]); if (!pen) { ctx.moveTo(x, y); pen = true; } else ctx.lineTo(x, y); }
    ctx.stroke();
    const i = poseIndex(S.frame);
    if (i >= 0 && pos[i]) { const [x, y] = P(pos[i][0], pos[i][1]); ctx.fillStyle = '#ff3b3b'; ctx.beginPath(); ctx.arc(x, y, 6, 0, Math.PI * 2); ctx.fill(); ctx.strokeStyle = '#fff'; ctx.lineWidth = 2; ctx.stroke(); }
    const fw = S.pose.footwork;
    if (fw && fw.base_m) {
      // Footwork figures are in the subject's own frame; a far-end subject's base sits a half turn away on the shared map.
      const b = S.pose.side === 'far' ? [meta.width - fw.base_m[0], meta.length - fw.base_m[1]] : fw.base_m;
      const [x, y] = P(b[0], b[1]); ctx.strokeStyle = '#ffdc00'; ctx.lineWidth = 2; ctx.beginPath(); ctx.moveTo(x - 7, y); ctx.lineTo(x + 7, y); ctx.moveTo(x, y - 7); ctx.lineTo(x, y + 7); ctx.stroke();
    }
  }
  // The other player's track, when one was tracked: a faint path and the current position.
  for (const o of (S.pose && S.pose.others) || []) {
    if (!o.positions) continue;
    ctx.strokeStyle = 'rgba(215,219,227,0.55)'; ctx.lineWidth = 1; ctx.beginPath(); let pen = false;
    for (const p of o.positions) { if (!p) { pen = false; continue; } const [x, y] = P(p[0], p[1]); if (!pen) { ctx.moveTo(x, y); pen = true; } else ctx.lineTo(x, y); }
    ctx.stroke();
    const i = poseIndex(S.frame);
    if (i >= 0 && o.positions[i]) { const [x, y] = P(o.positions[i][0], o.positions[i][1]); ctx.fillStyle = COLORS.other; ctx.beginPath(); ctx.arc(x, y, 5, 0, Math.PI * 2); ctx.fill(); ctx.strokeStyle = '#222'; ctx.lineWidth = 1.5; ctx.stroke(); }
  }
  // 1-sigma position error across the court: shows where the map can be trusted.
  const unc = S.showUncertainty !== false && S.state.court_uncertainty;
  if (unc && unc.points.length) {
    ctx.strokeStyle = 'rgba(90,230,255,0.85)'; ctx.lineWidth = 1;
    unc.points.forEach(([gx, gy], i) => {
      const [x, y] = P(gx, gy);
      ctx.beginPath();
      // Court y runs up the image while canvas rotation runs clockwise, so negate.
      ctx.ellipse(x, y, Math.max(1, unc.major_m[i] * scale), Math.max(1, unc.minor_m[i] * scale), -unc.angle_deg[i] * Math.PI / 180, 0, Math.PI * 2);
      ctx.stroke();
    });
    const worst = Math.max(...unc.major_m);
    ctx.fillStyle = 'rgba(90,230,255,0.9)'; ctx.font = '9px sans-serif';
    ctx.fillText(`95% error, worst ${Math.round(worst * 100)} cm`, 4, Hh - 4);
  }
  const fw = S.pose && S.pose.footwork; const el = $('#footwork-metrics');
  if (fw && fw.distance_m != null) {
    const zones = fw.zone_seconds ? Object.entries(fw.zone_seconds).map(([k, v]) => `${k} ${v} s`).join(', ') : '';
    const pi = poseIndex(S.frame);
    const cur = (pos && pi >= 0 && pos[pi]) ? pos[pi] : null;
    const sp = S.pose.speed && pi >= 0 ? S.pose.speed[pi] : null;
    const u = fw.position_uncertainty_m;
    const rows = [
      ['now', cur ? `x ${cur[0].toFixed(2)} m, y ${cur[1].toFixed(2)} m${sp != null ? `, ${sp.toFixed(1)} m/s` : ''}` : '-'],
      ['distance', `${fw.distance_m} m in ${fw.duration_s} s${fw.distance_uncertainty_m ? ` (+/- ${fw.distance_uncertainty_m} m)` : ''}`],
      ['speed', `avg ${fw.avg_speed_mps} m/s, peak ${fw.peak_speed_mps} m/s`],
      ['direction changes', String(fw.direction_changes)],
      ['base', `x ${fw.base_m[0]} m, y ${fw.base_m[1]} m`],
      ['from base', `mean ${fw.mean_distance_from_base_m} m, max ${fw.max_distance_from_base_m} m`],
      ['zones', zones],
      ['measured', `${fw.frames_measured} of ${fw.frames_total} frames`],
    ];
    if (u) rows.push(['position error (95%)', `${Math.round(u.median_major_m * 100)} cm ${u.major_axis_runs || 'on the long axis'}, ${Math.round(u.median_minor_m * 100)} cm the other way (worst ${Math.round(u.worst_major_m * 100)} cm)`]);
    if (fw.camera) {
      rows.push(['camera', `${fw.camera.height_m} m high at x ${fw.camera.ground_point_m[0]}, y ${fw.camera.ground_point_m[1]}`]);
      if (fw.ankle_bias_corrected_m) rows.push(['bias corrected', `${Math.round(fw.ankle_bias_corrected_m.far_baseline * 100)} cm at the far baseline`]);
    } else {
      rows.push(['camera', 'unknown -- click the net-post tops to correct the far-court bias']);
    }
    if (fw.jump_clearance_m != null) rows.push(['jump clearance', `${(fw.jump_clearance_m * 100).toFixed(0)} cm`]);
    el.innerHTML = rows.map(([k, v]) => `<span class="k">${k}</span><span>${v}</span>`).join('');
  } else {
    el.innerHTML = S.courtPick ? `<span class="k">clicking</span><span>${courtPickPrompt()}</span>` : '<span class="k">footwork</span><span>run tracking to see the track</span>';
  }
}

// compare
async function openCompare() {
  showModal('#modal-compare');
  const list = (await api('GET', '/api/projects')).filter((p) => p.id !== S.pid);
  const sel = $('#compare-project'); sel.innerHTML = '';
  for (const p of list) { const o = document.createElement('option'); o.value = p.id; o.textContent = `${p.name || p.video.name}${p.has_pose ? '' : ' (not tracked)'}`; sel.appendChild(o); }
  if (!list.length) { const o = document.createElement('option'); o.textContent = 'no other open project'; o.value = ''; sel.appendChild(o); }
  const ev = $('#compare-event'); ev.innerHTML = '<option value="">clip start</option>';
  for (const name of S.state.meta.preset_events) { const o = document.createElement('option'); o.value = name; o.textContent = name; if (name === 'takeoff') o.selected = true; ev.appendChild(o); }
}
async function startCompare() {
  const pid = $('#compare-project').value; if (!pid) return;
  const alignEvent = $('#compare-event').value;
  const state = await api('GET', `/api/projects/${pid}`);
  const pose = state.has_pose ? await api('GET', `/api/projects/${pid}/pose`) : null;
  let offset = 0;
  if (alignEvent) {
    const a = S.state.events.find((e) => e.name === alignEvent), b = state.events.find((e) => e.name === alignEvent);
    if (a && b) offset = b.frame - a.frame; else setStatus(`Event "${alignEvent}" missing in one project: aligned on clip start`, true);
  }
  S.compare = { pid, state, pose: pose && pose.has_pose ? pose : null, alignEvent, offset };
  closeModals(); renderAll();
}

// ---------------------------------------------------------------- file browser

let browseKind = 'video';
async function openBrowser(kind) {
  browseKind = kind; $('#open-kind-label').textContent = kind === 'project' ? 'project' : 'video';
  showModal('#modal-open');
  await browseTo(null);
}
async function browseTo(path) {
  try {
    const d = await api('GET', '/api/browse' + (path ? `?path=${encodeURIComponent(path)}` : ''));
    $('#browse-path').value = d.path;
    const drives = $('#browse-drives'); drives.innerHTML = '';
    for (const dr of d.drives || []) { const b = document.createElement('button'); b.className = 'btn small'; b.textContent = dr; b.addEventListener('click', () => browseTo(dr)); drives.appendChild(b); }
    const list = $('#browse-list'); list.innerHTML = '';
    if (d.parent) { const li = document.createElement('li'); li.innerHTML = '<span class="kind">up</span><span>..</span>'; li.addEventListener('click', () => browseTo(d.parent)); list.appendChild(li); }
    for (const dir of d.dirs) { const li = document.createElement('li'); li.innerHTML = `<span class="kind">dir</span><span>${dir.name}</span>`; li.addEventListener('click', () => browseTo(dir.path)); list.appendChild(li); }
    for (const f of d.files) {
      const li = document.createElement('li'); li.className = f.is_project ? 'project' : 'video';
      li.innerHTML = `<span class="kind">${f.is_project ? 'project' : 'video'}</span><span>${f.name}</span><span class="muted">${(f.size / 1e6).toFixed(1)} MB</span>`;
      li.addEventListener('click', () => openPath(f.path));
      list.appendChild(li);
    }
  } catch (e) { setStatus(e.message, true); }
}
async function openPath(path) {
  if (!path) return;
  try {
    setStatus('Opening ' + path + '...');
    const isProject = path.toLowerCase().endsWith('.json');
    const state = await api('POST', isProject ? '/api/project/open' : '/api/video/open', { path });
    closeModals(); await setProject(state);
  } catch (e) { setStatus(e.message, true); }
}
async function nativeDialog() {
  try {
    const r = await api('POST', `/api/dialog/open-file?kind=${browseKind}`);
    if (r.path) await openPath(r.path); else if (r.error) setStatus('System dialog unavailable: ' + r.error, true);
  } catch (e) { setStatus(e.message, true); }
}
// The entry point is the operating system's own file picker: the dialog every
// other program on the machine opens with, nothing to learn. The in-app
// browser exists for the case where there is no display to put a dialog on
// -- a remote or headless server -- and only then.
async function openNative(kind) {
  browseKind = kind;
  try {
    const r = await api('POST', `/api/dialog/open-file?kind=${kind}`);
    if (r.path) { await openPath(r.path); return; }
    if (r.error) { setStatus('No system file dialog here (' + r.error + '); browse instead', true); await openBrowser(kind); }
    // No path and no error: the user cancelled the dialog. Nothing to do.
  } catch (e) { setStatus(e.message, true); await openBrowser(kind); }
}
async function uploadFile(file) {
  const fd = new FormData(); fd.append('file', file);
  try { setStatus('Uploading ' + file.name + '...'); const state = await api('POST', '/api/video/upload', fd, true); await setProject(state); }
  catch (e) { setStatus(e.message, true); }
}

// ---------------------------------------------------------------- wiring

function bindUI() {
  $('#open-btn').addEventListener('click', () => openNative('video'));
  $('#welcome-open').addEventListener('click', () => openNative('video'));
  $('#open-project-btn').addEventListener('click', () => openNative('project'));
  $('#welcome-open-project').addEventListener('click', () => openNative('project'));
  $('#open-path-btn').addEventListener('click', () => openPath($('#open-path-input').value.trim()));
  $('#open-path-input').addEventListener('keydown', (e) => { if (e.key === 'Enter') openPath($('#open-path-input').value.trim()); });
  $('#upload-input').addEventListener('change', (e) => { if (e.target.files[0]) uploadFile(e.target.files[0]); });
  $('#native-dialog-btn').addEventListener('click', nativeDialog);
  $('#browse-go').addEventListener('click', () => browseTo($('#browse-path').value));
  $('#browse-path').addEventListener('keydown', (e) => { if (e.key === 'Enter') browseTo($('#browse-path').value); });
  $$('[data-close]').forEach((b) => b.addEventListener('click', closeModals));
  $('.modal-backdrop').addEventListener('click', closeModals);

  $('#play-btn').addEventListener('click', togglePlay);
  $('#prev-frame').addEventListener('click', () => step(-1));
  $('#next-frame').addEventListener('click', () => step(1));
  $('#speed').addEventListener('change', (e) => { S.speed = +e.target.value; if (S.playing) startPlayback(); });
  $('#show-skeleton').addEventListener('change', (e) => { S.showSkeleton = e.target.checked; renderVideo(); });
  $('#show-labels').addEventListener('change', (e) => { S.showLabels = e.target.checked; renderVideo(); });
  $('#swap-lr').addEventListener('change', async (e) => { applyState(await api('PUT', `/api/projects/${S.pid}/options`, { swap_lr: e.target.checked })); await loadPose(); renderAll(); });
  $('#tag-toggle').addEventListener('click', () => { S.tagging = !S.tagging; S.tagSeekKey = null; renderReview(); });
  $('#review-undo').addEventListener('click', undoReview);
  $('#edit-toggle').addEventListener('change', (e) => { S.edit = e.target.checked; $('#edit-panel').classList.toggle('hidden', !S.edit); if (S.edit && !$('#interp-start').value) $('#interp-start').value = S.frame; renderVideo(); });
  $('#undo-btn').addEventListener('click', undo);
  $('#redo-btn').addEventListener('click', redo);
  $('#set-in').addEventListener('click', () => setTrim(S.frame, Math.max(S.frame, S.state.trim_end)));
  $('#set-out').addEventListener('click', () => setTrim(Math.min(S.frame, S.state.trim_start), S.frame));
  const trimFromInputs = () => setTrim(+$('#trim-in').value, +$('#trim-out').value);
  $('#trim-in').addEventListener('change', trimFromInputs);
  $('#trim-out').addEventListener('change', trimFromInputs);
  $('#run-btn').addEventListener('click', openRunModal);
  $('#run-largest').addEventListener('click', () => startRun(null));
  $('#save-btn').addEventListener('click', (e) => saveProject(e.shiftKey));
  $('#export-btn').addEventListener('click', openExportModal);
  $('#export-run').addEventListener('click', runExport);
  $('#exp-browse').addEventListener('click', async () => { const r = await api('POST', '/api/dialog/open-file?kind=dir'); if (r.path) $('#exp-out').value = r.path; });
  $('#settings-btn').addEventListener('click', openSettings);
  $('#settings-save').addEventListener('click', saveSettings);
  $('#compare-btn').addEventListener('click', openCompare);
  $('#compare-start').addEventListener('click', startCompare);
  $('#compare-off').addEventListener('click', () => { S.compare = null; closeModals(); renderAll(); });
  $('#calib-btn').addEventListener('click', () => showModal('#modal-calib'));
  $('#court-btn').addEventListener('click', openCourtModal);
  $('#court-setup-btn').addEventListener('click', openCourtModal);
  $('#keys-btn').addEventListener('click', () => showModal('#modal-keys'));
  $('#next-step-dismiss').addEventListener('click', () => { S.hideStepFor = S.pid; renderNextStep(); });
  $$('.side-tab').forEach((t) => t.addEventListener('click', () => showSidePage(t.dataset.page)));
  // A video dropped anywhere on the page opens the way an upload does.
  window.addEventListener('dragover', (e) => { e.preventDefault(); $('#welcome').classList.add('dropping'); });
  window.addEventListener('dragleave', () => $('#welcome').classList.remove('dropping'));
  window.addEventListener('drop', (e) => { e.preventDefault(); $('#welcome').classList.remove('dropping'); const f = e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0]; if (f) uploadFile(f); });
  try { S.sidePage = localStorage.getItem('mikiri.sidePage') || 'movement'; } catch (e) { /* private mode */ }
  $('#court-start').addEventListener('click', startCourtPick);
  $('#court-clear').addEventListener('click', async () => { applyState(await api('DELETE', `/api/projects/${S.pid}/court`)); await loadPose(); closeModals(); renderAll(); });
  $('#show-court').addEventListener('change', renderVideo);
  $('#show-racket').addEventListener('change', renderVideo);
  $('#calib-height-btn').addEventListener('click', calibHeight);
  $('#calib-points-btn').addEventListener('click', calibPointsStart);
  $('#calib-clear-btn').addEventListener('click', async () => { applyState(await api('PUT', `/api/projects/${S.pid}/calibration`, { method: 'clear' })); });
  $('#custom-event-btn').addEventListener('click', () => { const n = $('#custom-event-name').value.trim(); if (n) { addEvent(n, S.frame); $('#custom-event-name').value = ''; } });
  $('#custom-event-name').addEventListener('keydown', (e) => { if (e.key === 'Enter') $('#custom-event-btn').click(); });
  $('#suggest-btn').addEventListener('click', async () => {
    if (!S.pose) { setStatus('Run tracking first', true); return; }
    const r = await api('GET', `/api/projects/${S.pid}/events/suggest`);
    S.suggestions = r.suggestions; renderSuggestions();
    if (!r.suggestions.length) setStatus('No flight phase found in the ankle signal; mark takeoff and landing by hand', true);
  });
  $('#interp-btn').addEventListener('click', async () => {
    const body = { joint: +$('#interp-joint').value, start: +$('#interp-start').value, end: +$('#interp-end').value };
    try { const st = await api('POST', `/api/projects/${S.pid}/corrections/interpolate`, body); applyState(st); await loadPose(); renderAll(); setStatus(`Interpolated ${st.interpolated_frames} frames`); }
    catch (e) { setStatus(e.message, true); }
  });
  $('#clear-corrections-btn').addEventListener('click', async () => { applyState(await api('DELETE', `/api/projects/${S.pid}/corrections/all`)); await loadPose(); renderAll(); });
  $('#smooth-method').addEventListener('change', (e) => { $('#smooth-savgol-opts').classList.toggle('hidden', e.target.value !== 'savgol'); $('#smooth-butter-opts').classList.toggle('hidden', e.target.value !== 'butterworth'); });
  $('#smooth-apply').addEventListener('click', async () => {
    const body = { method: $('#smooth-method').value, window: +$('#smooth-window').value, polyorder: +$('#smooth-poly').value, cutoff_hz: +$('#smooth-cutoff').value, order: +$('#smooth-order').value };
    try { applyState(await api('PUT', `/api/projects/${S.pid}/smoothing`, body)); await loadPose(); renderAll(); setStatus('Smoothing: ' + body.method); } catch (e) { setStatus(e.message, true); }
  });
  $('#camera-side').addEventListener('change', async (e) => applyState(await api('PUT', `/api/projects/${S.pid}/options`, { camera_side: e.target.value })));
  const optionThenPose = (key) => async (e) => { applyState(await api('PUT', `/api/projects/${S.pid}/options`, { [key]: e.target.value })); await loadPose(); S.suggestions = S.pose ? S.pose.suggestions : null; renderAll(); };
  $('#movement').addEventListener('change', optionThenPose('movement'));
  $('#coach-level').addEventListener('change', optionThenPose('coach_level'));
  $('#shuttle-btn').addEventListener('click', async () => {
    const btn = $('#shuttle-btn');
    btn.disabled = true; setStatus('Looking for the shuttle after each contact...');
    try {
      const { job_id } = await api('POST', `/api/projects/${S.pid}/shuttle`);
      const res = await pollJob(job_id, (j) => setStatus(j.message || 'Searching for the shuttle...'));
      const n = (res && res.result && res.result.launches) || 0;
      applyState(await api('GET', `/api/projects/${S.pid}`)); await loadPose(); renderAll();
      setStatus(n ? `Shuttle found after ${n} contact${n === 1 ? '' : 's'}` : 'No shuttle found; strokes keep their descriptions');
    } catch (e) { setStatus(e.message, true); } finally { btn.disabled = false; }
  });
  $('#racket-hand').addEventListener('change', optionThenPose('racket_hand'));
  $('#lead-leg').addEventListener('change', optionThenPose('lead_leg'));
  $('#conf-threshold').addEventListener('change', async (e) => { applyState(await api('PUT', `/api/projects/${S.pid}/options`, { conf_threshold: +e.target.value })); await loadPose(); renderAll(); });
  $('#notes').addEventListener('change', async (e) => applyState(await api('PUT', `/api/projects/${S.pid}/options`, { notes: e.target.value })));
  $('#player').addEventListener('change', async (e) => applyState(await api('PUT', `/api/projects/${S.pid}/options`, { player: e.target.value })));

  window.addEventListener('resize', () => { renderVideo(); renderTimeline(); renderPlots(); });
  window.addEventListener('keydown', (e) => {
    if (!S.state) return;
    const tag = (document.activeElement && document.activeElement.tagName) || '';
    if (tag === 'INPUT' || tag === 'SELECT' || tag === 'TEXTAREA') return;
    if (!$('#modal-root').classList.contains('hidden')) { if (e.key === 'Escape') closeModals(); return; }
    // Escape abandons a click-on-the-video pick. Without this a half-finished
    // calibration traps the user: the points are not enough to store, and every
    // further click adds another one.
    if (e.key === 'Escape' && (S.courtPick || S.calibPick)) {
      S.courtPick = null; S.calibPick = null;
      setStatus('Calibration cancelled'); renderVideo(); renderCourt(); e.preventDefault(); return;
    }
    // The review queue answers by number, and leaves a question with d. Only
    // while a question is showing, so the keys mean nothing otherwise.
    const reviewShowing = !$('#review-panel').classList.contains('hidden') && S.sidePage === 'match' && !e.ctrlKey && !e.metaKey;
    if (reviewShowing && (e.key === 't' || e.key === 'T') && !$('#tag-toggle').classList.contains('hidden')) {
      $('#tag-toggle').click(); e.preventDefault(); return;
    }
    if (S.tagging && S.tag && reviewShowing) {
      if (/^[1-9]$/.test(e.key)) { tagShot(Number(e.key) - 1); e.preventDefault(); return; }
      if (e.key === 's' || e.key === 'S') { tagShot('s'); e.preventDefault(); return; }
    }
    if (S.review && reviewShowing) {
      if (/^[1-9]$/.test(e.key)) { answerReview(Number(e.key) - 1); e.preventDefault(); return; }
      if (e.key === 'd' || e.key === 'D') { answerReview('d'); e.preventDefault(); return; }
    }
    if (reviewShowing && (e.key === 'u' || e.key === 'U') && !$('#review-undo').disabled) {
      undoReview(); e.preventDefault(); return;
    }
    const mult = e.shiftKey ? 10 : 1;
    if (e.key === 'ArrowLeft') { step(-mult); e.preventDefault(); }
    else if (e.key === 'ArrowRight') { step(mult); e.preventDefault(); }
    else if (e.key === ' ') { togglePlay(); e.preventDefault(); }
    else if (e.key === 'i' || e.key === 'I') $('#set-in').click();
    else if (e.key === 'o' || e.key === 'O') $('#set-out').click();
    else if (e.key === 'Home') seek(S.state.trim_start);
    else if (e.key === 'End') seek(S.state.trim_end);
    else if (e.key === '[') { const prev = S.state.events.filter((x) => x.frame < S.frame).pop(); if (prev) seek(prev.frame); }
    else if (e.key === ']') { const next = S.state.events.find((x) => x.frame > S.frame); if (next) seek(next.frame); }
    else if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'z') { e.preventDefault(); if (e.shiftKey) redo(); else undo(); }
    else if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'y') { e.preventDefault(); redo(); }
    else if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 's') { e.preventDefault(); saveProject(false); }
    else if (e.key === 'e' || e.key === 'E') { $('#edit-toggle').checked = !$('#edit-toggle').checked; $('#edit-toggle').dispatchEvent(new Event('change')); }
    else if (e.key === '?') showModal('#modal-keys');
  });
}

// The kept sessions that still have a project file, newest saved first, so
// coming back to a clip is one click rather than a walk through the disk.
async function renderRecent() {
  let rows = [];
  try { rows = await api('GET', '/api/history'); } catch (e) { rows = []; }
  const seen = new Set();
  const recent = rows.filter((r) => r.project_path && !seen.has(r.project_path) && seen.add(r.project_path))
    .sort((a, b) => String(b.saved_at || '').localeCompare(String(a.saved_at || ''))).slice(0, 8);
  $('#recent').classList.toggle('hidden', !recent.length);
  const list = $('#recent-list'); list.innerHTML = '';
  for (const r of recent) {
    const li = document.createElement('li');
    const when = String(r.recorded_at || r.saved_at || '').slice(0, 10);
    li.innerHTML = `<span class="when">${when}</span><span class="name">${r.name || r.video || 'project'}</span>` +
      (r.player ? `<span class="who">${r.player}</span>` : '') +
      (r.n_rallies ? `<span class="muted small-text">${r.n_rallies} rallies</span>` : '') +
      `<span class="path" title="${r.project_path}">${r.project_path}</span>`;
    li.addEventListener('click', () => openPath(r.project_path));
    list.appendChild(li);
  }
}

async function init() {
  bindUI(); bindTimeline(); bindVideoCanvas(); bindFolding();
  renderRecent();
  try {
    const list = await api('GET', '/api/projects');
    if (list.length) await setProject(await api('GET', `/api/projects/${list[list.length - 1].id}`));
  } catch (e) { setStatus(e.message, true); }
}

// Debug/test hooks.
window.JL = {
  S, api, seek, setProject, openPath,
  jointCanvasPos(joint) {
    const kp = keypointsAt(S.frame); if (!kp) return null;
    const canvas = $('#video-canvas'); const r = canvas.getBoundingClientRect(); const scale = r.width / S.state.video.width;
    return { x: r.left + kp[joint][0] * scale, y: r.top + kp[joint][1] * scale, scale, rect: r };
  },
  plotCanvas(name) { const p = S.plots.find((x) => x.name === name); return p ? p.canvas : null; },
  timelineBands,
};

document.addEventListener('DOMContentLoaded', init);
