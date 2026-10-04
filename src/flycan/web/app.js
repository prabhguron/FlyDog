"use strict";
const $ = (id) => document.getElementById(id);
let graph = null,
  frame = null,
  mode = "demo",
  busy = false,
  stream = null,
  photo = null,
  boxes = [],
  selected = -1,
  generation = 0;
let yaw = 0,
  pitch = -0.12,
  zoom = 1,
  drag = null,
  projected = [],
  levels = [],
  lastDraw = 0;
let correcting = false,
  corrections = [],
  boxDrag = null,
  frozenSequence = null,
  imageRect = null;
const brain = $("brain"),
  bc = brain.getContext("2d"),
  scene = $("scene"),
  sc = scene.getContext("2d"),
  dc = $("dog").getContext("2d");
const reduced = matchMedia("(prefers-reduced-motion: reduce)").matches;
function error(message) {
  $("error").textContent = message;
  $("error").hidden = !message;
}
async function api(path, payload) {
  const r = await fetch(
    path,
    payload === undefined
      ? {}
      : {
          method: "POST",
          headers: { "Content-Type": "application/json", "X-FlyCan": "1" },
          body: JSON.stringify(payload),
        },
  );
  const data = await r.json();
  if (!r.ok) throw Error(data.error || "Request failed");
  return data;
}
function online(value) {
  $("connectionDot").classList.toggle("online", value);
  $("connectionText").textContent = value
    ? "Connected · local model"
    : "Disconnected · dog timeout active";
}
function resize(canvas) {
  const r = canvas.getBoundingClientRect(),
    d = Math.min(devicePixelRatio || 1, 2);
  if (
    canvas.width !== Math.round(r.width * d) ||
    canvas.height !== Math.round(r.height * d)
  ) {
    canvas.width = Math.round(r.width * d);
    canvas.height = Math.round(r.height * d);
  }
  const c = canvas.getContext("2d");
  c.setTransform(d, 0, 0, d, 0, 0);
  return [r.width, r.height];
}
function project(p, w, h) {
  let [x, y, z] = p;
  const a = x * Math.cos(yaw) + z * Math.sin(yaw),
    b = -x * Math.sin(yaw) + z * Math.cos(yaw);
  const yy = y * Math.cos(pitch) - b * Math.sin(pitch),
    zz = y * Math.sin(pitch) + b * Math.cos(pitch);
  const s = Math.min(w * 0.39, h * 0.48) * zoom;
  return [w / 2 + a * s, h * 0.48 + yy * s, zz];
}
function updateFrame(data) {
  frame = data;
  online(true);
  $("actionText").textContent = {
    forward: "Forward",
    left: "Turn left",
    right: "Turn right",
    stop: "Stop",
  }[data.action];
  $("latency").textContent = `${data.latency_ms.toFixed(0)} ms inference`;
  $("sourceBadge").textContent =
    {
      demo: "DEMO INPUT",
      image: "PHOTO DETECTOR",
      camera: "CAMERA DETECTOR",
      dataset_sample: "DATASET PHOTO",
    }[data.source] || data.source.toUpperCase();
  const found = data.observation[0] > 0;
  $("detectionStatus").textContent = found ? "CAN PRESENT" : "NO CAN";
  $("detectionStatus").classList.toggle("detected", found);
  for (const [a, p] of Object.entries(data.probabilities)) {
    $(`${a}Prob`).textContent = `${Math.round(p * 100)}%`;
    $(`${a}Fill`).style.width = `${p * 100}%`;
  }
  const r = data.robot;
  $("driverStatus").textContent = r.reason;
  $("motionValues").textContent =
    `${r.command.linear_mps.toFixed(2)} m/s · ${r.command.angular_rps.toFixed(2)} rad/s`;
  $("armButton").textContent = r.armed
    ? "Disable simulated dog"
    : "Enable simulated dog";
  $("armButton").disabled = r.estop || mode === "photo" || correcting;
  $("clearStop").hidden = !r.estop;
  if (!correcting) boxes = data.detections || [];
  detail();
}
function detail() {
  if (selected < 0 || !frame) return;
  const n = graph.nodes[selected];
  $("neuronInfo").textContent =
    `${n.cell_type} · ${n.side} · ${n.transmitter} | FlyWire ${n.root_id} | activation ${frame.activity[selected].toFixed(4)} | can-linked change ${frame.can_response[selected].toFixed(4)}`;
}
function drawBrain() {
  if (!graph) return;
  const [w, h] = resize(brain);
  bc.clearRect(0, 0, w, h);
  const stage = Number($("stage").value),
    total = $("signalMode").value === "activity";
  const target = frame
    ? total
      ? frame.activity
      : frame.stages[stage]
    : graph.nodes.map(() => 0);
  const signs = frame ? frame.stage_signs[stage] : [];
  levels = target.map((v, i) =>
    reduced ? v : (levels[i] ?? 0) + (v - (levels[i] ?? 0)) * 0.22,
  );
  const context = graph.context.map((p) => project(p, w, h));
  for (const p of context) {
    bc.fillStyle = `rgba(114,150,168,${Math.max(0.1, Math.min(0.38, 0.22 + p[2] * 0.13))})`;
    bc.beginPath();
    bc.arc(p[0], p[1], 0.85, 0, Math.PI * 2);
    bc.fill();
  }
  projected = graph.nodes.map((n) => project(n.position, w, h));
  if ($("edges").checked) {
    for (const e of graph.edges) {
      const a = projected[e.source],
        b = projected[e.target],
        l = Math.min(1, levels[e.source] / (total ? 0.8 : 0.3));
      bc.strokeStyle = `rgba(89,176,172,${0.015 + 0.065 * l})`;
      bc.lineWidth = 0.55;
      bc.beginPath();
      bc.moveTo(a[0], a[1]);
      bc.lineTo(b[0], b[1]);
      bc.stroke();
    }
  }
  const sorted = graph.nodes
    .map((n, i) => i)
    .sort((a, b) => projected[a][2] - projected[b][2]);
  for (const i of sorted) {
    const [x, y] = projected[i],
      intensity = Math.min(1, levels[i] / (total ? 0.8 : 0.3));
    const rgb = total || signs[i] >= 0 ? "101,222,208" : "238,184,108";
    if (intensity > 0.015) {
      const radius = 5 + intensity * 13,
        g = bc.createRadialGradient(x, y, 0, x, y, radius);
      g.addColorStop(0, `rgba(${rgb},${0.3 * intensity})`);
      g.addColorStop(1, `rgba(${rgb},0)`);
      bc.fillStyle = g;
      bc.beginPath();
      bc.arc(x, y, radius, 0, Math.PI * 2);
      bc.fill();
    }
    bc.fillStyle =
      intensity > 0.015
        ? `rgba(${rgb},${0.3 + 0.7 * intensity})`
        : "rgba(158,186,197,.5)";
    bc.beginPath();
    bc.arc(x, y, 1.7 + intensity * 2.4, 0, Math.PI * 2);
    bc.fill();
    if (i === selected) {
      bc.strokeStyle = "#effcff";
      bc.lineWidth = 1;
      bc.beginPath();
      bc.arc(x, y, 7 + intensity * 2, 0, Math.PI * 2);
      bc.stroke();
    }
  }
  const mean = target.reduce((a, b) => a + b, 0) / target.length;
  $("responseValue").textContent = mean.toFixed(3);
  $("responseValue").nextElementSibling.textContent = total
    ? "mean absolute activation"
    : "mean can-linked response";
}
function drawScene() {
  const [w, h] = resize(scene);
  sc.clearRect(0, 0, w, h);
  sc.fillStyle = "#0b151b";
  sc.fillRect(0, 0, w, h);
  imageRect = null;
  if (mode !== "demo" && photo) {
    const s = Math.min(w / photo.width, h / photo.height);
    imageRect = {
      x: (w - photo.width * s) / 2,
      y: (h - photo.height * s) / 2,
      w: photo.width * s,
      h: photo.height * s,
    };
    const r = imageRect;
    sc.drawImage(photo, r.x, r.y, r.w, r.h);
    const shown = correcting ? corrections : boxes.map((b) => b.xywhn);
    for (const b of shown) {
      const [cx, cy, bw, bh] = b,
        x = r.x + (cx - bw / 2) * r.w,
        y = r.y + (cy - bh / 2) * r.h;
      sc.strokeStyle = correcting ? "#eeb86c" : "#65ded0";
      sc.lineWidth = 1.5;
      sc.strokeRect(x, y, bw * r.w, bh * r.h);
      sc.fillStyle = sc.strokeStyle;
      sc.font = "10px sans-serif";
      sc.fillText(
        correcting ? "can · corrected" : "can",
        x + 3,
        Math.max(y + 11, 11),
      );
    }
    if (boxDrag) {
      sc.strokeStyle = "#eeb86c";
      sc.setLineDash([4, 3]);
      sc.strokeRect(
        boxDrag.x,
        boxDrag.y,
        boxDrag.endX - boxDrag.x,
        boxDrag.endY - boxDrag.y,
      );
      sc.setLineDash([]);
    }
  } else if (mode === "demo") {
    sc.strokeStyle = "#253a45";
    sc.lineWidth = 0.7;
    for (let i = 0; i < 8; i++) {
      sc.beginPath();
      sc.moveTo(w / 2, h * 0.4);
      sc.lineTo((i * w) / 7, h);
      sc.stroke();
    }
    for (let i = 0; i < 4; i++) {
      sc.beginPath();
      sc.moveTo(0, h * (0.55 + i * 0.13));
      sc.lineTo(w, h * (0.55 + i * 0.13));
      sc.stroke();
    }
    if ($("visible").checked) {
      const x = w / 2 + Number($("bearing").value) * w * 0.4,
        cw = 10 + Number($("width").value) * w * 0.75,
        ch = cw * 1.7,
        y = h * 0.44;
      sc.fillStyle = "#366d72";
      sc.fillRect(x - cw / 2, y - ch / 2, cw, ch);
      sc.fillStyle = "#70b4b3";
      sc.beginPath();
      sc.ellipse(x, y - ch / 2, cw / 2, 3, 0, 0, Math.PI * 2);
      sc.fill();
      sc.fillStyle = "#badcd7";
      sc.fillRect(x - cw / 2 + 2, y - 2, cw - 4, 9);
      sc.strokeStyle = "#65ded0";
      sc.strokeRect(x - cw / 2 - 5, y - ch / 2 - 6, cw + 10, ch + 12);
    }
  }
}
function drawDog() {
  const [w, h] = resize($("dog"));
  dc.clearRect(0, 0, w, h);
  dc.strokeStyle = "#263944";
  dc.lineWidth = 0.6;
  for (let x = 0; x < w; x += 14) {
    dc.beginPath();
    dc.moveTo(x, 0);
    dc.lineTo(x, h);
    dc.stroke();
  }
  for (let y = 0; y < h; y += 14) {
    dc.beginPath();
    dc.moveTo(0, y);
    dc.lineTo(w, y);
    dc.stroke();
  }
  const r = frame?.robot,
    turn = r?.driver?.heading || 0,
    moving = r?.armed && r.command.action !== "stop";
  dc.save();
  dc.translate(w / 2, h / 2);
  dc.rotate(-turn - Math.PI / 2);
  dc.fillStyle = moving ? "#65ded0" : "#6d858f";
  dc.fillRect(-13, -9, 26, 18);
  dc.fillRect(13, -6, 9, 12);
  for (const x of [-10, 7]) for (const y of [-15, 10]) dc.fillRect(x, y, 5, 6);
  dc.restore();
}
function animation(t) {
  if (t - lastDraw > 32) {
    drawBrain();
    drawScene();
    drawDog();
    lastDraw = t;
  }
  requestAnimationFrame(animation);
}
function stopCamera() {
  if (stream) stream.getTracks().forEach((t) => t.stop());
  stream = null;
  $("video").srcObject = null;
}
function cancelCorrection() {
  correcting = false;
  corrections = [];
  frozenSequence = null;
  boxDrag = null;
  scene.classList.remove("annotating");
  $("saveFeedback").hidden = true;
  $("cancelFeedback").hidden = true;
}
async function setMode(next) {
  generation++;
  cancelCorrection();
  stopCamera();
  mode = next;
  photo = null;
  boxes = [];
  for (const m of ["demo", "photo", "camera"]) {
    $(`${m}Tab`).classList.toggle("active", m === next);
    $(`${m}Tab`).setAttribute("aria-pressed", String(m === next));
    $(`${m}Controls`).hidden = m !== next;
  }
  $("feedbackPanel").hidden = true;
  $("previewLabel").textContent =
    next === "demo"
      ? "Synthetic stimulus · bypasses detector"
      : next === "photo"
        ? "Upload a photo or try the dataset"
        : "Camera stopped";
  await api("/api/robot", { command: "disarm" });
}
async function imageToData(blob) {
  return await new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result.split(",")[1]);
    reader.onerror = reject;
    reader.readAsDataURL(blob);
  });
}
async function bitmapFrom(blob) {
  return await createImageBitmap(blob, { imageOrientation: "from-image" });
}
async function processPhoto(blob, isSample = false) {
  if (blob.size > 4 * 1024 * 1024)
    throw Error("Choose an image smaller than 4 MB.");
  const gen = generation;
  photo = await bitmapFrom(blob);
  $("previewLabel").textContent = "Running can detector…";
  busy = true;
  try {
    const payload = { front_range_m: Number($("frontRange").value) };
    let result;
    if (isSample) result = await api("/api/sample-detect", payload);
    else
      result = await api("/api/image", {
        ...payload,
        image_base64: await imageToData(blob),
        live: false,
      });
    if (gen !== generation) return;
    updateFrame(result);
    $("previewLabel").textContent = "Trained detector · snapshot";
    $("feedbackPanel").hidden = false;
    $("feedbackStatus").textContent = isSample
      ? "Dataset test photo: feedback is saved for audit, excluded from retraining."
      : "";
    error("");
  } finally {
    busy = false;
  }
}
async function cameraFrame() {
  const v = $("video");
  if (!stream || v.readyState < 2) return;
  const c = document.createElement("canvas");
  c.width = 640;
  c.height = Math.round((v.videoHeight * 640) / v.videoWidth);
  c.getContext("2d").drawImage(v, 0, 0, c.width, c.height);
  const blob = await new Promise((resolve) =>
    c.toBlob(resolve, "image/jpeg", 0.85),
  );
  photo = await bitmapFrom(blob);
  const gen = generation;
  const result = await api("/api/image", {
    image_base64: await imageToData(blob),
    front_range_m: Number($("frontRange").value),
    live: true,
  });
  if (gen === generation) {
    updateFrame(result);
    $("feedbackPanel").hidden = false;
    $("previewLabel").textContent = "Live can detector · simulated range";
  }
}
async function tick() {
  if (!busy && !correcting) {
    busy = true;
    try {
      if (mode === "demo") {
        const data = await api("/api/stimulus", {
          visible: $("visible").checked,
          bearing: -Number($("bearing").value),
          width: Number($("width").value),
          front_range_m: Number($("frontRange").value),
        });
        updateFrame(data);
      } else if (mode === "camera" && stream) {
        await cameraFrame();
      } else updateFrame(await api("/api/state"));
    } catch (e) {
      online(false);
      error(e.message);
    } finally {
      busy = false;
    }
  }
  setTimeout(tick, 220);
}
for (const a of ["forward", "left", "right", "stop"]) {
  const div = document.createElement("div");
  div.innerHTML = `<div class="prob-label"><span>${a}</span><span id="${a}Prob">0%</span></div><div class="prob-track"><div id="${a}Fill" class="prob-fill"></div></div>`;
  $("probabilities").append(div);
}
for (const m of ["demo", "photo", "camera"])
  $(`${m}Tab`).onclick = () => setMode(m).catch((e) => error(e.message));
$("bearing").oninput = () => {
  const n = Number($("bearing").value);
  $("bearingValue").textContent =
    Math.abs(n) < 0.04
      ? "Center"
      : `${n < 0 ? "Left" : "Right"} ${Math.round(Math.abs(n) * 100)}%`;
};
$("width").oninput = () =>
  ($("widthValue").textContent =
    `${Math.round(Number($("width").value) * 100)}%`);
$("frontRange").oninput = () =>
  ($("rangeValue").textContent =
    `${Number($("frontRange").value).toFixed(2)} m`);
$("signalMode").onchange = () => {
  const total = $("signalMode").value === "activity";
  $("stage").disabled = total;
  $("positiveLegend").textContent = total ? "Activation magnitude" : "Increased activation";
  $("negativeLegend").hidden = total;
};
$("resetView").onclick = () => {
  yaw = 0;
  pitch = -0.12;
  zoom = 1;
};
$("neuronSelect").onchange = () => {
  selected = Number($("neuronSelect").value);
  if ($("neuronSelect").value === "") selected = -1;
  detail();
};
brain.onpointerdown = (e) => {
  brain.setPointerCapture(e.pointerId);
  drag = { x: e.clientX, y: e.clientY, moved: 0 };
};
brain.onpointermove = (e) => {
  if (!drag) return;
  const dx = e.clientX - drag.x,
    dy = e.clientY - drag.y;
  yaw += dx * 0.008;
  pitch = Math.max(-1.4, Math.min(1.4, pitch + dy * 0.008));
  drag.moved += Math.abs(dx) + Math.abs(dy);
  drag.x = e.clientX;
  drag.y = e.clientY;
};
brain.onpointerup = (e) => {
  if (drag && drag.moved < 5) {
    const r = brain.getBoundingClientRect(),
      x = e.clientX - r.left,
      y = e.clientY - r.top;
    let best = 14;
    selected = -1;
    projected.forEach((p, i) => {
      const d = Math.hypot(p[0] - x, p[1] - y);
      if (d < best) {
        best = d;
        selected = i;
      }
    });
    $("neuronSelect").value = selected < 0 ? "" : String(selected);
    detail();
  }
  drag = null;
};
brain.onpointercancel = () => (drag = null);
brain.addEventListener(
  "wheel",
  (e) => {
    e.preventDefault();
    zoom = Math.max(0.5, Math.min(3, zoom * Math.exp(-e.deltaY * 0.001)));
  },
  { passive: false },
);
$("photoFile").onchange = () => {
  const file = $("photoFile").files[0];
  if (file) processPhoto(file).catch((e) => error(e.message));
};
$("sampleButton").onclick = async () => {
  try {
    const r = await fetch("/api/sample");
    await processPhoto(await r.blob(), true);
  } catch (e) {
    error(e.message);
  }
};
$("startCamera").onclick = async () => {
  try {
    generation++;
    stream = await navigator.mediaDevices.getUserMedia({
      video: { width: 640, height: 480 },
      audio: false,
    });
    $("video").srcObject = stream;
    await $("video").play();
    $("previewLabel").textContent = "Camera ready";
    error("");
  } catch (e) {
    error(`Camera unavailable: ${e.message}`);
  }
};
$("stopCamera").onclick = async () => {
  stopCamera();
  await api("/api/robot", { command: "disarm" });
  $("previewLabel").textContent = "Camera stopped";
};
$("armButton").onclick = () =>
  api("/api/robot", { command: frame?.robot.armed ? "disarm" : "arm" }).catch(
    (e) => error(e.message),
  );
$("stopButton").onclick = async () => {
  try {
    const r = await api("/api/robot", { command: "stop" });
    if (frame) updateFrame({ ...frame, robot: r });
  } catch (e) {
    error(e.message);
  }
};
$("clearStop").onclick = () =>
  api("/api/robot", { command: "clear" }).catch((e) => error(e.message));
function startCorrection() {
  if (busy || !photo || !frame) {
    error("Wait for the current image to finish processing.");
    return false;
  }
  correcting = true;
  stopCamera();
  frozenSequence = frame.sequence;
  corrections = [];
  $("saveFeedback").hidden = false;
  $("cancelFeedback").hidden = false;
  scene.classList.add("annotating");
  $("feedbackStatus").textContent =
    "Draw all can boxes on the image. Existing predictions have been cleared.";
  api("/api/robot", { command: "disarm" })
    .then((r) => {
      if (frame) updateFrame({ ...frame, robot: r });
    })
    .catch((e) => error(e.message));
  return true;
}
$("drawFeedback").onclick = startCorrection;
$("cancelFeedback").onclick = cancelCorrection;
async function saveCorrections(negative = false) {
  if (!correcting && !startCorrection()) return;
  if (!negative && corrections.length === 0) {
    error("Draw at least one can box, or choose “No cans here”.");
    return;
  }
  try {
    const r = await api("/api/feedback", {
      frame_sequence: frozenSequence,
      boxes: negative ? [] : corrections,
      confirmed_complete: true,
    });
    $("feedbackStatus").textContent = r.training_eligible
      ? "Correction saved for reviewed retraining. Current weights unchanged."
      : "Correction saved for audit; held-out test photo excluded from training.";
    cancelCorrection();
    error("");
  } catch (e) {
    error(e.message);
  }
}
$("negativeFeedback").onclick = () => saveCorrections(true);
$("saveFeedback").onclick = () => saveCorrections(false);
function imagePoint(e) {
  const r = scene.getBoundingClientRect();
  return {
    x: Math.max(
      imageRect.x,
      Math.min(imageRect.x + imageRect.w, e.clientX - r.left),
    ),
    y: Math.max(
      imageRect.y,
      Math.min(imageRect.y + imageRect.h, e.clientY - r.top),
    ),
  };
}
scene.onpointerdown = (e) => {
  if (!correcting || !imageRect) return;
  scene.setPointerCapture(e.pointerId);
  const p = imagePoint(e);
  boxDrag = { ...p, endX: p.x, endY: p.y };
};
scene.onpointermove = (e) => {
  if (boxDrag) {
    const p = imagePoint(e);
    boxDrag.endX = p.x;
    boxDrag.endY = p.y;
  }
};
scene.onpointerup = () => {
  if (boxDrag && imageRect) {
    const b = boxDrag,
      r = imageRect,
      w = Math.abs(b.endX - b.x),
      h = Math.abs(b.endY - b.y);
    if (w > 3 && h > 3)
      corrections.push([
        ((b.x + b.endX) / 2 - r.x) / r.w,
        ((b.y + b.endY) / 2 - r.y) / r.h,
        w / r.w,
        h / r.h,
      ]);
    $("feedbackStatus").textContent =
      `${corrections.length} corrected can box(es). Include every can before saving.`;
  }
  boxDrag = null;
};
scene.onpointercancel = () => (boxDrag = null);
window.addEventListener("pagehide", stopCamera);
async function start() {
  try {
    graph = await api("/api/graph");
    for (const n of graph.nodes) {
      const o = document.createElement("option");
      o.value = n.index;
      o.textContent = `${n.index + 1} · ${n.cell_type} · ${n.side}`;
      $("neuronSelect").append(o);
    }
    updateFrame(await api("/api/state"));
    requestAnimationFrame(animation);
    tick();
  } catch (e) {
    error(e.message);
    online(false);
  }
}
start();
