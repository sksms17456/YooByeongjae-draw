// byunjae-leeds web viewer — vanilla JS, no deps.
// Hybrid render strategy:
//   - DOM <img> tiles handle the FLIP animation (CSS does the heavy lifting).
//   - After ~700ms (animation settle), snapshot to <canvas> and switch container
//     into "canvas-mode" → DOM tiles become invisible (layout preserved for next FLIP).
//   - Zoom/pan: a single CSS transform on the container scales the canvas — no per-tile work.
//   - When zoom crosses threshold (>=2x), redraw canvas at higher internal resolution
//     using 256px full-res tile sources (debounced 250ms).
//   - On new target selection: leave canvas mode → DOM tiles visible → FLIP animate
//     → snapshot back to canvas.

const els = {
  bar: document.getElementById("targets-bar"),
  container: document.getElementById("mosaic-container"),
  viewport: document.getElementById("mosaic-viewport"),
  canvas: document.getElementById("mosaic-canvas"),
  originalImg: document.getElementById("original-img"),
  hint: document.getElementById("hint"),
  scrollLeft: document.getElementById("scroll-left"),
  scrollRight: document.getElementById("scroll-right"),
};

function updateScrollButtons() {
  if (!els.scrollLeft || !els.scrollRight) return;
  const max = els.bar.scrollWidth - els.bar.clientWidth;
  els.scrollLeft.disabled = els.bar.scrollLeft <= 1;
  els.scrollRight.disabled = els.bar.scrollLeft >= max - 1;
}

function setupScrollButtons() {
  if (!els.scrollLeft || !els.scrollRight) return;
  const STEP = 320;
  els.scrollLeft.addEventListener("click", () => {
    els.bar.scrollBy({ left: -STEP, behavior: "smooth" });
  });
  els.scrollRight.addEventListener("click", () => {
    els.bar.scrollBy({ left: STEP, behavior: "smooth" });
  });
  els.bar.addEventListener("scroll", updateScrollButtons, { passive: true });
  window.addEventListener("resize", updateScrollButtons);
  updateScrollButtons();
}

const ctx = els.canvas.getContext("2d");

const state = {
  targets: [],
  current: null, // { stem, layout, tileEls, targetImg }
  zoom: 1,
  zoomThreshold: 2.0,
  isFullRes: false,
  gridSize: 60,
  containerSize: 0,
  inCanvasMode: false,
  panX: 0,
  panY: 0,
  targetOverlayAlpha: 0.30, // 0=off, 0.3=권장 (셀별 타겟 색을 모자이크 위에 합성해 색감 매칭)
};

function clampPan() {
  // 줌이 1배 이하면 이동 금지. 그 이상은 (zoom-1)*size/2 까지.
  const maxPan = Math.max(0, ((state.zoom - 1) * state.containerSize) / 2);
  state.panX = Math.max(-maxPan, Math.min(maxPan, state.panX));
  state.panY = Math.max(-maxPan, Math.min(maxPan, state.panY));
}

function applyTransform() {
  els.viewport.style.transform = `translate(${state.panX}px, ${state.panY}px) scale(${state.zoom})`;
}

// In-memory cache: tilePath -> HTMLImageElement (per resolution)
const imageCache = {
  thumb: new Map(), // 50px thumbnails
  full: new Map(), // 256px originals
};

// ---- mobile fallback grid (handled server-side via params; here we just adapt initial render)
function pickGridFallback() {
  if (window.innerWidth < 768 || (navigator.hardwareConcurrency || 4) < 4) {
    return 40;
  }
  return 60;
}

// ---- loaders
async function loadTargets() {
  // 정적 빌드(/data/targets.json) 우선, 없으면 FastAPI(/api/targets)
  let res = await fetch("/data/targets.json");
  if (!res.ok) res = await fetch("/api/targets");
  state.targets = await res.json();
  renderTargetsBar();
  if (state.targets.length === 0) {
    els.hint.textContent =
      "targets/ 가 비어있어요. 사용자가 마음에 드는 사진 4~6장을 targets/에 저장해 주세요.";
  }
}

function renderTargetsBar() {
  els.bar.innerHTML = "";
  for (const t of state.targets) {
    const div = document.createElement("div");
    div.className = "target-thumb" + (t.layout ? "" : " no-layout");
    div.setAttribute("data-stem", t.stem);
    const img = document.createElement("img");
    img.src = t.thumb_url;
    img.alt = t.stem;
    img.loading = "lazy";
    div.appendChild(img);
    div.addEventListener("click", () => selectTarget(t));
    els.bar.appendChild(div);
  }
}

async function selectTarget(t) {
  if (!t.layout) {
    els.hint.textContent = `${t.stem}: 모자이크 미생성`;
    return;
  }
  for (const el of els.bar.children) {
    el.classList.toggle("selected", el.getAttribute("data-stem") === t.stem);
  }
  // 즉시 시각 반응: 원본 갱신 + 캔버스 모드 해제 (이전 모자이크 가림 해제)
  els.originalImg.src = t.thumb_url;
  els.originalImg.alt = t.stem;
  exitCanvasMode();
  // 새 선택 시 줌·팬도 리셋 (이전 줌 상태가 새 모자이크에 그대로 남아있으면 혼란)
  state.panX = 0;
  state.panY = 0;
  requestZoomApply(1);

  // 풀해상도 타겟 이미지 비동기 프리로드
  state.pendingTargetImg = (async () => {
    const img = new Image();
    img.crossOrigin = "anonymous";
    img.decoding = "async";
    img.src = t.thumb_url;
    await img.decode().catch(() => {});
    return img;
  })();

  els.hint.textContent = "타일 재배치 중…";
  const layoutPromise = (async () => {
    let r = await fetch(`/data/layouts/${encodeURIComponent(t.stem)}.json`);
    if (!r.ok) r = await fetch(`/api/layouts/${t.stem}`);
    return r.json();
  })();
  const [layout, score, targetImg] = await Promise.all([
    layoutPromise,
    Promise.resolve({}),
    state.pendingTargetImg,
  ]);
  await applyLayout(layout);
  if (state.current) state.current.targetImg = targetImg;
  renderScore(score);
  els.hint.textContent = "Ctrl+휠/핀치 줌, 클릭+드래그로 이동";
  scheduleCanvasSnapshot(layout);
}

// ---- DOM tile management (FLIP)
function ensureTiles(cellCount) {
  const existing = state.current?.tileEls || Array.from(els.viewport.querySelectorAll("img.tile"));
  if (existing.length === cellCount) return existing;

  for (let i = cellCount; i < existing.length; i++) existing[i].remove();
  const tiles = existing.slice(0, cellCount);
  for (let i = existing.length; i < cellCount; i++) {
    const img = document.createElement("img");
    img.className = "tile";
    img.decoding = "async";
    img.draggable = false;
    els.viewport.appendChild(img);
    tiles.push(img);
  }
  return tiles;
}

function setCellSize(grid) {
  // offsetWidth/Height는 CSS transform(scale)에 영향 받지 않음 — 줌 상태에서도 정확
  state.containerSize = Math.min(els.container.offsetWidth, els.container.offsetHeight);
  const cell = state.containerSize / grid;
  els.container.style.setProperty("--cell-size", cell + "px");
}

function thumbUrl(p) {
  return `/tiles_thumb/${p}`;
}
function fullUrl(p) {
  return `/tiles_full/${p}`;
}

const FLIP_DURATION = 700;

async function applyLayout(layout) {
  const grid = layout.grid.cols;
  state.gridSize = grid;
  setCellSize(grid);

  const cells = layout.cells;
  const tiles = ensureTiles(cells.length);

  // 진행 중인 애니메이션 모두 취소 (연속 클릭 안전)
  for (const t of tiles) {
    for (const a of t.getAnimations()) a.cancel();
  }

  // 핵심 결정: 각 DOM 타일은 항상 자신의 격자 셀(i)에 고정.
  // 따라서 단순히 새 layout으로 src만 바꾸면 위치 변화가 없어 FLIP 무의미.
  // 매 선택마다 "흩뿌림 → 새 격자로 재조립" 효과로 구성.
  const cellSize = state.containerSize / grid;
  const w = els.container.offsetWidth;
  const h = els.container.offsetHeight;

  // Step 0: 모든 타일을 viewport 안 무작위 위치로 흩뿌림 (현재 화면 = scatter)
  for (const t of tiles) {
    const x = Math.random() * Math.max(0, w - cellSize);
    const y = Math.random() * Math.max(0, h - cellSize);
    t.style.left = x + "px";
    t.style.top = y + "px";
    t.style.transform = "translate3d(0,0,0)";
  }

  void els.container.offsetHeight;
  // Step 1: capture FIRST = scatter 위치
  const first = tiles.map((t) => t.getBoundingClientRect());

  // Step 2: 새 layout의 셀 위치로 이동 + src 교체
  for (let i = 0; i < cells.length; i++) {
    const cell = cells[i];
    const t = tiles[i];
    t.style.left = cell.x * cellSize + "px";
    t.style.top = cell.y * cellSize + "px";
    if (t.dataset.tilePath !== cell.tile) {
      t.src = thumbUrl(cell.tile);
      t.dataset.tilePath = cell.tile;
    }
  }
  void els.container.offsetHeight;
  const last = tiles.map((t) => t.getBoundingClientRect());

  // 부모 scale(zoom) 보정
  const zoom = state.zoom || 1;

  // Web Animations API로 FLIP — scatter → 새 격자
  let anyAnimated = 0;
  for (let i = 0; i < tiles.length; i++) {
    const dx = (first[i].left - last[i].left) / zoom;
    const dy = (first[i].top - last[i].top) / zoom;
    if (Math.abs(dx) < 0.5 && Math.abs(dy) < 0.5) continue;
    anyAnimated += 1;
    tiles[i].animate(
      [
        { transform: `translate3d(${dx}px, ${dy}px, 0)` },
        { transform: "translate3d(0, 0, 0)" },
      ],
      {
        duration: FLIP_DURATION,
        easing: "cubic-bezier(0.22, 0.95, 0.26, 1)",
        fill: "none",
      }
    );
  }
  console.log(`[flip] animating ${anyAnimated} tiles`);

  state.current = { stem: layout.target, layout, tileEls: tiles };
}

// ---- Canvas snapshot
let snapshotTimer = null;
function scheduleCanvasSnapshot(layout) {
  if (snapshotTimer) clearTimeout(snapshotTimer);
  // FLIP 종료(700ms) + 안전 마진 100ms
  snapshotTimer = setTimeout(() => {
    drawCanvas(layout, state.isFullRes ? "full" : "thumb").then(enterCanvasMode);
  }, FLIP_DURATION + 100);
}

async function loadImage(url) {
  return new Promise((resolve) => {
    const img = new Image();
    img.crossOrigin = "anonymous";
    img.onload = () => resolve(img);
    img.onerror = () => resolve(null);
    img.src = url;
  });
}

async function getCachedTile(path, kind) {
  const cache = imageCache[kind];
  if (cache.has(path)) return cache.get(path);
  const url = kind === "full" ? fullUrl(path) : thumbUrl(path);
  const img = await loadImage(url);
  if (img) cache.set(path, img);
  return img;
}

async function drawCanvas(layout, kind) {
  const grid = layout.grid.cols;
  const cells = layout.cells;
  // 내부 해상도: thumb 모드는 2x, full 모드는 격자 크기 * 70 (256px 타일 적정)
  const dpr = window.devicePixelRatio || 1;
  const baseSize = state.containerSize || els.container.getBoundingClientRect().width;
  const internal =
    kind === "full"
      ? Math.max(2400, grid * 50) // full: 충분한 해상도 (3000px 정도)
      : Math.round(baseSize * dpr); // thumb: 표시 크기 × DPR
  els.canvas.width = internal;
  els.canvas.height = internal;

  const cellPx = internal / grid;

  // 고유 타일만 사전 로드 (병렬, 100개씩 청크)
  const uniqueTiles = [...new Set(cells.map((c) => c.tile))];
  ctx.fillStyle = "#1a1a20";
  ctx.fillRect(0, 0, internal, internal);
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = "high";

  const chunkSize = 80;
  for (let i = 0; i < uniqueTiles.length; i += chunkSize) {
    const chunk = uniqueTiles.slice(i, i + chunkSize);
    await Promise.all(chunk.map((p) => getCachedTile(p, kind)));
  }

  for (const cell of cells) {
    const img = imageCache[kind].get(cell.tile);
    if (!img) continue;
    ctx.drawImage(img, cell.x * cellPx, cell.y * cellPx, cellPx, cellPx);
  }
}

function enterCanvasMode() {
  state.inCanvasMode = true;
  els.container.classList.add("canvas-mode");
}
function exitCanvasMode() {
  state.inCanvasMode = false;
  els.container.classList.remove("canvas-mode");
}

// ---- score (footer 제거됨, no-op)
function renderScore(_scoreData) {}

// ---- zoom (rAF throttled, debounced redraw)
let pendingZoom = state.zoom;
let zoomRafId = null;
let redrawTimer = null;
let zoomSettleTimer = null;

function requestZoomApply(z) {
  pendingZoom = Math.max(0.5, Math.min(6, z));
  if (zoomRafId !== null) return;
  zoomRafId = requestAnimationFrame(() => {
    zoomRafId = null;
    state.zoom = pendingZoom;
    els.viewport.classList.remove("zoom-settle");
    clampPan();
    applyTransform();
    scheduleRedraw();
    scheduleZoomSettle();
  });
}

function scheduleRedraw() {
  if (redrawTimer) clearTimeout(redrawTimer);
  redrawTimer = setTimeout(async () => {
    if (!state.current) return;
    const wantsFull = state.zoom >= state.zoomThreshold;
    if (wantsFull !== state.isFullRes) {
      state.isFullRes = wantsFull;
      await drawCanvas(state.current.layout, wantsFull ? "full" : "thumb");
    }
  }, 250);
}

function scheduleZoomSettle() {
  if (zoomSettleTimer) clearTimeout(zoomSettleTimer);
  zoomSettleTimer = setTimeout(() => {
    els.viewport.classList.add("zoom-settle");
  }, 120);
}

els.container.addEventListener(
  "wheel",
  (e) => {
    if (!e.ctrlKey && !e.metaKey) return;
    e.preventDefault();
    const delta = e.deltaY > 0 ? -0.1 : 0.1;
    requestZoomApply(state.zoom + delta * Math.max(1, state.zoom));
  },
  { passive: false }
);

// ---- pan (drag) — 마우스 + 1-finger touch
let panActive = null; // { startX, startY, panX, panY }
let pinchStart = null;

function beginPan(clientX, clientY) {
  panActive = { startX: clientX, startY: clientY, panX: state.panX, panY: state.panY };
  els.container.classList.add("dragging");
  els.viewport.classList.remove("zoom-settle"); // 팬 중엔 transition 끔
}

function updatePan(clientX, clientY) {
  if (!panActive) return;
  state.panX = panActive.panX + (clientX - panActive.startX);
  state.panY = panActive.panY + (clientY - panActive.startY);
  clampPan();
  applyTransform();
}

function endPan() {
  panActive = null;
  els.container.classList.remove("dragging");
}

els.container.addEventListener("mousedown", (e) => {
  if (e.button !== 0) return;
  beginPan(e.clientX, e.clientY);
  e.preventDefault();
});
window.addEventListener("mousemove", (e) => {
  if (panActive) updatePan(e.clientX, e.clientY);
});
window.addEventListener("mouseup", () => {
  if (panActive) endPan();
});
window.addEventListener("mouseleave", () => {
  if (panActive) endPan();
});

// Touch — 1 finger pan, 2 finger pinch
els.container.addEventListener(
  "touchstart",
  (e) => {
    if (e.touches.length === 2) {
      panActive = null;
      const [a, b] = e.touches;
      pinchStart = { dist: Math.hypot(a.pageX - b.pageX, a.pageY - b.pageY), zoom: state.zoom };
    } else if (e.touches.length === 1) {
      const t = e.touches[0];
      beginPan(t.clientX, t.clientY);
    }
  },
  { passive: false }
);
els.container.addEventListener(
  "touchmove",
  (e) => {
    if (e.touches.length === 2 && pinchStart) {
      const [a, b] = e.touches;
      const d = Math.hypot(a.pageX - b.pageX, a.pageY - b.pageY);
      requestZoomApply(pinchStart.zoom * (d / pinchStart.dist));
      e.preventDefault();
    } else if (e.touches.length === 1 && panActive) {
      const t = e.touches[0];
      updatePan(t.clientX, t.clientY);
      e.preventDefault();
    }
  },
  { passive: false }
);
els.container.addEventListener("touchend", () => {
  pinchStart = null;
  if (panActive) endPan();
});

// resize
window.addEventListener("resize", () => {
  if (state.current) {
    setCellSize(state.gridSize);
    // canvas 표시 크기는 CSS 100% 기준이라 재계산만
    if (state.inCanvasMode) {
      drawCanvas(state.current.layout, state.isFullRes ? "full" : "thumb");
    }
  }
});

// init
setupScrollButtons();
loadTargets().then(updateScrollButtons);
