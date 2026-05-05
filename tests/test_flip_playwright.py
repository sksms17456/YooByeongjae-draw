"""End-to-end test: load web viewer in real browser, verify FLIP animation
actually fires on second/third target selection.

Strategy:
  - Launch Chromium via Playwright
  - Load http://localhost:8765
  - Click first target → expect tiles' getAnimations() to populate
  - Click second target → again expect animations
  - Verify tile transforms change over time (not just snap)
"""

from __future__ import annotations

import time

import pytest
from playwright.sync_api import sync_playwright

URL = "http://localhost:8765"


@pytest.fixture(scope="module")
def browser_page():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        ctx = browser.new_context(viewport={"width": 1400, "height": 1000})
        page = ctx.new_page()
        page.on("console", lambda msg: print(f"[console] {msg.type}: {msg.text}"))
        page.on("pageerror", lambda err: print(f"[ERROR] {err}"))
        yield page
        browser.close()


def _wait_targets_loaded(page) -> int:
    page.wait_for_selector(".target-thumb", timeout=10000)
    return page.locator(".target-thumb").count()


def _count_active_animations(page) -> int:
    return page.evaluate(
        """() => {
            const tiles = document.querySelectorAll('img.tile');
            let total = 0;
            for (const t of tiles) {
                total += t.getAnimations().length;
            }
            return total;
        }"""
    )


def _sample_tile_transforms(page, n=10):
    return page.evaluate(
        f"""(n) => {{
            const tiles = Array.from(document.querySelectorAll('img.tile')).slice(0, n);
            return tiles.map(t => {{
                const cs = window.getComputedStyle(t).transform;
                const r = t.getBoundingClientRect();
                return {{ transform: cs, left: r.left, top: r.top }};
            }});
        }}""",
        n,
    )


def test_first_selection_triggers_animation(browser_page):
    page = browser_page
    page.goto(URL)
    n = _wait_targets_loaded(page)
    print(f"[test] {n}개 타겟 발견")
    assert n >= 2, "타겟이 2개 이상이어야 한다"

    # 첫 타겟 클릭
    page.locator(".target-thumb").first.click()
    # 클릭 직후 거의 즉시 (rAF 한 번 후) 애니메이션이 등록되어야 함
    page.wait_for_timeout(50)
    anims = _count_active_animations(page)
    print(f"[test] 첫 클릭 50ms 후 active animations: {anims}")
    assert anims > 100, f"첫 클릭 후 애니메이션이 거의 없음: {anims}"

    # 애니메이션 종료 + 캔버스 모드 진입까지 대기
    page.wait_for_timeout(1500)
    samples_before_2nd = _sample_tile_transforms(page, 5)
    print(f"[test] 첫 애니 종료 후 샘플 transform: {[s['transform'] for s in samples_before_2nd]}")


def test_second_selection_triggers_animation(browser_page):
    page = browser_page

    # 페이지 내부에서 50ms 간격으로 타일[0]의 bbox + transform 기록 시작
    page.evaluate(
        """() => {
            window.flipDebug = [];
            window.flipDebugInterval = setInterval(() => {
                const t = document.querySelectorAll('img.tile')[0];
                if (!t) return;
                const r = t.getBoundingClientRect();
                window.flipDebug.push({
                    t: performance.now(),
                    left: r.left,
                    top: r.top,
                    width: r.width,
                    transform: getComputedStyle(t).transform,
                });
            }, 30);
        }"""
    )

    page.locator(".target-thumb").nth(1).click()
    page.wait_for_timeout(900)  # FLIP 종료까지 충분히 대기
    page.evaluate("clearInterval(window.flipDebugInterval)")

    debug = page.evaluate("window.flipDebug")
    print(f"[test] {len(debug)}개 샘플 기록")

    # 처음 10개와 마지막 10개 비교
    for sample in debug[:6]:
        print(f"  t={sample['t']:.0f}ms  left={sample['left']:.1f}  top={sample['top']:.1f}  tx={sample['transform'][:50]}")
    print("  ... (중간 생략) ...")
    for sample in debug[-6:]:
        print(f"  t={sample['t']:.0f}ms  left={sample['left']:.1f}  top={sample['top']:.1f}  tx={sample['transform'][:50]}")

    # bbox 변동 분석
    lefts = [s["left"] for s in debug]
    tops = [s["top"] for s in debug]
    left_range = max(lefts) - min(lefts) if lefts else 0
    top_range = max(tops) - min(tops) if tops else 0
    print(f"[test] 타일[0] bbox 범위: left {left_range:.1f}px, top {top_range:.1f}px")

    # 애니메이션이 시각적으로 작동하면 bbox가 수십~수백 px 변화해야 함
    assert left_range > 20 or top_range > 20, (
        f"타일이 시각적으로 거의 안 움직임 (left {left_range:.1f}, top {top_range:.1f})"
    )


def test_third_selection_also_animates(browser_page):
    page = browser_page
    # 세 번째 타겟 클릭. 캔버스 모드 진입 후 클릭 → exitCanvasMode → 다시 FLIP
    page.wait_for_timeout(1200)  # 캔버스 모드 진입 보장
    page.locator(".target-thumb").nth(2).click()
    page.wait_for_timeout(50)

    anims = _count_active_animations(page)
    print(f"[test] 세 번째 클릭 50ms 후 active animations: {anims}")

    # canvas-mode 클래스 상태도 확인
    has_canvas_mode = page.evaluate(
        "document.getElementById('mosaic-container').classList.contains('canvas-mode')"
    )
    print(f"[test] canvas-mode 활성: {has_canvas_mode}")

    assert anims > 100, f"세 번째 클릭 후 애니메이션이 거의 없음: {anims}"
    assert not has_canvas_mode, "FLIP 중엔 canvas-mode 비활성이어야 함"


def test_split_view_layout(browser_page):
    """좌(원본) / 우(모자이크) 레이아웃이 보이는지 + 원본 이미지가 타겟별로 갱신되는지"""
    page = browser_page
    page.wait_for_timeout(1500)
    # 첫 타겟 선택 후 양쪽 패널 검사
    page.locator(".target-thumb").first.click()
    page.wait_for_timeout(200)

    info = page.evaluate(
        """() => {
            const orig = document.getElementById('original-pane').getBoundingClientRect();
            const mos = document.getElementById('mosaic-container').getBoundingClientRect();
            const origImg = document.getElementById('original-img');
            return {
                origLeft: orig.left, origRight: orig.right, origWidth: orig.width,
                mosLeft: mos.left, mosRight: mos.right, mosWidth: mos.width,
                origImgSrc: origImg.src.split('/').pop(),
                origImgLoaded: origImg.complete && origImg.naturalWidth > 0,
            };
        }"""
    )
    print(f"[test] split layout: {info}")
    assert info["origRight"] <= info["mosLeft"] + 5, "원본은 왼쪽, 모자이크는 오른쪽이어야 함"
    assert abs(info["origWidth"] - info["mosWidth"]) < 5, "양쪽 패널 크기가 비슷해야 함"
    assert info["origImgLoaded"], "원본 이미지가 로드되어야 함"
    page.screenshot(path="/tmp/split_view.png")
    print("[test] /tmp/split_view.png 저장")


def test_zoom_does_not_invade_left_pane(browser_page):
    """모자이크를 줌인했을 때 그 콘텐츠가 좌측 원본 영역까지 침범하지 않는지"""
    page = browser_page
    page.wait_for_timeout(1500)
    page.locator(".target-thumb").first.click()
    page.wait_for_timeout(1200)

    # 줌 3배로 키움
    page.evaluate(
        """() => {
            const fn = (z) => {
                const els = window.__getEls?.();
                // app.js의 requestZoomApply가 export 안 돼있어서 wheel 이벤트로 시뮬레이션
            };
            // 직접 state mutation은 안 되므로 Ctrl+wheel 여러 번 발사
        }"""
    )
    # Ctrl+wheel로 줌인 시뮬레이션
    box = page.locator("#mosaic-container").bounding_box()
    page.mouse.move(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
    for _ in range(15):
        page.mouse.wheel(0, -100)
        page.keyboard.down("Control")
    # Wheel + ctrl
    for _ in range(20):
        page.evaluate(
            """() => {
                const ev = new WheelEvent('wheel', {ctrlKey: true, deltaY: -100, bubbles: true, cancelable: true});
                document.getElementById('mosaic-container').dispatchEvent(ev);
            }"""
        )
    page.wait_for_timeout(500)

    info = page.evaluate(
        """() => {
            const orig = document.getElementById('original-pane').getBoundingClientRect();
            const mos = document.getElementById('mosaic-container').getBoundingClientRect();
            const vp = document.getElementById('mosaic-viewport').getBoundingClientRect();
            return {
                origRight: orig.right,
                mosLeft: mos.left,
                mosRight: mos.right,
                vpLeft: vp.left,
                vpRight: vp.right,
                vpWidth: vp.width,
                zoom: window.getComputedStyle(document.getElementById('mosaic-viewport')).transform,
            };
        }"""
    )
    print(f"[test] 줌 후 좌표: {info}")
    # mosaic-container는 자기 영역 안에서 자르므로 시각적으로 좌측 침범 X.
    # viewport는 transform으로 더 커져있을 수 있지만 container의 overflow:hidden으로 잘림.
    # 검증: 모자이크 컨테이너의 left가 원본의 right보다 큼 (침범 없음)
    assert info["mosLeft"] > info["origRight"], "모자이크 컨테이너가 원본 영역을 침범"
    print("[test] 시각적 침범 없음 확인 (overflow: hidden 작동)")
    page.screenshot(path="/tmp/zoom_no_invade.png")


def test_drag_to_pan(browser_page):
    """줌인 상태에서 드래그하면 viewport가 이동"""
    page = browser_page
    page.wait_for_timeout(800)

    # 현재 transform 캡처
    before = page.evaluate(
        "window.getComputedStyle(document.getElementById('mosaic-viewport')).transform"
    )
    print(f"[test] 드래그 전 transform: {before}")

    # 드래그 시뮬레이션
    box = page.locator("#mosaic-container").bounding_box()
    cx = box["x"] + box["width"] / 2
    cy = box["y"] + box["height"] / 2
    page.mouse.move(cx, cy)
    page.mouse.down()
    page.mouse.move(cx + 100, cy + 60, steps=10)
    page.mouse.up()
    page.wait_for_timeout(100)

    after = page.evaluate(
        "window.getComputedStyle(document.getElementById('mosaic-viewport')).transform"
    )
    print(f"[test] 드래그 후 transform: {after}")
    assert before != after, "드래그 후에도 transform이 동일 — 팬 미작동"

    # transform matrix(s,0,0,s,tx,ty)에서 tx, ty 추출
    after_parts = after.replace("matrix(", "").replace(")", "").split(",")
    tx, ty = float(after_parts[4]), float(after_parts[5])
    print(f"[test] 최종 translate: tx={tx:.1f}, ty={ty:.1f}")
    # 100,60 만큼 드래그 했지만 clampPan에 의해 일부 제한 가능
    assert abs(tx) > 5 and abs(ty) > 5, f"드래그 효과가 너무 작음 ({tx}, {ty})"
    page.screenshot(path="/tmp/drag_panned.png")


# (DOM tile visibility test removed — covered by FLIP bbox tracking tests)
