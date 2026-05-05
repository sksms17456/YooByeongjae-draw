"""Classify tiles: face detection + CLIP zero-shot + writes index/tiles_meta.json.

Also calls preprocess.process_tile to fill numeric index.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
from tqdm import tqdm

from preprocess import CSV_FIELDS, process_tile
from utils import (
    INDEX_CSV,
    META_JSON,
    TILES_DIR,
    ensure_dirs,
    iter_tile_paths,
    load_image_rgb,
    rel_to_project,
    source_of,
)

CONTENT_PROMPTS = {
    "face": "a close-up photo of a single person's face",
    "half_body": "a half-body shot of one person",
    "full_body": "a full body shot of a person standing",
    "scene": "a landscape or scene without people",
    "text_poster": "a poster or text overlay or graphic design",
    "multi_person": "a photo of multiple people together",
}

EXPRESSION_PROMPTS = {
    "smile": "a smiling face",
    "serious": "a serious or neutral face",
    "surprised": "a surprised face with wide eyes",
    "sad": "a sad or melancholic face",
    "angry": "an angry face",
    "laughing": "a laughing face with mouth open",
}

CONFIDENCE_THRESHOLD = 0.4


_FACE_CASCADE = None


def _face_cascade():
    global _FACE_CASCADE
    if _FACE_CASCADE is None:
        import cv2

        path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        _FACE_CASCADE = cv2.CascadeClassifier(path)
    return _FACE_CASCADE


def _detect_face(rgb: np.ndarray) -> tuple[bool, list[int]]:
    """OpenCV haarcascade — 가볍고 안정적. Bundled with opencv-python."""
    import cv2

    cascade = _face_cascade()
    if cascade.empty():
        return False, [-1, -1, -1, -1]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    faces = cascade.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=4, minSize=(48, 48))
    if len(faces) == 0:
        return False, [-1, -1, -1, -1]
    # pick largest
    x, y, bw, bh = max(faces, key=lambda f: f[2] * f[3])
    return True, [int(x), int(y), int(bw), int(bh)]


class CLIPClassifier:
    def __init__(self):
        try:
            import open_clip
            import torch
        except ImportError as e:
            raise RuntimeError("open_clip/torch 미설치. 'uv add open-clip-torch torch' 후 재시도.") from e

        self.torch = torch
        device = (
            "mps"
            if torch.backends.mps.is_available()
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.device = device
        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="laion2b_s34b_b79k"
        )
        model = model.to(device).eval()
        self.model = model
        self.preprocess = preprocess
        self.tokenizer = open_clip.get_tokenizer("ViT-B-32")

        with torch.no_grad():
            self.content_text = self._encode_text(list(CONTENT_PROMPTS.values()))
            self.expression_text = self._encode_text(list(EXPRESSION_PROMPTS.values()))
        self.content_keys = list(CONTENT_PROMPTS.keys())
        self.expression_keys = list(EXPRESSION_PROMPTS.keys())

    def _encode_text(self, prompts):
        toks = self.tokenizer(prompts).to(self.device)
        feats = self.model.encode_text(toks)
        feats /= feats.norm(dim=-1, keepdim=True)
        return feats

    def _encode_image(self, rgb: np.ndarray):
        from PIL import Image

        img = Image.fromarray(rgb)
        x = self.preprocess(img).unsqueeze(0).to(self.device)
        with self.torch.no_grad():
            f = self.model.encode_image(x)
        f /= f.norm(dim=-1, keepdim=True)
        return f

    def classify(self, rgb: np.ndarray, has_face: bool) -> dict:
        feat = self._encode_image(rgb)
        with self.torch.no_grad():
            content_logits = (100.0 * feat @ self.content_text.T).softmax(dim=-1).cpu().numpy()[0]
        idx = int(content_logits.argmax())
        content_conf = float(content_logits[idx])
        content_type = self.content_keys[idx] if content_conf >= CONFIDENCE_THRESHOLD else "unknown"

        out = {"content_type": content_type, "content_conf": round(content_conf, 3)}

        if has_face:
            with self.torch.no_grad():
                expr_logits = (
                    (100.0 * feat @ self.expression_text.T).softmax(dim=-1).cpu().numpy()[0]
                )
            ei = int(expr_logits.argmax())
            ec = float(expr_logits[ei])
            if ec >= CONFIDENCE_THRESHOLD:
                out["expression"] = self.expression_keys[ei]
                out["expression_conf"] = round(ec, 3)
        return out


def main() -> None:
    parser = argparse.ArgumentParser(description="타일 분류 (얼굴 + CLIP + 색)")
    parser.add_argument("--tiles", type=Path, default=TILES_DIR)
    parser.add_argument("--out-csv", type=Path, default=INDEX_CSV)
    parser.add_argument("--out-meta", type=Path, default=META_JSON)
    parser.add_argument("--no-clip", action="store_true")
    parser.add_argument("--use-aesthetic", action="store_true")
    args = parser.parse_args()

    ensure_dirs()
    paths = iter_tile_paths(args.tiles)
    print(f"[classify] {len(paths)}장 처리 시작 (CLIP={'OFF' if args.no_clip else 'ON'})")

    clip_clf = None if args.no_clip else CLIPClassifier()

    rows = []
    meta: dict[str, dict] = {}

    for p in tqdm(paths):
        rgb = load_image_rgb(p)
        if rgb is None:
            continue

        # numeric features (also writes 256 normalized + thumb)
        row = process_tile(p, do_thumb=True)
        if row is None:
            continue

        # face detection
        has_face, bbox = _detect_face(rgb)
        row["has_face"] = 1 if has_face else 0
        row["face_x"], row["face_y"], row["face_w"], row["face_h"] = bbox

        rows.append(row)

        m: dict = {"source": source_of(p)}
        if clip_clf is not None:
            m.update(clip_clf.classify(rgb, has_face))
        else:
            ratio = rgb.shape[1] / max(1, rgb.shape[0])
            if has_face and ratio < 1.3:
                m["content_type"] = "face"
            elif ratio > 1.5:
                m["content_type"] = "scene"
            else:
                m["content_type"] = "unknown"
            m["content_conf"] = 0.0

        if args.use_aesthetic:
            m["aesthetic"] = 5.0  # placeholder

        meta[row["path"]] = m

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    args.out_meta.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_meta, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    # 분포 보고
    print(f"\n[classify] 인덱스 → {args.out_csv}")
    print(f"[classify] 메타  → {args.out_meta}")
    print("\n[classify] content_type 분포:")
    counter = Counter(v["content_type"] for v in meta.values())
    for k, n in counter.most_common():
        print(f"  {k}: {n}")

    expr = Counter(v.get("expression", "—") for v in meta.values() if v.get("expression"))
    if expr:
        print("\n[classify] expression 분포:")
        for k, n in expr.most_common():
            print(f"  {k}: {n}")

    src_counter = Counter(v["source"] for v in meta.values())
    print("\n[classify] source 분포:")
    for k, n in src_counter.most_common():
        print(f"  {k}: {n}")


if __name__ == "__main__":
    main()
