"""README 처리 과정 그림 — 입력 → 단계 → 결과를 상자와 화살표로(실측 결과가 아닌 동작 설명 그림).

사용:
    python docs/scripts/make_diagram.py <spec.json> <docs 폴더> <ko|en>
spec.json:
    {"title": {"ko": "...", "en": "..."},
     "steps": [{"head": {"ko": "...", "en": "..."}, "body": {"ko": "...", "en": "..."}}, ...],
     "footer": {"ko": "...", "en": "..."}, "out": "process.png"}
<docs>/<언어>/<out> 을 쓴다. 필요한 것: matplotlib · Pretendard 글꼴(Windows 사용자 글꼴 폴더 또는 README_FONT_DIR).
"""
import json
import os
import sys
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

FONT_DIR = os.environ.get("README_FONT_DIR") or os.path.expandvars(r"%LOCALAPPDATA%\Microsoft\Windows\Fonts")
BG, BOX, LINE, TEXT, MUTED, ACCENT = "#171717", "#232323", "#3a3a3a", "#ededed", "#a5a5a5", "#f97316"


def font(weight):
    for name in (f"Pretendard-{weight}.otf", f"Pretendard-{weight}.ttf"):
        path = os.path.join(FONT_DIR, name)
        if os.path.exists(path):
            font_manager.fontManager.addfont(path)
            return font_manager.FontProperties(fname=path)
    sys.exit(f"Pretendard-{weight} 글꼴이 없습니다: {FONT_DIR}")


def main():
    spec_path, docs, lang = sys.argv[1:4]
    spec = json.load(open(spec_path, encoding="utf-8"))
    bold, regular = font("Bold"), font("Regular")
    steps = spec["steps"]
    n = len(steps)
    w_in, h_in = 12.0, 4.6
    fig = plt.figure(figsize=(w_in, h_in), dpi=100, facecolor=BG)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, w_in)
    ax.set_ylim(0, h_in)
    ax.axis("off")
    ax.text(0.5, h_in - 0.55, spec["title"][lang], color=TEXT, fontproperties=bold, fontsize=19, va="center")
    gap, left, right = 0.42, 0.5, 0.5
    bw = (w_in - left - right - gap * (n - 1)) / n
    top, bh = h_in - 1.15, 2.55
    wrap = 15 if lang == "ko" else 24
    for i, st in enumerate(steps):
        x = left + i * (bw + gap)
        ax.add_patch(FancyBboxPatch((x, top - bh), bw, bh, boxstyle="round,pad=0,rounding_size=0.12", fc=BOX, ec=ACCENT if i in (0, n - 1) else LINE, lw=1.4))
        ax.text(x + 0.18, top - 0.32, st["head"][lang], color=ACCENT if i in (0, n - 1) else TEXT, fontproperties=bold, fontsize=13.5, va="top")
        body = "\n".join(textwrap.fill(p, wrap) for p in st["body"][lang].split("\n"))
        ax.text(x + 0.18, top - 0.82, body, color=MUTED, fontproperties=regular, fontsize=11.5, va="top", linespacing=1.45)
        if i < n - 1:
            y = top - bh / 2
            ax.add_patch(FancyArrowPatch((x + bw + 0.05, y), (x + bw + gap - 0.05, y), arrowstyle="-|>", mutation_scale=16, color=MUTED, lw=1.6))
    ax.text(0.5, 0.45, spec["footer"][lang], color=MUTED, fontproperties=regular, fontsize=11.5, va="center")
    out = os.path.join(docs, lang, spec.get("out", "process.png"))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, facecolor=BG)
    print(out, os.path.getsize(out))


if __name__ == "__main__":
    main()
