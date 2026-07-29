from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "docs" / "figures" / "qc_agent_schemes"

FONT_REG = Path(r"C:\Windows\Fonts\Noto Sans SC (TrueType).otf")
FONT_MED = Path(r"C:\Windows\Fonts\Noto Sans SC Medium (TrueType).otf")
FONT_BOLD = Path(r"C:\Windows\Fonts\Noto Sans SC Bold (TrueType).otf")
if not FONT_REG.exists():
    FONT_REG = Path(r"C:\Windows\Fonts\msyh.ttc")
if not FONT_MED.exists():
    FONT_MED = FONT_REG
if not FONT_BOLD.exists():
    FONT_BOLD = Path(r"C:\Windows\Fonts\msyhbd.ttc")


W, H = 1800, 1050
BG = "#FAFBFC"
INK = "#1E2A36"
MUTED = "#5B6876"
LINE = "#D8DEE6"
BLUE = "#2F6BFF"
BLUE_L = "#EAF0FF"
GREEN = "#1F9D68"
GREEN_L = "#EAF7F1"
ORANGE = "#D9822B"
ORANGE_L = "#FFF3E4"
RED = "#C44747"
RED_L = "#FCECEC"
PURPLE = "#7A5CCF"
PURPLE_L = "#F0ECFF"
TEAL = "#157C8A"
TEAL_L = "#E8F6F8"


def font(size: int, weight: str = "regular") -> ImageFont.FreeTypeFont:
    file = FONT_REG
    if weight == "medium":
        file = FONT_MED
    elif weight == "bold":
        file = FONT_BOLD
    return ImageFont.truetype(str(file), size=size)


def text_size(draw: ImageDraw.ImageDraw, text: str, fnt: ImageFont.FreeTypeFont) -> tuple[int, int]:
    box = draw.textbbox((0, 0), text, font=fnt)
    return box[2] - box[0], box[3] - box[1]


def wrap_text(draw: ImageDraw.ImageDraw, text: str, fnt: ImageFont.FreeTypeFont, max_width: int) -> list[str]:
    lines: list[str] = []
    current = ""
    for ch in text:
        if ch == "\n":
            if current:
                lines.append(current)
                current = ""
            continue
        trial = current + ch
        if text_size(draw, trial, fnt)[0] <= max_width:
            current = trial
        else:
            if current:
                lines.append(current)
            current = ch
    if current:
        lines.append(current)
    return lines


def draw_text_block(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    fnt: ImageFont.FreeTypeFont,
    fill: str,
    max_width: int,
    line_gap: int = 8,
) -> int:
    x, y = xy
    for line in wrap_text(draw, text, fnt, max_width):
        draw.text((x, y), line, fill=fill, font=fnt)
        y += text_size(draw, line, fnt)[1] + line_gap
    return y


def rounded(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int, int, int],
    fill: str,
    outline: str = LINE,
    width: int = 2,
    radius: int = 18,
) -> None:
    draw.rounded_rectangle(xy, radius=radius, fill=fill, outline=outline, width=width)


def card(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int, int, int],
    title: str,
    body: str,
    accent: str,
    fill: str,
    icon: str | None = None,
) -> None:
    x1, y1, x2, y2 = xy
    rounded(draw, xy, fill=fill, outline=accent, width=2, radius=20)
    draw.rectangle((x1, y1, x1 + 10, y2), fill=accent)
    tx = x1 + 34
    ty = y1 + 28
    if icon:
        draw.text((tx, ty - 2), icon, fill=accent, font=font(28, "bold"))
        tx += 48
    draw.text((tx, ty), title, fill=INK, font=font(34, "bold"))
    draw_text_block(draw, (x1 + 34, y1 + 88), body, font(25), MUTED, x2 - x1 - 68, line_gap=9)


def compact_card(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int, int, int],
    title: str,
    body: str,
    accent: str,
    fill: str,
) -> None:
    x1, y1, x2, y2 = xy
    rounded(draw, xy, fill=fill, outline=accent, width=2, radius=18)
    draw.rectangle((x1, y1, x1 + 9, y2), fill=accent)
    draw.text((x1 + 28, y1 + 22), title, fill=INK, font=font(30, "bold"))
    draw_text_block(draw, (x1 + 28, y1 + 72), body, font(21), MUTED, x2 - x1 - 56, line_gap=6)


def arrow(draw: ImageDraw.ImageDraw, start: tuple[int, int], end: tuple[int, int], color: str = "#9AA5B1", width: int = 5) -> None:
    draw.line((start, end), fill=color, width=width)
    sx, sy = start
    ex, ey = end
    if abs(ex - sx) >= abs(ey - sy):
        direction = 1 if ex >= sx else -1
        pts = [(ex, ey), (ex - 18 * direction, ey - 12), (ex - 18 * direction, ey + 12)]
    else:
        direction = 1 if ey >= sy else -1
        pts = [(ex, ey), (ex - 12, ey - 18 * direction), (ex + 12, ey - 18 * direction)]
    draw.polygon(pts, fill=color)


def header(draw: ImageDraw.ImageDraw, title: str, subtitle: str) -> None:
    draw.text((80, 54), title, fill=INK, font=font(46, "bold"))
    draw_text_block(draw, (80, 118), subtitle, font(25), MUTED, 1250, line_gap=7)


def chip(draw: ImageDraw.ImageDraw, xy: tuple[int, int, int, int], text: str, fill: str, outline: str) -> None:
    rounded(draw, xy, fill=fill, outline=outline, width=2, radius=24)
    tw, th = text_size(draw, text, font(24, "medium"))
    x1, y1, x2, y2 = xy
    draw.text((x1 + (x2 - x1 - tw) / 2, y1 + (y2 - y1 - th) / 2 - 2), text, fill=outline, font=font(24, "medium"))


def save(name: str, draw_fn) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(img)
    draw_fn(draw)
    path = OUT_DIR / name
    img.save(path, "PNG", optimize=True)
    return path


def fig_overall(draw: ImageDraw.ImageDraw) -> None:
    header(draw, "质检 Agent 总流程", "先把两批数据接进同一套检查口径，再按风险决定是否进入 VLM 和人工复核。")

    card(draw, (80, 220, 430, 390), "YOLO 批", "原始帧 + 人体 crop\n已完成资源层检查", BLUE, BLUE_L)
    card(draw, (80, 440, 430, 620), "VLM 生成批", "metadata + crop + mask\n发现一个断链样本", TEAL, TEAL_L)

    levels = [
        ("L0 资源", "文件存在、可读、目录成套", GREEN, GREEN_L),
        ("L1 字段", "metadata、schema、枚举值", GREEN, GREEN_L),
        ("L2 几何", "bbox、crop、mask 是否对齐", ORANGE, ORANGE_L),
        ("L3 语义", "目标可见、标签匹配、生成真实感", PURPLE, PURPLE_L),
        ("L4 人工", "高风险样本和随机抽检", RED, RED_L),
    ]
    x = 520
    y = 205
    for i, (t, b, c, f) in enumerate(levels):
        compact_card(draw, (x, y + i * 128, x + 450, y + 112 + i * 128), t, b, c, f)
        if i < len(levels) - 1:
            arrow(draw, (x + 225, y + 114 + i * 128), (x + 225, y + 126 + i * 128), "#A8B2BE", 3)

    outputs = [
        ("通过样本", "进入训练或 Label Studio", GREEN, GREEN_L),
        ("复核队列", "needs_human_review", ORANGE, ORANGE_L),
        ("质检报告", "field_path + evidence_uri", BLUE, BLUE_L),
    ]
    for i, (t, b, c, f) in enumerate(outputs):
        card(draw, (1110, 250 + i * 190, 1680, 410 + i * 190), t, b, c, f)
    arrow(draw, (970, 330), (1110, 330), "#A8B2BE", 4)
    arrow(draw, (970, 520), (1110, 520), "#A8B2BE", 4)
    arrow(draw, (970, 710), (1110, 710), "#A8B2BE", 4)

    draw.line((180, 860, 1590, 860), fill="#A8B2BE", width=4)
    for x in (240, 590, 940, 1290):
        draw.ellipse((x - 8, 852, x + 8, 868), fill=BLUE)
    draw.text((180, 892), "汇报指标：规则通过率    成套完整率    VLM 语义通过率    人工抽检错误率", fill=INK, font=font(30, "medium"))
    draw.text((650, 970), "字段穿透 / 证据追溯", fill=MUTED, font=font(28, "medium"))


def fig_manual(draw: ImageDraw.ImageDraw) -> None:
    header(draw, "方案一：全人工复核", "人工逐条看框和生成图，适合做第一批校准样本，不适合长期全量跑。")
    xs = [90, 500, 910, 1320]
    titles = ["输入样本", "人工检查", "人工结论", "金标样本"]
    bodies = ["YOLO 框\nVLM 生成图\n高风险样本", "看目标是否存在\n看标签是否匹配\n看异常是否真实", "通过\n退回修改\n进入复核队列", "用于校准规则\n用于校准 VLM\n用于估算残余错误"]
    colors = [(BLUE, BLUE_L), (ORANGE, ORANGE_L), (GREEN, GREEN_L), (PURPLE, PURPLE_L)]
    for i, x in enumerate(xs):
        card(draw, (x, 290, x + 320, 550), titles[i], bodies[i], colors[i][0], colors[i][1])
        if i < 3:
            arrow(draw, (x + 320, 420), (xs[i + 1] - 20, 420))
    chip(draw, (150, 715, 500, 790), "准确性高", GREEN_L, GREEN)
    chip(draw, (580, 715, 930, 790), "人工成本高", RED_L, RED)
    chip(draw, (1010, 715, 1450, 790), "只建议做校准", ORANGE_L, ORANGE)
    draw_text_block(draw, (145, 860), "汇报口径：第一批需要人工看一部分样本，把标准定清楚。等标准稳定后，人工只看高风险队列和抽检样本。", font(30, "medium"), INK, 1450, 10)


def fig_rules(draw: ImageDraw.ImageDraw) -> None:
    header(draw, "方案二：规则质检", "所有样本先过规则层，把文件、字段、几何这类硬错误快速拦下来。")
    steps = [
        ("资源检查", "图片 / crop / mask\n存在且可读", BLUE, BLUE_L),
        ("字段检查", "metadata\n必填字段\n枚举值", TEAL, TEAL_L),
        ("几何检查", "bbox 越界\n面积异常\ncrop 覆盖", ORANGE, ORANGE_L),
        ("结果分流", "通过\n失败\n进入复核", GREEN, GREEN_L),
    ]
    for i, (t, b, c, f) in enumerate(steps):
        x = 120 + i * 410
        card(draw, (x, 300, x + 330, 560), t, b, c, f)
        if i < len(steps) - 1:
            arrow(draw, (x + 330, 430), (x + 390, 430))
    chip(draw, (160, 710, 480, 780), "全量必跑", BLUE_L, BLUE)
    chip(draw, (560, 710, 880, 780), "秒级完成", GREEN_L, GREEN)
    chip(draw, (960, 710, 1360, 780), "看不懂语义", ORANGE_L, ORANGE)
    draw_text_block(draw, (145, 860), "汇报口径：规则层不负责判断画面语义，但它能稳定拦住文件缺失、字段缺失、bbox 越界这些确定性问题。", font(30, "medium"), INK, 1450, 10)


def fig_vlm(draw: ImageDraw.ImageDraw) -> None:
    header(draw, "方案三：VLM 语义复核", "把规则看不出来的样本交给 H20/VLM，看目标、标签和生成效果是否对得上。")
    card(draw, (100, 280, 420, 540), "风险样本", "规则 warning\n小目标或贴边\n标签容易混淆", ORANGE, ORANGE_L)
    card(draw, (550, 240, 930, 580), "输入包", "原图\n带框图\n裁剪图\n标签\n巡检内容", BLUE, BLUE_L)
    card(draw, (1060, 280, 1380, 540), "H20/VLM", "结构化判断\n原因说明\n是否进人工", PURPLE, PURPLE_L)
    card(draw, (1500, 280, 1700, 540), "分流", "通过\n人工\n退回", GREEN, GREEN_L)
    arrow(draw, (420, 410), (550, 410))
    arrow(draw, (930, 410), (1060, 410))
    arrow(draw, (1380, 410), (1500, 410))
    chip(draw, (170, 705, 520, 775), "能看语义", PURPLE_L, PURPLE)
    chip(draw, (605, 705, 955, 775), "要记录延迟", BLUE_L, BLUE)
    chip(draw, (1040, 705, 1510, 775), "先做 pilot 再扩量", ORANGE_L, ORANGE)
    draw_text_block(draw, (145, 860), "汇报口径：VLM 不直接替代全部人工。第一步先抽 20 条做 pilot，看误判率、解析成功率和 p95 延迟。", font(30, "medium"), INK, 1450, 10)


def fig_model_check(draw: ImageDraw.ImageDraw) -> None:
    header(draw, "方案四：模型一致性反检", "用另一套模型重新看一遍，再和原始标注对比，主要服务 YOLO 批。")
    card(draw, (110, 250, 450, 520), "原始标注", "YOLO bbox\nconfidence\nclass", BLUE, BLUE_L)
    card(draw, (110, 610, 450, 880), "复检模型", "二次检测\n二次分类\n可换模型版本", TEAL, TEAL_L)
    card(draw, (680, 395, 1040, 735), "对比器", "IoU 对齐\n类别比较\n置信度比较", ORANGE, ORANGE_L)
    card(draw, (1260, 270, 1660, 850), "冲突类型", "漏框\n重复框\n明显偏框\n分类冲突\n背景误检", RED, RED_L)
    arrow(draw, (450, 385), (680, 500))
    arrow(draw, (450, 745), (680, 640))
    arrow(draw, (1040, 565), (1260, 565))
    draw_text_block(draw, (145, 925), "汇报口径：这条路有价值，但要等 YOLO bbox 坐标和一批人工金标补齐后再做。", font(30, "medium"), INK, 1450, 10)


def fig_tools(draw: ImageDraw.ImageDraw) -> None:
    header(draw, "方案五：数据质量工具", "借鉴成熟工具的能力，用来发现重复图、模糊图、离群图和格式问题。")
    tools = [
        ("CleanVision", "模糊、低信息量、异常尺寸", BLUE, BLUE_L),
        ("fastdup", "重复图、近重复图、离群图", GREEN, GREEN_L),
        ("FiftyOne", "可视化复核、问题样本排序", PURPLE, PURPLE_L),
        ("Datumaro", "格式转换、数据校验、导出", TEAL, TEAL_L),
    ]
    for i, (t, b, c, f) in enumerate(tools):
        row = i // 2
        col = i % 2
        x = 190 + col * 760
        y = 260 + row * 280
        card(draw, (x, y, x + 610, y + 210), t, b, c, f)
    card(draw, (570, 800, 1230, 955), "接入方式", "先做旁路分析\n需要时接入质检 Agent", ORANGE, ORANGE_L)
    draw_text_block(draw, (145, 965), "汇报口径：这些工具先作为补充能力，不急着上完整平台，避免把第一版做复杂。", font(27, "medium"), INK, 1450, 8)


def fig_hybrid(draw: ImageDraw.ImageDraw) -> None:
    header(draw, "方案六：混合分层质检", "这是当前最推荐的路线：规则跑全量，VLM 看风险样本，人工做校准。")
    card(draw, (90, 290, 360, 530), "全部样本", "YOLO 批\nVLM 生成批", BLUE, BLUE_L)
    card(draw, (500, 210, 820, 450), "规则层", "文件\n字段\n几何", GREEN, GREEN_L)
    card(draw, (500, 590, 820, 830), "通过样本", "进入训练\n或 Label Studio", GREEN, GREEN_L)
    card(draw, (990, 210, 1320, 450), "风险样本", "warning\n语义不确定\n随机抽检", ORANGE, ORANGE_L)
    card(draw, (990, 590, 1320, 830), "VLM 复核", "目标可见\n标签匹配\n生成真实感", PURPLE, PURPLE_L)
    card(draw, (1470, 390, 1720, 650), "人工队列", "高风险\nVLM 拿不准\n抽检", RED, RED_L)
    arrow(draw, (360, 410), (500, 330))
    arrow(draw, (660, 450), (660, 590))
    arrow(draw, (820, 330), (990, 330))
    arrow(draw, (1155, 450), (1155, 590))
    arrow(draw, (1320, 710), (1470, 520))
    rounded(draw, (845, 805, 1440, 865), fill="#FFF9EC", outline="#B7A06A", width=2, radius=18)
    draw.text((900, 818), "人工结论用于更新规则和 prompt", fill=MUTED, font=font(25, "medium"))
    chip(draw, (170, 900, 500, 970), "人工量可控", GREEN_L, GREEN)
    chip(draw, (580, 900, 930, 970), "能追字段证据", BLUE_L, BLUE)
    chip(draw, (1010, 900, 1480, 970), "适合当前第一版推进", ORANGE_L, ORANGE)


def main() -> None:
    figures = {
        "fig_01_overall_flow.png": fig_overall,
        "fig_02_manual_review.png": fig_manual,
        "fig_03_rule_qc.png": fig_rules,
        "fig_04_vlm_review.png": fig_vlm,
        "fig_05_model_consistency.png": fig_model_check,
        "fig_06_data_quality_tools.png": fig_tools,
        "fig_07_hybrid_layered_qc.png": fig_hybrid,
    }
    for name, fn in figures.items():
        path = save(name, fn)
        print(path)


if __name__ == "__main__":
    main()
