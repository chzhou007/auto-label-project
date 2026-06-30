from __future__ import annotations

ANOMALY_SELECTION_HINTS = {
    "diesel_leak": "fuel pipe, yellow metal supply pipe, fuel flange, valve joint, generator lower side, equipment-floor boundary",
    "oil_leak": "engine block, bolt flange, sealing cover edge, oil cooler connection, middle or upper-middle metal connection structures",
    "coolant_leak": "black rubber hose, silver clamp, radiator connection pipe, cooling circuit hose joint, engine-radiator connection, middle-lower hose area",
    "water_leak": (
        "equipment lower side, base edge, local floor contact area, drain path, water pipe, condensate pipe, drain pipe, "
        "pipe joint, valve, metal connector, or any plausible small floor area where a clear early wet trace can appear"
    ),
    "water_leakage": (
        "equipment lower side, base edge, local floor contact area, drain path, water pipe, condensate pipe, drain pipe, "
        "pipe joint, valve, metal connector, or any plausible small floor area where a clear early wet trace can appear"
    ),
}

ANOMALY_WAN_APPEARANCE = {
    "diesel_leak": "clear transparent light yellow or amber liquid, low viscosity, water-like flow, thin reflective wet film on gray epoxy floor",
    "oil_leak": "deep black or black-brown, opaque, high viscosity, poor flowability, local sticky oil stain attached to engine block, flange, or seal area",
    "coolant_leak": "silver-green or light silver-green transparent liquid, clear and bright but not sci-fi fluorescent, low viscosity, reflective green liquid film",
    "water_leak": (
        "extremely small early clear-water wet trace: nearly colorless or slightly whitish transparent thin film, subtle bright highlights, "
        "low viscosity, local and controlled, no yellow diesel tint, no black oil, no silver-green coolant, no fluorescent color"
    ),
    "water_leakage": (
        "extremely small early clear-water wet trace: nearly colorless or slightly whitish transparent thin film, subtle bright highlights, "
        "low viscosity, local and controlled, no yellow diesel tint, no black oil, no silver-green coolant, no fluorescent color"
    ),
}

SEVERITY_HINTS = {
    "early": "very small early-stage trace, subtle but visible, controlled scale",
    "moderate": "moderate local leakage, visible but still controlled and localized",
    "obvious_but_controlled": "obvious localized leakage, still controlled, no disaster scene, no broad spill",
}


def _excluded_grid_instruction(excluded_grids: list[str] | None, chinese: bool = False) -> str:
    if not excluded_grids:
        return ""
    grids = ", ".join(excluded_grids)
    if chinese:
        return (
            f"\n\n硬性去敏要求：禁止选择这些网格：{grids}。这些网格包含或贴近监控画面左上角日期时间、"
            "右下角地点/机房标识等敏感叠加文字。candidate_grids、selected_grid 和备选理由都必须避开这些网格。"
        )
    return (
        f"\n\nHard privacy rule: never select these grid cells: {grids}. They contain or touch CCTV timestamp/site-label overlays. "
        "selected_grid and candidate_grids must exclude them."
    )


def build_qwen_grid_prompt(anomaly_type: str, candidate_count: int = 3, excluded_grids: list[str] | None = None) -> str:
    candidate_count = max(1, min(16, int(candidate_count)))
    excluded_instruction_zh = _excluded_grid_instruction(excluded_grids, chinese=True)
    excluded_instruction_en = _excluded_grid_instruction(excluded_grids, chinese=False)
    if anomaly_type in {"water_leak", "water_leakage"}:
        return """
你是一名工业巡检图像分析专家。现在给你一张已经叠加 4x4 网格编号的工业机房监控图像，网格编号从左到右、从上到下依次为：

A1 A2 A3 A4
B1 B2 B3 B4
C1 C2 C3 C4
D1 D2 D3 D4

你的任务是判断哪一个网格区域最适合进行“漏水 / 清水泄漏”异常图像编辑。

漏水异常的合理发生位置通常包括：
1. 设备下部、底座边缘、设备与地面交界处、排水路径或局部地面接触区域；
2. 靠近水管、冷凝水管、排水管、接口、阀门、金属接头或软管连接的位置；
3. 机组底部附近、地面低洼处或能自然出现一小片透明水膜/轻微湿痕的位置；
4. 可以只有裸露水渍，例如灰色环氧地坪上一小片浅薄透明水膜、局部湿痕或轻微高光反射；
5. 不需要制造明确墙面漏水点、管道破口、喷涌源头或可见水流线，只要位置合理且像早期清水泄漏即可。

漏水异常不应发生在：
1. 天花板、远处墙面、纸箱、蓝色托盘等无关背景区域；
2. 完全远离设备、管路、接头和底座的普通地面中央；
3. 发电机顶部排气管、空气滤清器等不适合产生清水泄漏的位置；
4. 没有任何设备结构、管路结构、底座边缘或合理湿痕语境的区域。

请优先选择能够出现“极少量、早期、受控、局部透明清水水渍”的网格。水渍可以没有明显源头，但不能像黄色柴油、黑褐色机油、银绿色防冻液、荧光液体，也不能是大面积积水、喷涌、水柱、爆管、墙面渗漏、天花板漏水、霉斑或水流线。
%s

请严格输出 JSON，不要输出额外解释文字。必须输出 top-%d candidate_grids；如果不确定，也必须给出 %d 个不重复网格。

输出格式：
{
  "selected_grid": "C2",
  "confidence": 0.86,
  "candidate_grids": [
    {
      "grid": "C2",
      "score": 0.86,
      "reason": "该区域靠近设备下部、管路接头或底座边缘，并且下方有地面可形成清水湿痕，适合生成漏水异常。"
    },
    {
      "grid": "C3",
      "score": 0.73,
      "reason": "该区域可能包含底座边缘和地面扩散区域，可作为清水流动或湿痕扩散的备选区域。"
    },
    {
      "grid": "B2",
      "score": 0.62,
      "reason": "该区域包含部分设备结构，但与地面湿痕形成路径不如首选区域明确。"
    }
  ],
  "edit_region_hint": "在所选网格内生成极少量早期清水泄漏痕迹，可以是一小片裸露透明水渍、浅薄水膜、局部湿痕或轻微高光反射，不需要明确漏水点。",
  "risk_note": "漏水应表现为透明、微白反光、局部、受控、极少量清水湿痕；禁止黄色柴油、黑褐机油、银绿色防冻液、荧光液体、大面积积水、喷涌、水柱、爆管、墙面渗漏、天花板漏水、霉斑和水流线。",
  "rejected_region_reason": "避开天花板、远处墙面、纸箱、托盘、普通空地、无管路或无设备连接结构的区域。"
}
""" % (excluded_instruction_zh, candidate_count, candidate_count)
    hint = ANOMALY_SELECTION_HINTS[anomaly_type]
    return f"""
You are an industrial inspection image analysis expert. You receive a clean industrial image preview with a visible 4x4 grid and grid IDs A1-D4.
Select the top-{candidate_count} coarse grid cells most suitable for realistic {anomaly_type} image editing.
Prefer realistic industrial anomaly locations: {hint}.
Do not select wall, ceiling, paper boxes, pallets, ordinary empty floor, or unrelated background unless no better region exists.
{excluded_instruction_en}
Return strict JSON only, with no markdown and no extra text. The JSON schema is:
{{
  "selected_grid": "C2",
  "confidence": 0.86,
  "candidate_grids": [
    {{"grid": "C2", "score": 0.86, "reason": "..."}},
    {{"grid": "C3", "score": 0.74, "reason": "..."}},
    {{"grid": "B2", "score": 0.62, "reason": "..."}}
  ],
  "edit_region_hint": "...",
  "risk_note": "...",
  "rejected_region_reason": "..."
}}
Always output exactly {candidate_count} unique candidate_grids when possible. Scores must be between 0 and 1.
""".strip()


def build_qwen_fine_grid_prompt(anomaly_type: str, coarse_grid: str) -> str:
    if anomaly_type in {"water_leak", "water_leakage"}:
        return f"""
你正在对粗网格 {coarse_grid} 内的“漏水 / 清水泄漏”编辑区域做 3x3 局部细选。局部网格编号为 A1-C3。
请选择最适合生成微量清水泄漏的位置，优先靠近水管、冷凝水管、排水管、阀门、接头、设备底部、底座边缘或排水路径。
细选区域最好同时包含可能漏水起点，以及清水向下流动或在灰色环氧地坪形成纯白透明高亮湿痕的空间。
不要选择墙面、天花板、普通空地中央、纸箱、托盘、无设备结构或无液体来源的位置。
请严格输出 JSON，不要输出额外解释文字，格式为：
{{
  "selected_grid": "B2",
  "confidence": 0.82,
  "candidate_grids": [
    {{"grid": "B2", "score": 0.82, "reason": "靠近设备底部或管路接头，且下方有清水湿痕扩散空间。"}},
    {{"grid": "B1", "score": 0.68, "reason": "包含部分可能漏水起点，但地面扩散路径较弱。"}},
    {{"grid": "C2", "score": 0.61, "reason": "适合形成地面清水湿痕，但漏水起点不如首选明确。"}}
  ],
  "edit_region_hint": "在局部网格内靠近接头、阀门、底座边缘或排水路径处生成纯白透明偏高亮的清水水迹。",
  "risk_note": "不要生成黄色柴油、绿色防冻液、黑褐色机油、荧光液体、高压喷水或大面积洪水。",
  "rejected_region_reason": "避开无管路、无接头、无设备底部结构、无法形成清水滴落路径的位置。"
}}
""".strip()
    hint = ANOMALY_SELECTION_HINTS[anomaly_type]
    return f"""
You are refining a coarse grid selection for {anomaly_type}. The image is a crop from coarse grid {coarse_grid} with a local 3x3 grid A1-C3.
Select the best local fine grid for a small realistic anomaly near: {hint}.
Avoid plain wall, ceiling, unrelated empty floor, boxes, labels, and visual clutter.
Return strict JSON only with this schema:
{{
  "selected_grid": "B2",
  "confidence": 0.82,
  "candidate_grids": [
    {{"grid": "B2", "score": 0.82, "reason": "..."}},
    {{"grid": "B1", "score": 0.68, "reason": "..."}},
    {{"grid": "C2", "score": 0.61, "reason": "..."}}
  ],
  "edit_region_hint": "...",
  "risk_note": "...",
  "rejected_region_reason": "..."
}}
""".strip()


def build_wan_edit_prompt(anomaly_type: str, edit_region_hint: str, severity_level: str = "early") -> str:
    if anomaly_type in {"water_leak", "water_leakage"}:
        severity = SEVERITY_HINTS.get(severity_level, SEVERITY_HINTS["early"])
        return f"""
请基于输入的原始工业监控图像进行局部图像编辑，只在指定的 bbox_list 区域附近生成“漏水 / 清水泄漏”异常，其余区域必须严格保持与原图一致。

整体画面必须保持原图的固定 CCTV 监控视角、工业机房布局、设备位置、背景物体、照明条件、监控画面质感和空间关系。不得改变主体设备结构，不得改变机房内原有设备和背景元素，不得新增人物、车辆、工具、文字、箭头、红框、标注框、网格线或网格编号。不要生成高压喷涌、大规模爆裂漏水、洪水、烟雾、火焰、爆炸、蒸汽或任何灾难化效果。

异常类型为漏水 / 清水泄漏。当前严重程度为：{severity}。

请在指定 bbox_list 内生成极少量、早期、受控、局部的清水泄漏痕迹。区域提示：{edit_region_hint or '设备底部、底座边缘、排水路径、局部地面或靠近水管/冷凝水管/排水管的位置'}。不需要制造明确的墙面漏水点、管道破口、接头喷涌或设备上的可见漏水源头；可以只在合理位置生成一小片裸露透明水渍、浅薄水膜、轻微湿痕或局部高光反射。

泄漏出的液体为清水。水量必须很少，范围必须局部、受控、边缘自然。水应接近无色透明或微白反光，主要通过环氧地坪或设备表面的轻微高光反射体现。不是黄色柴油、不是琥珀色油液、不是黑褐色机油、不是银绿色防冻液、不是荧光绿色或彩色液体。水的粘度极低，不能有油腻、粘稠、浑浊、泡沫、泥污或彩虹油膜。

水渍可以位于灰色环氧地坪、设备底部附近、底座边缘或排水路径附近。它应像非常薄的一小片透明水膜或局部湿痕，边缘不规则但面积很小，只轻微改变地面反光一致性。不要生成明显长水流线、墙面渗水痕、天花板漏水、霉斑、水柱、喷溅、爆管、大面积积水或洪水。

最终结果应突出“极少量、早期、透明清水、局部薄水膜、轻微湿痕、高光反射”的工业异常特征，而不是事故现场。异常应微妙但可见。

请确保异常只出现在指定 bbox_list 区域内部或其非常邻近位置。除漏水水迹、水滴和地面湿痕之外，所有设备、背景、警戒线、地面纹理、时间戳和机房环境必须尽量保持与原图一致。
""".strip()
    appearance = ANOMALY_WAN_APPEARANCE[anomaly_type]
    severity = SEVERITY_HINTS.get(severity_level, SEVERITY_HINTS["early"])
    return f"""
Edit only near bbox_list. Keep the original CCTV view, device layout, background, lighting, timestamp, and camera perspective unchanged.
Generate a realistic {severity} {anomaly_type}. Physical appearance: {appearance}.
Preferred local context: {edit_region_hint or 'industrial pipe, flange, hose, valve, seal, or equipment-floor boundary'}.
Do not change non-anomaly regions. Do not add people, vehicles, arrows, labels, red/yellow boxes, grid lines, grid IDs, masks, or text.
Do not generate fire, smoke, explosion, vapor, spark, high-pressure spray, large splash, or disaster scene.
The anomaly must be localized around bbox_list and must not affect unrelated background.
""".strip()


def build_negative_prompt() -> str:
    return (
        "people, vehicles, red box, yellow box, bounding box, arrow, label, text, grid line, grid id, mask overlay, "
        "segmentation color, fire, smoke, explosion, vapor, spark, high-pressure spray, large splash, disaster scene, "
        "changed camera angle, changed machine layout, rewritten background, unrealistic liquid, sci-fi glow, "
        "yellow oil liquid, light yellow diesel, amber oil, silver-green coolant, green liquid, black engine oil, "
        "black-brown oil stain, fluorescent liquid, sci-fi glowing liquid, rainbow oil film, large-scale flood, "
        "large puddle, flooding, high-pressure water spray, water jet, pipe burst, spray, wall leak point, ceiling leak, "
        "mold, wall seepage stain, stain trail on wall, long water flow line, foam, muddy water, turbid sewage, detection box, text label, "
        "黄色油液，淡黄色柴油，琥珀色油液，银绿色防冻液，绿色液体，黑色机油，黑褐色油污，荧光液体，"
        "科幻发光液体，彩虹油膜，大面积积水，大面积洪水，高压喷水，水柱喷射，喷溅，管道爆裂，"
        "墙面漏水点，天花板漏水，霉斑，墙面渗水痕，明显水流线，泡沫，泥水，浑浊污水，"
        "红框，检测框，箭头，网格线，网格编号，文字标签"
    )


def build_retry_visibility_prompt(anomaly_type: str) -> str:
    return (
        f"Make the {anomaly_type} slightly more visible while keeping it early-stage, controlled, local, and realistic. "
        "Do not expand outside bbox_list and do not alter the background."
    )


def build_qwen_review_prompt(anomaly_type: str) -> str:
    return f"""
You are a strict industrial anomaly quality reviewer. Check whether the image contains a realistic, early-stage, controlled {anomaly_type}.
Verify: anomaly type match, reasonable industrial location, no grid lines, no red/yellow boxes, no text, no arrows, no masks, and no severe disaster artifacts.
Return strict JSON only:
{{
  "is_valid": true,
  "anomaly_type_match": true,
  "location_reasonable": true,
  "visual_artifact": false,
  "score": 0.87,
  "reason": "..."
}}
""".strip()
