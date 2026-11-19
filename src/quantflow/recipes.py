"""量化 recipe 工厂：根据 (algorithm, scheme, position) 构建可复现的修饰器列表。

设计要点：
- 位置（position）通过显式正则 targets 精确圈定，避免误量化；
- transform 修饰器（SmoothQuant/AWQ）传入显式 mappings，使其只作用于目标位置；
- 算法-方案做合法性校验，与 configs/experiment_matrix.yaml 保持一致。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from llmcompressor.modifiers.gptq import GPTQModifier
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.modifiers.transform.awq import AWQModifier, AWQMapping
from llmcompressor.modifiers.transform.smoothquant import SmoothQuantModifier

from quantflow.logging_utils import get_logger

logger = get_logger(__name__)

# 算法-方案合法搭配（与 experiment_matrix.yaml axes 一致）
SUPPORTED_SCHEMES: dict[str, tuple[str, ...]] = {
    "smoothquant": ("W8A8",),
    "awq": ("W4A16", "W4A16_ASYM"),
}

# -----------------------------------------------------------------------------
# 位置定义：量化目标正则 + 各 transform 的显式映射
# -----------------------------------------------------------------------------

# 各位置需要量化的 Linear（正则匹配模块名）
POSITION_TARGETS: dict[str, list[str]] = {
    "all": ["Linear"],
    "attention": [r"re:.*self_attn\.(q_proj|k_proj|v_proj|o_proj)$"],
    "mlp": [r"re:.*mlp\.(gate_proj|up_proj|down_proj)$"],
}

# 各位置是否需要忽略 lm_head（仅 all 时需要，其余正则已天然排除）
POSITION_IGNORE: dict[str, list[str]] = {
    "all": ["lm_head"],
    "attention": [],
    "mlp": [],
}

# SmoothQuant 显式 mappings：结构为 [[balance_layers...], smooth_layer]
SMOOTHQUANT_MAPPINGS: dict[str, list[list[Any]]] = {
    "all": [
        [["re:.*q_proj", "re:.*k_proj", "re:.*v_proj"], "re:.*input_layernorm"],
        [["re:.*gate_proj", "re:.*up_proj"], "re:.*post_attention_layernorm"],
    ],
    "attention": [
        [["re:.*q_proj", "re:.*k_proj", "re:.*v_proj"], "re:.*input_layernorm"],
    ],
    "mlp": [
        [["re:.*gate_proj", "re:.*up_proj"], "re:.*post_attention_layernorm"],
    ],
}

# AWQ 显式 mappings（AWQMapping: smooth_layer + balance_layers）
_AWQ_ATTN_IN = AWQMapping(
    smooth_layer="re:.*input_layernorm$",
    balance_layers=["re:.*q_proj$", "re:.*k_proj$", "re:.*v_proj$"],
)
_AWQ_ATTN_OUT = AWQMapping(
    smooth_layer="re:.*v_proj$", balance_layers=["re:.*o_proj$"]
)
_AWQ_MLP_IN = AWQMapping(
    smooth_layer="re:.*post_attention_layernorm$",
    balance_layers=["re:.*gate_proj$", "re:.*up_proj$"],
)
_AWQ_MLP_OUT = AWQMapping(
    smooth_layer="re:.*up_proj$", balance_layers=["re:.*down_proj$"]
)

AWQ_MAPPINGS: dict[str, list[AWQMapping]] = {
    "all": [_AWQ_ATTN_IN, _AWQ_ATTN_OUT, _AWQ_MLP_IN, _AWQ_MLP_OUT],
    "attention": [_AWQ_ATTN_IN, _AWQ_ATTN_OUT],
    "mlp": [_AWQ_MLP_IN, _AWQ_MLP_OUT],
}


@dataclass(frozen=True)
class RecipeSpec:
    algorithm: str
    scheme: str
    position: str


def validate(spec: RecipeSpec) -> None:
    if spec.algorithm not in SUPPORTED_SCHEMES:
        raise ValueError(f"不支持的量化算法: {spec.algorithm}")
    if spec.scheme not in SUPPORTED_SCHEMES[spec.algorithm]:
        raise ValueError(
            f"算法 {spec.algorithm} 不支持方案 {spec.scheme}，"
            f"合法方案: {SUPPORTED_SCHEMES[spec.algorithm]}"
        )
    if spec.position not in POSITION_TARGETS:
        raise ValueError(f"不支持的量化位置: {spec.position}")


def build_recipe(
    algorithm: str,
    scheme: str,
    position: str,
    smoothquant_strength: float = 0.8,
    awq_duo_scaling: str = "both",
) -> list[Any]:
    """构建量化 recipe（修饰器列表），可直接传入 llmcompressor.oneshot。"""
    spec = RecipeSpec(algorithm, scheme, position)
    validate(spec)

    targets = POSITION_TARGETS[position]
    ignore = POSITION_IGNORE[position]

    if algorithm == "smoothquant":
        recipe = [
            SmoothQuantModifier(
                smoothing_strength=smoothquant_strength,
                mappings=SMOOTHQUANT_MAPPINGS[position],
            ),
            GPTQModifier(targets=targets, scheme=scheme, ignore=ignore),
        ]
    elif algorithm == "awq":
        recipe = [
            AWQModifier(
                duo_scaling=awq_duo_scaling,
                mappings=AWQ_MAPPINGS[position],
            ),
            QuantizationModifier(
                targets=targets, scheme=scheme, ignore=ignore
            ),
        ]
    else:  # pragma: no cover - validate 已拦截
        raise ValueError(algorithm)

    logger.info(
        "构建 recipe: algorithm=%s scheme=%s position=%s targets=%s",
        algorithm,
        scheme,
        position,
        targets,
    )
    return recipe
