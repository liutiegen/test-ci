"""TRL SFTTrainer 与 llmcompressor QAT 生命周期的桥接。

transformers TrainerCallback 事件 -> llmcompressor session 回调：
- on_step_begin        -> batch_start
- on_pre_optimizer_step -> optim_pre_step
- on_optimizer_step    -> optim_post_step
- on_step_end          -> batch_end
- on_train_end         -> session.finalize

注意：AWQ/SmoothQuant 是一次性 transform，不能用于训练；QAT 阶段只挂
QuantizationModifier，按已选方案保持伪量化并随训练更新。
"""

from __future__ import annotations

from typing import Any

from llmcompressor import create_session
from llmcompressor.core.session_functions import callbacks
from llmcompressor.modifiers.quantization import QuantizationModifier
from transformers import TrainerCallback

from quantflow.logging_utils import get_logger
from quantflow.recipes import POSITION_IGNORE, POSITION_TARGETS

logger = get_logger(__name__)


def build_qat_modifier(scheme: str, position: str) -> QuantizationModifier:
    """构建与已选量化方案一致的 QAT 修饰器（仅伪量化，不含一次性 transform）。"""
    return QuantizationModifier(
        targets=POSITION_TARGETS[position],
        scheme=scheme,
        ignore=POSITION_IGNORE[position],
    )


class QATCallback(TrainerCallback):
    """把 HF/TRL Trainer 的训练步骤接入 llmcompressor 压缩会话。"""

    def __init__(self, trainer: Any, recipe: list[Any]):
        self.trainer = trainer
        self.recipe = recipe
        self.session_cm = create_session()
        self.session = self.session_cm.__enter__()
        self._initialized = False

    def _ensure_initialized(self) -> None:
        if self._initialized:
            return
        # 此时 Trainer 已创建好 optimizer
        self.session.initialize(
            recipe=self.recipe,
            model=self.trainer.model,
            optimizer=self.trainer.optimizer,
            attach_optim_callbacks=True,
        )
        self._initialized = True
        logger.info("QAT 会话已初始化")

    def on_step_begin(self, args, state, control, **kwargs):
        self._ensure_initialized()
        callbacks.batch_start()

    def on_pre_optimizer_step(self, args, state, control, **kwargs):
        callbacks.optim_pre_step()

    def on_optimizer_step(self, args, state, control, **kwargs):
        callbacks.optim_post_step()

    def on_step_end(self, args, state, control, **kwargs):
        callbacks.batch_end()

    def on_train_end(self, args, state, control, **kwargs):
        if self._initialized:
            self.session.finalize()
        self.session_cm.__exit__(None, None, None)
        logger.info("QAT 会话已结束并 finalize")
