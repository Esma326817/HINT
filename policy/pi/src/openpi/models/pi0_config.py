from collections.abc import Sequence
import dataclasses
from typing import TYPE_CHECKING, Literal

import flax.nnx as nnx
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
import openpi.models.gemma as _gemma
from openpi.shared import array_typing as at
import openpi.shared.nnx_utils as nnx_utils

if TYPE_CHECKING:
    from openpi.models.pi0 import Pi0


@dataclasses.dataclass(frozen=True)
class Pi0Config(_model.BaseModelConfig):
    dtype: str = "bfloat16"
    paligemma_variant: _gemma.Variant = "gemma_2b"
    action_expert_variant: _gemma.Variant = "gemma_300m"

    # Set the model specific defaults.
    action_dim: int = 32
    action_horizon: int = 50
    max_token_len: int = None  # type: ignore
    # Pi05 has two differences from Pi0:
    # - the state input is part of the discrete language tokens rather than a continuous input that is part of the suffix
    # - the action expert uses adaRMSNorm to inject the flow matching timestep
    pi05: bool = False
    # This config option is not used directly by the model, but it is read by the ModelTransformFactory.
    discrete_state_input: bool = None  # type: ignore

    # Optional grounding-guided action-to-image attention.
    attention_injection: bool = False
    # One-based Gemma transformer layer numbers (1 through the model depth).
    attention_injection_layers: Sequence[int] = ()
    attention_alpha_mode: Literal["fixed", "learned_bounded"] = "learned_bounded"
    attention_alpha: float = 0.5
    attention_alpha_max: float = 2.0

    # Optional training-free grounding bias inside the SigLIP vision encoder.
    vit_attention_injection: bool = False
    # One-based SigLIP transformer layer numbers (1 through the vision depth).
    vit_attention_injection_layers: Sequence[int] = ()
    vit_attention_alpha_mode: Literal["fixed"] = "fixed"
    vit_attention_alpha: float = 1.0
    # Optional camera-specific effective ViT alphas. None preserves the shared
    # vit_attention_alpha behavior for existing configs.
    vit_attention_global_alpha: float | None = None
    vit_attention_wrist_alpha: float | None = None

    def __post_init__(self):
        if self.max_token_len is None:
            object.__setattr__(self, "max_token_len", 200 if self.pi05 else 48)
        if self.discrete_state_input is None:
            object.__setattr__(self, "discrete_state_input", self.pi05)
        object.__setattr__(self, "attention_injection_layers", tuple(self.attention_injection_layers))
        object.__setattr__(self, "vit_attention_injection_layers", tuple(self.vit_attention_injection_layers))
        global_alpha = (
            self.vit_attention_alpha if self.vit_attention_global_alpha is None else self.vit_attention_global_alpha
        )
        wrist_alpha = (
            self.vit_attention_alpha if self.vit_attention_wrist_alpha is None else self.vit_attention_wrist_alpha
        )
        object.__setattr__(self, "vit_attention_global_alpha", float(global_alpha))
        object.__setattr__(self, "vit_attention_wrist_alpha", float(wrist_alpha))

    @property
    @override
    def model_type(self) -> _model.ModelType:
        if self.pi05:
            return _model.ModelType.PI05
        return _model.ModelType.PI0

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0":
        from openpi.models.pi0 import Pi0

        return Pi0(self, rngs=nnx.Rngs(rng))

    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[_model.Observation, _model.Actions]:
        image_spec = jax.ShapeDtypeStruct([batch_size, *_model.IMAGE_RESOLUTION, 3], jnp.float32)
        image_mask_spec = jax.ShapeDtypeStruct([batch_size], jnp.bool_)

        with at.disable_typechecking():
            observation_spec = _model.Observation(
                images={
                    "base_0_rgb": image_spec,
                    "left_wrist_0_rgb": image_spec,
                    "right_wrist_0_rgb": image_spec,
                },
                image_masks={
                    "base_0_rgb": image_mask_spec,
                    "left_wrist_0_rgb": image_mask_spec,
                    "right_wrist_0_rgb": image_mask_spec,
                },
                state=jax.ShapeDtypeStruct([batch_size, self.action_dim], jnp.float32),
                semantic_grounding_maps=(
                    {
                        "base_0_rgb": jax.ShapeDtypeStruct([batch_size, *_model.PATCH_GRID_RESOLUTION], jnp.float32),
                        "left_wrist_0_rgb": jax.ShapeDtypeStruct(
                            [batch_size, *_model.PATCH_GRID_RESOLUTION], jnp.float32
                        ),
                        "right_wrist_0_rgb": jax.ShapeDtypeStruct(
                            [batch_size, *_model.PATCH_GRID_RESOLUTION], jnp.float32
                        ),
                    }
                    if self.attention_injection or self.vit_attention_injection
                    else None
                ),
                tokenized_prompt=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32),
                tokenized_prompt_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], bool),
            )
        action_spec = jax.ShapeDtypeStruct([batch_size, self.action_horizon, self.action_dim], jnp.float32)

        return observation_spec, action_spec

    def get_freeze_filter(self) -> nnx.filterlib.Filter:
        """Returns the freeze filter based on the model config."""
        filters = []
        has_lora = False
        gemma_params_filter = nnx_utils.PathRegex(".*llm.*")
        action_expert_params_filter = nnx_utils.PathRegex(".*llm.*_1.*")
        if "lora" in self.paligemma_variant:
            filters.append(
                gemma_params_filter,
            )
            if "lora" not in self.action_expert_variant:
                # If only freeze gemma params, exclude action expert params.
                filters.append(
                    nnx.Not(action_expert_params_filter),
                )
            has_lora = True
        elif "lora" in self.action_expert_variant:
            filters.append(
                action_expert_params_filter,
            )
            has_lora = True

        if has_lora:
            # If any lora is used, exclude all lora params.
            filters.append(
                nnx.Not(nnx_utils.PathRegex(".*lora.*")),
            )
            if self.attention_injection:
                # Grounding strength remains trainable even when the Gemma base
                # weights are frozen for a LoRA fine-tune.
                filters.append(
                    nnx.Not(nnx_utils.PathRegex(".*attention_beta.*")),
                )
        if not filters:
            return nnx.Nothing
        return nnx.All(*filters)
