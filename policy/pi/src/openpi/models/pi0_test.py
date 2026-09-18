import flax.nnx as nnx
import jax

import openpi.models.gemma as _gemma
import openpi.models.pi0_config as _pi0_config


def _get_frozen_state(config: _pi0_config.Pi0Config) -> nnx.State:
    abstract_model = nnx.eval_shape(config.create, jax.random.key(0))

    freeze_filter = config.get_freeze_filter()
    return nnx.state(abstract_model, nnx.All(nnx.Param, freeze_filter)).flat_state()


def test_pi0_full_finetune():
    config = _pi0_config.Pi0Config()
    state = _get_frozen_state(config)
    assert len(state) == 0


def test_pi0_gemma_lora():
    config = _pi0_config.Pi0Config(paligemma_variant="gemma_2b_lora")
    state = _get_frozen_state(config)
    assert len(state) == 9
    assert all("lora" not in p for p in state)
    assert all("llm" in p for p in state)
    assert all("_1" not in p for p in state)


def test_pi0_action_expert_lora():
    config = _pi0_config.Pi0Config(action_expert_variant="gemma_300m_lora")
    state = _get_frozen_state(config)
    # excluding embedder, rest of the params should be same as gemma_lora.
    assert len(state) == 8
    assert all("lora" not in p for p in state)
    assert all("llm" in p for p in state)
    # all frozen params should have _1 in their path since it's the action expert.
    assert all(any("_1" in p for p in path) for path in state)


def test_pi0_all_lora():
    config = _pi0_config.Pi0Config(paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora")
    state = _get_frozen_state(config)
    # sum of gemma_lora and action_expert_lora's frozen params.
    assert len(state) == 17
    assert all("lora" not in p for p in state)
    assert all("llm" in p for p in state)


def test_grounding_attention_parameter_is_initialized_per_layer():
    config = _pi0_config.Pi0Config(
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        attention_injection=True,
        attention_injection_layers=[3, 4],
    )
    model = nnx.eval_shape(config.create, jax.random.key(0))
    attention_params = [
        (path, value)
        for path, value in nnx.state(model, nnx.Param).flat_state().items()
        if any("attention_beta" in str(part) for part in path)
    ]
    assert len(attention_params) == 1
    assert attention_params[0][1].value.shape == (4,)


def test_grounding_attention_uses_one_based_layer_numbers():
    layer_gates = _gemma._make_attention_injection_layer_gates(4, [1, 3])  # noqa: SLF001
    assert layer_gates.tolist() == [True, False, True, False]


def test_grounding_attention_parameter_is_trainable_with_lora():
    config = _pi0_config.Pi0Config(
        paligemma_variant="gemma_2b_lora",
        attention_injection=True,
        attention_injection_layers=[15, 16, 17, 18],
    )
    frozen_state = _get_frozen_state(config)
    assert all("attention_beta" not in str(path) for path in frozen_state)


def test_vit_grounding_attention_config_is_fixed_and_uses_one_based_layers():
    config = _pi0_config.Pi0Config(
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        vit_attention_injection=True,
        vit_attention_injection_layers=[6, 12, 18, 24],
        vit_attention_alpha_mode="fixed",
        vit_attention_alpha=1.0,
    )

    assert config.vit_attention_injection_layers == (6, 12, 18, 24)
    assert config.vit_attention_alpha == 1.0
    assert config.vit_attention_global_alpha == 1.0
    assert config.vit_attention_wrist_alpha == 1.0


def test_vit_grounding_attention_config_supports_camera_specific_alphas():
    config = _pi0_config.Pi0Config(
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        vit_attention_injection=True,
        vit_attention_injection_layers=[6],
        vit_attention_global_alpha=2.0,
        vit_attention_wrist_alpha=1.0,
    )

    assert config.vit_attention_global_alpha == 2.0
    assert config.vit_attention_wrist_alpha == 1.0
