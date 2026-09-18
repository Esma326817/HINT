import jax
import jax.numpy as jnp
import numpy as np

from openpi.models import siglip


def _small_vit(*, attention_injection: bool) -> siglip._Module:
    return siglip._Module(  # noqa: SLF001
        num_classes=8,
        patch_size=(4, 4),
        width=16,
        depth=4,
        mlp_dim=32,
        num_heads=2,
        pool_type="none",
        head_zeroinit=False,
        scan=True,
        attention_injection=attention_injection,
        attention_injection_layers=(2, 4),
        attention_alpha_mode="fixed",
        attention_alpha=1.0,
    )


def test_vit_attention_injection_is_parameter_free_and_zero_map_is_noop():
    image = jax.random.normal(jax.random.key(1), (1, 8, 8, 3))
    zero_weights = jnp.zeros((1, 4), dtype=jnp.float32)
    base_model = _small_vit(attention_injection=False)
    injection_model = _small_vit(attention_injection=True)

    variables = base_model.init(jax.random.key(2), image)
    base_output, _ = base_model.apply(variables, image)
    injection_output, _ = injection_model.apply(variables, image, zero_weights)

    np.testing.assert_array_equal(injection_output, base_output)


def test_vit_attention_injection_changes_output_without_adding_parameters():
    image = jax.random.normal(jax.random.key(3), (1, 8, 8, 3))
    target_weights = jnp.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=jnp.float32)
    model = _small_vit(attention_injection=True)

    variables = model.init(jax.random.key(4), image)
    base_variables = _small_vit(attention_injection=False).init(jax.random.key(4), image)
    zero_output, _ = model.apply(variables, image, jnp.zeros_like(target_weights))
    target_output, _ = model.apply(variables, image, target_weights)
    doubled_target_output, _ = model.apply(variables, image, target_weights * 2.0)

    assert not np.allclose(target_output, zero_output)
    assert not np.allclose(doubled_target_output, target_output)
    assert jax.tree.structure(variables["params"]) == jax.tree.structure(base_variables["params"])
