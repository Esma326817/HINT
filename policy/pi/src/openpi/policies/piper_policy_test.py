import numpy as np

from openpi import transforms
from openpi.models import model as _model
from openpi.policies import piper_policy


def _example():
    image = np.zeros((270, 360, 3), dtype=np.uint8)
    attention_map = np.zeros((16, 16), dtype=np.float32)
    attention_map[5:11, 4:12] = 1.0
    return {
        "observation/global": image,
        "observation/left_wrist": image.copy(),
        "observation/right_wrist": image.copy(),
        "observation/state": np.zeros(14, dtype=np.float32),
        "semantic_grounding": {
            "global": {
                "spatial_encoding": "vit_patch_attention",
                "attention_map": attention_map.tolist(),
                "grid_size": [16, 16],
                "model_input_resolution": [224, 224],
                "patch_size": [14, 14],
                "valid": True,
            },
            "left_wrist": np.zeros((16, 16), dtype=np.float32),
            "right_wrist": None,
        },
    }


def test_piper_reads_reasoning_agent_patch_attention_map_directly():
    inputs = piper_policy.PiperInputs(
        model_type=_model.ModelType.PI05,
        attention_injection=True,
    )(_example())
    assert inputs["semantic_grounding_map"]["base_0_rgb"].shape == (16, 16)
    assert inputs["semantic_grounding_map"]["base_0_rgb"].sum() == 48
    assert not inputs["semantic_grounding_map"]["left_wrist_0_rgb"].any()
    assert not inputs["semantic_grounding_map"]["right_wrist_0_rgb"].any()

    resized = transforms.ResizeImages(224, 224)(inputs)
    mask = resized["semantic_grounding_map"]["base_0_rgb"]
    assert mask.shape == (16, 16)
    np.testing.assert_array_equal(mask[5:11, 4:12], np.ones((6, 8)))


def test_piper_attention_warmup_always_has_fixed_zero_masks():
    example = _example()
    example.pop("semantic_grounding")
    inputs = piper_policy.PiperInputs(
        model_type=_model.ModelType.PI05,
        attention_injection=True,
    )(example)
    assert set(inputs["semantic_grounding_map"]) == {
        "base_0_rgb",
        "left_wrist_0_rgb",
        "right_wrist_0_rgb",
    }
    assert all(not mask.any() for mask in inputs["semantic_grounding_map"].values())


def test_piper_can_max_normalize_only_global_grounding():
    example = _example()
    global_map = np.zeros((16, 16), dtype=np.float32)
    global_map[3, 4:8] = [0.65, 0.39, 0.02, 0.02]
    example["semantic_grounding"]["global"]["attention_map"] = global_map.tolist()
    wrist_map = np.zeros((16, 16), dtype=np.float32)
    wrist_map[2, 2] = 0.4
    example["semantic_grounding"]["left_wrist"] = wrist_map

    inputs = piper_policy.PiperInputs(
        model_type=_model.ModelType.PI05,
        attention_injection=True,
        global_semantic_grounding_max_normalize=True,
    )(example)

    np.testing.assert_allclose(
        inputs["semantic_grounding_map"]["base_0_rgb"][3, 4:8],
        [1.0, 0.6, 0.02 / 0.65, 0.02 / 0.65],
    )
    assert np.isclose(inputs["semantic_grounding_map"]["left_wrist_0_rgb"][2, 2], 0.4)
