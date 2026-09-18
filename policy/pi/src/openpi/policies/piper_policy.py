import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_libero_example() -> dict:
    """Creates a random input example for the Libero policy."""
    return {
        "observation/state": np.random.rand(8),
        "observation/global": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/left_wrist": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/right_wrist": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "prompt": "do something",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


def _patch_attention_map(
    payload,
    *,
    model_input_resolution: tuple[int, int],
    vit_patch_size: int,
) -> np.ndarray:
    """Read ReasoningAgent's model-space ViT patch weights without rerasterizing."""
    height, width = model_input_resolution
    if height % vit_patch_size or width % vit_patch_size:
        raise ValueError(
            f"model input resolution {model_input_resolution} must be divisible by ViT patch size {vit_patch_size}"
        )
    expected_shape = (height // vit_patch_size, width // vit_patch_size)
    if payload is None:
        return np.zeros(expected_shape, dtype=np.float32)
    if isinstance(payload, dict):
        encoding = payload.get("spatial_encoding")
        if encoding != "vit_patch_attention":
            raise ValueError(f"Unsupported semantic grounding encoding: {encoding}")
        response_resolution = tuple(payload.get("model_input_resolution") or ())
        if response_resolution != tuple(model_input_resolution):
            raise ValueError(
                f"ReasoningAgent attention resolution {response_resolution} does not match "
                f"policy input resolution {model_input_resolution}"
            )
        response_patch_size = tuple(payload.get("patch_size") or ())
        if response_patch_size != (vit_patch_size, vit_patch_size):
            raise ValueError(
                f"ReasoningAgent patch size {response_patch_size} does not match "
                f"policy patch size {(vit_patch_size, vit_patch_size)}"
            )
        if not bool(payload.get("valid", False)):
            return np.zeros(expected_shape, dtype=np.float32)
        payload = payload.get("attention_map")
    attention_map = np.asarray(payload, dtype=np.float32)
    if attention_map.shape != expected_shape:
        raise ValueError(f"semantic patch attention map must be {expected_shape}, got {attention_map.shape}")
    if not np.isfinite(attention_map).all():
        raise ValueError("semantic patch attention map contains non-finite weights")
    return np.clip(attention_map, 0.0, 1.0)


def _max_normalize_attention_map(attention_map: np.ndarray) -> np.ndarray:
    """Scale a non-empty patch map so its strongest patch has weight one."""
    max_value = float(np.max(attention_map, initial=0.0))
    if max_value <= 0.0:
        return attention_map
    return attention_map / max_value


@dataclasses.dataclass(frozen=True)
class PiperInputs(transforms.DataTransformFn):
    """
    This class is used to convert inputs to the model to the expected format. It is used for both training and inference.

    For your own dataset, you can copy this class and modify the keys based on the comments below to pipe
    the correct elements of your dataset into the model.
    """

    # Determines which model will be used.
    # Do not change this for your own dataset.
    model_type: _model.ModelType
    attention_injection: bool = False
    global_semantic_grounding_max_normalize: bool = False
    model_input_resolution: tuple[int, int] = _model.IMAGE_RESOLUTION
    vit_patch_size: int = 14

    def __call__(self, data: dict) -> dict:
        # Possibly need to parse images to uint8 (H,W,C) since LeRobot automatically
        # stores as float32 (C,H,W), gets skipped for policy inference.
        # Keep this for your own dataset, but if your dataset stores the images
        # in a different key than "observation/image" or "observation/wrist_image",
        # you should change it below.
        # Pi0 models support three image inputs at the moment: one third-person view,
        # and two wrist views (left and right). If your dataset does not have a particular type
        # of image, e.g. wrist images, you can comment it out here and replace it with zeros like we do for the
        # right wrist image below.
        global_image = _parse_image(data["observation/global"])
        left_wrist_image = _parse_image(data["observation/left_wrist"])
        right_wrist_image = _parse_image(data["observation/right_wrist"])

        # Create inputs dict. Do not change the keys in the dict below.
        inputs = {
            "state": data["observation/state"],
            "image": {
                "base_0_rgb": global_image,
                "left_wrist_0_rgb": left_wrist_image,
                # Pad any non-existent images with zero-arrays of the appropriate shape.
                "right_wrist_0_rgb": right_wrist_image,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                # We only mask padding images for pi0 model, not pi0-FAST. Do not change this for your own dataset.
                "right_wrist_0_rgb": np.True_,
            },
        }

        if self.attention_injection:
            semantic_grounding = data.get("semantic_grounding") or {}
            global_attention_map = _patch_attention_map(
                semantic_grounding.get("global"),
                model_input_resolution=self.model_input_resolution,
                vit_patch_size=self.vit_patch_size,
            )
            if self.global_semantic_grounding_max_normalize:
                global_attention_map = _max_normalize_attention_map(global_attention_map)
            inputs["semantic_grounding_map"] = {
                "base_0_rgb": global_attention_map,
                "left_wrist_0_rgb": _patch_attention_map(
                    semantic_grounding.get("left_wrist"),
                    model_input_resolution=self.model_input_resolution,
                    vit_patch_size=self.vit_patch_size,
                ),
                "right_wrist_0_rgb": _patch_attention_map(
                    semantic_grounding.get("right_wrist"),
                    model_input_resolution=self.model_input_resolution,
                    vit_patch_size=self.vit_patch_size,
                ),
            }

        # Pad actions to the model action dimension. Keep this for your own dataset.
        # Actions are only available during training.
        if "actions" in data:
            # inputs["actions"] = data["actions"]
            if data["actions"].shape[-1] == 14:
                actions_np = np.asarray(data["actions"])
                inputs["actions"] = actions_np

        # Pass the prompt (aka language instruction) to the model.
        # Keep this for your own dataset (but modify the key if the instruction is not
        # stored in "prompt"; the output dict always needs to have the key "prompt").
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class PiperOutputs(transforms.DataTransformFn):
    """
    This class is used to convert outputs from the model back the the dataset specific format. It is
    used for inference only.

    For your own dataset, you can copy this class and modify the action dimension based on the comments below.
    """

    def __call__(self, data: dict) -> dict:
        # Only return the first N actions -- since we padded actions above to fit the model action
        # dimension, we need to now parse out the correct number of actions in the return dict.
        return {"actions": np.asarray(data["actions"][:, :14])}
