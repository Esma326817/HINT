from collections.abc import Callable, Mapping, Sequence
import dataclasses
import re
from typing import Protocol, TypeAlias, TypeVar, runtime_checkable

import flax.traverse_util as traverse_util
import jax
import numpy as np
import torch
from openpi_client import image_tools
from transforms3d.axangles import axangle2mat, mat2axangle
from transforms3d.quaternions import mat2quat, quat2mat

from openpi.models import tokenizer as _tokenizer
from openpi.shared import array_typing as at
from openpi.shared import normalize as _normalize


def _to_numpy(x):
    """Convert torch tensor to numpy array if needed."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return x

DataDict: TypeAlias = at.PyTree
NormStats: TypeAlias = _normalize.NormStats


T = TypeVar("T")
S = TypeVar("S")


@runtime_checkable
class DataTransformFn(Protocol):
    def __call__(self, data: DataDict) -> DataDict:
        """Apply transformation to the data.

        Args:
            data: The data to apply the transform to. This is a possibly nested dictionary that contains
                unbatched data elements. Each leaf is expected to be a numpy array. Using JAX arrays is allowed
                but not recommended since it may result in extra GPU memory usage inside data loader worker
                processes.

        Returns:
            The transformed data. Could be the input `data` that was modified in place, or a new data structure.
        """


@dataclasses.dataclass(frozen=True)
class Group:
    """A group of transforms."""

    # Transforms that are applied to the model input data.
    inputs: Sequence[DataTransformFn] = ()

    # Transforms that are applied to the model output data.
    outputs: Sequence[DataTransformFn] = ()

    def push(self, *, inputs: Sequence[DataTransformFn] = (), outputs: Sequence[DataTransformFn] = ()) -> "Group":
        """Append transforms to the group and return a new group.

        Args:
            inputs: Appended to the *end* of the current input transforms.
            outputs: Appended to the *beginning* of the current output transforms.

        Returns:
            A new group with the appended transforms.
        """
        return Group(inputs=(*self.inputs, *inputs), outputs=(*outputs, *self.outputs))


@dataclasses.dataclass(frozen=True)
class CompositeTransform(DataTransformFn):
    """A composite transform that applies a sequence of transforms in order."""

    transforms: Sequence[DataTransformFn]

    def __call__(self, data: DataDict) -> DataDict:
        for transform in self.transforms:
            data = transform(data)
        return data


def compose(transforms: Sequence[DataTransformFn]) -> DataTransformFn:
    """Compose a sequence of transforms into a single transform."""
    return CompositeTransform(transforms)


@dataclasses.dataclass(frozen=True)
class RepackTransform(DataTransformFn):
    """Repacks an input dictionary into a new dictionary.

    Repacking is defined using a dictionary where the keys are the new keys and the values
    are the flattened paths to the old keys. We use '/' as the separator during flattening.

    Example:
    {
        "images": {
            "cam_high": "observation.images.top",
            "cam_low": "observation.images.bottom",
        },
        "state": "observation.state",
        "actions": "action",
    }
    """

    structure: at.PyTree[str]

    def __call__(self, data: DataDict) -> DataDict:
        flat_item = flatten_dict(data)
        return jax.tree.map(lambda k: flat_item[k], self.structure)


@dataclasses.dataclass(frozen=True)
class InjectDefaultPrompt(DataTransformFn):
    prompt: str | None

    def __call__(self, data: DataDict) -> DataDict:
        if self.prompt is not None and "prompt" not in data:
            data["prompt"] = np.asarray(self.prompt)
        return data


@dataclasses.dataclass(frozen=True)
class Normalize(DataTransformFn):
    norm_stats: at.PyTree[NormStats] | None
    # If true, will use quantile normalization. Otherwise, normal z-score normalization will be used.
    use_quantiles: bool = False
    # If true, will raise an error if any of the keys in the norm stats are not present in the data.
    strict: bool = False

    def __post_init__(self):
        if self.norm_stats is not None and self.use_quantiles:
            _assert_quantile_stats(self.norm_stats)

    def __call__(self, data: DataDict) -> DataDict:
        if self.norm_stats is None:
            return data

        return apply_tree(
            data,
            self.norm_stats,
            self._normalize_quantile if self.use_quantiles else self._normalize,
            strict=self.strict,
        )

    def _normalize(self, x, stats: NormStats):
        mean, std = stats.mean[..., : x.shape[-1]], stats.std[..., : x.shape[-1]]
        return (x - mean) / (std + 1e-6)

    def _normalize_quantile(self, x, stats: NormStats):
        assert stats.q01 is not None
        assert stats.q99 is not None
        q01, q99 = stats.q01[..., : x.shape[-1]], stats.q99[..., : x.shape[-1]]
        return (x - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0


@dataclasses.dataclass(frozen=True)
class Unnormalize(DataTransformFn):
    norm_stats: at.PyTree[NormStats] | None
    # If true, will use quantile normalization. Otherwise, normal z-score normalization will be used.
    use_quantiles: bool = False

    def __post_init__(self):
        if self.norm_stats is not None and self.use_quantiles:
            _assert_quantile_stats(self.norm_stats)

    def __call__(self, data: DataDict) -> DataDict:
        if self.norm_stats is None:
            return data

        # Make sure that all the keys in the norm stats are present in the data.
        return apply_tree(
            data,
            self.norm_stats,
            self._unnormalize_quantile if self.use_quantiles else self._unnormalize,
            strict=True,
        )

    def _unnormalize(self, x, stats: NormStats):
        mean = pad_to_dim(stats.mean, x.shape[-1], axis=-1, value=0.0)
        std = pad_to_dim(stats.std, x.shape[-1], axis=-1, value=1.0)
        return x * (std + 1e-6) + mean

    def _unnormalize_quantile(self, x, stats: NormStats):
        assert stats.q01 is not None
        assert stats.q99 is not None
        q01, q99 = stats.q01, stats.q99
        if (dim := q01.shape[-1]) < x.shape[-1]:
            return np.concatenate([(x[..., :dim] + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01, x[..., dim:]], axis=-1)
        return (x + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01


@dataclasses.dataclass(frozen=True)
class ResizeImages(DataTransformFn):
    height: int
    width: int

    def __call__(self, data: DataDict) -> DataDict:
        data["image"] = {k: image_tools.resize_with_pad(v, self.height, self.width) for k, v in data["image"].items()}
        # semantic_grounding_map is already a model-space ViT patch grid from
        # ReasoningAgent. It must not be resized as a pixel-space mask.
        return data


@dataclasses.dataclass(frozen=True)
class SubsampleActions(DataTransformFn):
    stride: int

    def __call__(self, data: DataDict) -> DataDict:
        data["actions"] = data["actions"][:: self.stride]
        return data


@dataclasses.dataclass(frozen=True)
class DeltaActions(DataTransformFn):
    """Repacks absolute actions into delta action space."""

    # Boolean mask for the action dimensions to be repacked into delta action space. Length
    # can be smaller than the actual number of dimensions. If None, this transform is a no-op.
    # See `make_bool_mask` for more details.
    mask: Sequence[bool] | None

    def __call__(self, data: DataDict) -> DataDict:
        if "actions" not in data or self.mask is None:
            return data

        state, actions = data["state"], data["actions"]
        mask = np.asarray(self.mask)
        dims = mask.shape[-1]
        actions[..., :dims] -= np.expand_dims(np.where(mask, state[..., :dims], 0), axis=-2)
        data["actions"] = actions

        return data


@dataclasses.dataclass(frozen=True)
class AbsoluteActions(DataTransformFn):
    """Repacks delta actions into absolute action space."""

    # Boolean mask for the action dimensions to be repacked into absolute action space. Length
    # can be smaller than the actual number of dimensions. If None, this transform is a no-op.
    # See `make_bool_mask` for more details.
    mask: Sequence[bool] | None

    def __call__(self, data: DataDict) -> DataDict:
        if "actions" not in data or self.mask is None:
            return data

        state, actions = data["state"], data["actions"]
        mask = np.asarray(self.mask)
        dims = mask.shape[-1]
        actions[..., :dims] += np.expand_dims(np.where(mask, state[..., :dims], 0), axis=-2)
        data["actions"] = actions

        return data

@dataclasses.dataclass(frozen=True)
class DeltaTCPActions(DataTransformFn):
    """Repacks absolute actions into delta action space."""

    # Boolean mask for the action dimensions to be repacked into delta action space. Length
    # can be smaller than the actual number of dimensions. If None, this transform is a no-op.
    # See `make_bool_mask` for more details.
    mask: Sequence[bool] | None

    def __call__(self, data: DataDict) -> DataDict:
        if "actions" not in data or self.mask is None:
            return data

        state, actions = data["state"], data["actions"]
        mask = np.asarray(self.mask)
        dims = mask.shape[-1]
        actions[..., :dims] -= np.expand_dims(np.where(mask, state[..., :dims], 0), axis=-2)
        data["actions"] = actions

        return data


@dataclasses.dataclass(frozen=True)
class AbsoluteTCPActions(DataTransformFn):
    """Repacks delta actions into absolute action space."""

    # Boolean mask for the action dimensions to be repacked into absolute action space. Length
    # can be smaller than the actual number of dimensions. If None, this transform is a no-op.
    # See `make_bool_mask` for more details.
    mask: Sequence[bool] | None

    def __call__(self, data: DataDict) -> DataDict:
        if "actions" not in data or self.mask is None:
            return data

        state, actions = data["state"], data["actions"]
        mask = np.asarray(self.mask)
        dims = mask.shape[-1]
        actions[..., :dims] += np.expand_dims(np.where(mask, state[..., :dims], 0), axis=-2)
        data["actions"] = actions

        return data


# =============================================================================
# Rotation conversion utilities for TCP pose delta transforms
# Uses transforms3d for axis-angle and quaternion; Euler XYZ is pure NumPy (vectorized).
# =============================================================================

def axis_angle_to_matrix(axis_angle: np.ndarray) -> np.ndarray:

    original_shape = axis_angle.shape[:-1]
    axis_angle_flat = axis_angle.reshape(-1, 3)

    matrices = []
    for aa in axis_angle_flat:
        angle = np.linalg.norm(aa)
        if angle < 1e-8:
            matrices.append(np.eye(3))
        else:
            axis = aa / angle
            matrices.append(axangle2mat(axis, angle))

    result = np.stack(matrices, axis=0)
    return result.reshape(original_shape + (3, 3))


def matrix_to_axis_angle(R: np.ndarray) -> np.ndarray:

    original_shape = R.shape[:-2]
    R_flat = R.reshape(-1, 3, 3)

    axis_angles = []
    for mat in R_flat:
        try:
            axis, angle = mat2axangle(mat)
            axis_angles.append(axis * angle)
        except Exception:
            # Fallback for edge cases (identity matrix, etc.)
            axis_angles.append(np.zeros(3))

    result = np.stack(axis_angles, axis=0)
    return result.reshape(original_shape + (3,))


def quaternion_to_matrix(quat: np.ndarray) -> np.ndarray:

    original_shape = quat.shape[:-1]
    quat_flat = quat.reshape(-1, 4)

    matrices = []
    for q in quat_flat:
        matrices.append(quat2mat(q))

    result = np.stack(matrices, axis=0)
    return result.reshape(original_shape + (3, 3))


def matrix_to_quaternion(R: np.ndarray) -> np.ndarray:

    original_shape = R.shape[:-2]
    R_flat = R.reshape(-1, 3, 3)

    quats = []
    for mat in R_flat:
        quats.append(mat2quat(mat))

    result = np.stack(quats, axis=0)
    return result.reshape(original_shape + (4,))


def rotation_6d_to_matrix(rot_6d: np.ndarray) -> np.ndarray:

    a1 = rot_6d[..., :3]
    a2 = rot_6d[..., 3:6]

    # Gram-Schmidt orthogonalization
    b1 = a1 / (np.linalg.norm(a1, axis=-1, keepdims=True) + 1e-8)
    b2 = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = b2 / (np.linalg.norm(b2, axis=-1, keepdims=True) + 1e-8)
    b3 = np.cross(b1, b2)

    # Stack as rows: R = [b1; b2; b3]
    return np.stack([b1, b2, b3], axis=-2)


def matrix_to_rotation_6d(R: np.ndarray) -> np.ndarray:
    # Extract first two rows and flatten
    return np.concatenate([R[..., 0, :], R[..., 1, :]], axis=-1)


def euler_xyz_to_matrix(euler: np.ndarray) -> np.ndarray:
    """Euler angles ``(rx, ry, rz)`` in radians to rotation matrix.

    Convention: **extrinsic static XYZ** (transforms3d ``sxyz``), i.e.
    ``R = Rz(rz) @ Ry(ry) @ Rx(rx)``, matching ``scipy.spatial.transform.Rotation.from_euler("xyz", ...)``.
    Fully vectorized over batch dimensions.
    """
    euler = np.asarray(euler, dtype=np.float64)
    original_shape = euler.shape[:-1]
    x, y, z = euler[..., 0], euler[..., 1], euler[..., 2]
    cx, sx = np.cos(x), np.sin(x)
    cy, sy = np.cos(y), np.sin(y)
    cz, sz = np.cos(z), np.sin(z)
    r00 = cz * cy
    r01 = cz * sx * sy - sz * cx
    r02 = cz * cx * sy + sz * sx
    r10 = sz * cy
    r11 = sz * sx * sy + cz * cx
    r12 = sz * cx * sy - cz * sx
    r20 = -sy
    r21 = cy * sx
    r22 = cy * cx
    return np.stack(
        [
            np.stack([r00, r01, r02], axis=-1),
            np.stack([r10, r11, r12], axis=-1),
            np.stack([r20, r21, r22], axis=-1),
        ],
        axis=-2,
    ).reshape(original_shape + (3, 3))


def matrix_to_euler_xyz(R: np.ndarray) -> np.ndarray:
    """Rotation matrix to Euler angles ``(rx, ry, rz)`` in radians (inverse of :func:`euler_xyz_to_matrix`).

    Uses ``atan2`` for a stable branch; at gimbal lock (``|cos(ry)|`` tiny) angles remain finite
    but may be non-unique, same as standard Euler extraction.
    """
    R = np.asarray(R, dtype=np.float64)
    original_shape = R.shape[:-2]
    r00, r10 = R[..., 0, 0], R[..., 1, 0]
    r20, r21, r22 = R[..., 2, 0], R[..., 2, 1], R[..., 2, 2]
    rx = np.arctan2(r21, r22)
    ry = np.arctan2(-r20, np.hypot(r21, r22))
    rz = np.arctan2(r10, r00)
    return np.stack([rx, ry, rz], axis=-1).reshape(original_shape + (3,))


def _canonical_tcp_pose_rotation_repr(rotation_repr: str) -> str:
    """Normalize aliases (e.g. ``euler`` → ``euler_xyz``)."""
    if rotation_repr == "euler":
        return "euler_xyz"
    return rotation_repr


@dataclasses.dataclass(frozen=True)
class DeltaTCPPoseActions(DataTransformFn):
    """Repacks absolute TCP pose actions into delta action space with proper rotation handling.

    For position: delta_pos = action_pos - state_pos
    For rotation: delta_R = action_R @ state_R^(-1) (matrix multiplication with inverse)

    This ensures rotation deltas are computed correctly on the SO(3) manifold.
    """

    # Number of position dimensions (usually 3 for xyz)
    pos_dims: int = 3

    # Rotation representation: "axis_angle", "quaternion", "rotation_6d", "matrix",
    # "euler_xyz" (extrinsic XYZ / scipy ``xyz``), or alias "euler".
    rotation_repr: str = "axis_angle"

    # Number of rotation dimensions in the input (auto-detected if None)
    # axis_angle: 3, quaternion: 4, rotation_6d: 6, matrix: 9, euler_xyz: 3
    rot_dims: int | None = None

    # Whether to apply delta transform. If False, this is a no-op.
    enabled: bool = True

    def __post_init__(self):
        valid_reprs = ("axis_angle", "quaternion", "rotation_6d", "matrix", "euler_xyz")
        canon = _canonical_tcp_pose_rotation_repr(self.rotation_repr)
        if canon not in valid_reprs:
            raise ValueError(
                f"rotation_repr must be one of {valid_reprs} or 'euler', got {self.rotation_repr!r}"
            )
        if self.rotation_repr != canon:
            object.__setattr__(self, "rotation_repr", canon)

    def __call__(self, data: DataDict) -> DataDict:
        if "actions" not in data or not self.enabled:
            return data

        state, actions = _to_numpy(data["state"]), _to_numpy(data["actions"])
        rot_dims = self._get_rotation_dims()

        # Extract position
        state_pos = state[..., :self.pos_dims]
        actions_pos = actions[..., :self.pos_dims]

        # Compute position delta (simple subtraction)
        delta_pos = actions_pos - np.expand_dims(state_pos, axis=-2)

        # Extract rotation
        rot_start = self.pos_dims
        rot_end = self.pos_dims + rot_dims
        state_rot = state[..., rot_start:rot_end]
        actions_rot = actions[..., rot_start:rot_end]

        # Convert to rotation matrices
        state_R = self._to_matrix(state_rot)
        actions_R = self._to_matrix(actions_rot)

        # Compute delta rotation: R_delta = R_action @ R_state^(-1)
        # For rotation matrices, inverse = transpose
        # Expand state_R for broadcasting with action horizon
        state_R_inv = np.swapaxes(state_R, -1, -2)
        state_R_inv = np.expand_dims(state_R_inv, axis=-3)  # Add action horizon dim
        delta_R = actions_R @ state_R_inv

        # Convert back to original representation
        delta_rot = self._from_matrix(delta_R)

        # Get remaining dimensions (e.g., gripper)
        remaining = actions[..., rot_end:]

        # Reconstruct actions
        data["actions"] = np.concatenate([delta_pos, delta_rot, remaining], axis=-1)

        return data

    def _get_rotation_dims(self) -> int:
        if self.rot_dims is not None:
            return self.rot_dims
        if self.rotation_repr == "axis_angle":
            return 3
        elif self.rotation_repr == "quaternion":
            return 4
        elif self.rotation_repr == "rotation_6d":
            return 6
        elif self.rotation_repr == "matrix":
            return 9
        elif self.rotation_repr == "euler_xyz":
            return 3
        else:
            raise ValueError(f"Unknown rotation representation: {self.rotation_repr}")

    def _to_matrix(self, rot: np.ndarray) -> np.ndarray:
        if self.rotation_repr == "axis_angle":
            return axis_angle_to_matrix(rot)
        elif self.rotation_repr == "quaternion":
            return quaternion_to_matrix(rot)
        elif self.rotation_repr == "rotation_6d":
            return rotation_6d_to_matrix(rot)
        elif self.rotation_repr == "matrix":
            return rot.reshape(rot.shape[:-1] + (3, 3))
        elif self.rotation_repr == "euler_xyz":
            return euler_xyz_to_matrix(rot)
        else:
            raise ValueError(f"Unknown rotation representation: {self.rotation_repr}")

    def _from_matrix(self, R: np.ndarray) -> np.ndarray:
        if self.rotation_repr == "axis_angle":
            return matrix_to_axis_angle(R)
        elif self.rotation_repr == "quaternion":
            return matrix_to_quaternion(R)
        elif self.rotation_repr == "rotation_6d":
            return matrix_to_rotation_6d(R)
        elif self.rotation_repr == "matrix":
            return R.reshape(R.shape[:-2] + (9,))
        elif self.rotation_repr == "euler_xyz":
            return matrix_to_euler_xyz(R)
        else:
            raise ValueError(f"Unknown rotation representation: {self.rotation_repr}")


@dataclasses.dataclass(frozen=True)
class AbsoluteTCPPoseActions(DataTransformFn):
    """Repacks delta TCP pose actions into absolute action space with proper rotation handling.

    For position: action_pos = delta_pos + state_pos
    For rotation: action_R = delta_R @ state_R (matrix multiplication)

    This ensures rotation is recovered correctly on the SO(3) manifold.
    """

    # Number of position dimensions (usually 3 for xyz)
    pos_dims: int = 3

    # Rotation representation: "axis_angle", "quaternion", "rotation_6d", "matrix",
    # "euler_xyz" (extrinsic XYZ / scipy ``xyz``), or alias "euler".
    rotation_repr: str = "axis_angle"

    # Number of rotation dimensions in the input (auto-detected if None)
    # axis_angle: 3, quaternion: 4, rotation_6d: 6, matrix: 9, euler_xyz: 3
    rot_dims: int | None = None

    # Whether to apply absolute transform. If False, this is a no-op.
    enabled: bool = True

    def __post_init__(self):
        valid_reprs = ("axis_angle", "quaternion", "rotation_6d", "matrix", "euler_xyz")
        canon = _canonical_tcp_pose_rotation_repr(self.rotation_repr)
        if canon not in valid_reprs:
            raise ValueError(
                f"rotation_repr must be one of {valid_reprs} or 'euler', got {self.rotation_repr!r}"
            )
        if self.rotation_repr != canon:
            object.__setattr__(self, "rotation_repr", canon)

    def __call__(self, data: DataDict) -> DataDict:
        if "actions" not in data or not self.enabled:
            return data

        state, actions = _to_numpy(data["state"]), _to_numpy(data["actions"])
        rot_dims = self._get_rotation_dims()

        # Extract position
        state_pos = state[..., :self.pos_dims]
        delta_pos = actions[..., :self.pos_dims]

        # Compute absolute position
        abs_pos = delta_pos + np.expand_dims(state_pos, axis=-2)

        # Extract rotation
        rot_start = self.pos_dims
        rot_end = self.pos_dims + rot_dims
        state_rot = state[..., rot_start:rot_end]
        delta_rot = actions[..., rot_start:rot_end]

        # Convert to rotation matrices
        state_R = self._to_matrix(state_rot)
        delta_R = self._to_matrix(delta_rot)

        # Compute absolute rotation: R_action = R_delta @ R_state
        # Expand state_R for broadcasting with action horizon
        state_R_expanded = np.expand_dims(state_R, axis=-3)  # Add action horizon dim
        abs_R = delta_R @ state_R_expanded

        # Convert back to original representation
        abs_rot = self._from_matrix(abs_R)

        # Get remaining dimensions (e.g., gripper)
        remaining = actions[..., rot_end:]

        # Reconstruct actions
        data["actions"] = np.concatenate([abs_pos, abs_rot, remaining], axis=-1)

        return data

    def _get_rotation_dims(self) -> int:
        if self.rot_dims is not None:
            return self.rot_dims
        if self.rotation_repr == "axis_angle":
            return 3
        elif self.rotation_repr == "quaternion":
            return 4
        elif self.rotation_repr == "rotation_6d":
            return 6
        elif self.rotation_repr == "matrix":
            return 9
        elif self.rotation_repr == "euler_xyz":
            return 3
        else:
            raise ValueError(f"Unknown rotation representation: {self.rotation_repr}")

    def _to_matrix(self, rot: np.ndarray) -> np.ndarray:
        if self.rotation_repr == "axis_angle":
            return axis_angle_to_matrix(rot)
        elif self.rotation_repr == "quaternion":
            return quaternion_to_matrix(rot)
        elif self.rotation_repr == "rotation_6d":
            return rotation_6d_to_matrix(rot)
        elif self.rotation_repr == "matrix":
            return rot.reshape(rot.shape[:-1] + (3, 3))
        elif self.rotation_repr == "euler_xyz":
            return euler_xyz_to_matrix(rot)
        else:
            raise ValueError(f"Unknown rotation representation: {self.rotation_repr}")

    def _from_matrix(self, R: np.ndarray) -> np.ndarray:
        if self.rotation_repr == "axis_angle":
            return matrix_to_axis_angle(R)
        elif self.rotation_repr == "quaternion":
            return matrix_to_quaternion(R)
        elif self.rotation_repr == "rotation_6d":
            return matrix_to_rotation_6d(R)
        elif self.rotation_repr == "matrix":
            return R.reshape(R.shape[:-2] + (9,))
        elif self.rotation_repr == "euler_xyz":
            return matrix_to_euler_xyz(R)
        else:
            raise ValueError(f"Unknown rotation representation: {self.rotation_repr}")


@dataclasses.dataclass(frozen=True)
class DualArmDeltaTCPPoseActions(DataTransformFn):
    """Delta TCP pose transform for dual-arm robots.

    Applies DeltaTCPPoseActions independently to each arm's 7D slice:
        left  arm: state/actions[..., 0:7]   → [pos(3), rot(3), gripper(1)]
        right arm: state/actions[..., 7:14]  → [pos(3), rot(3), gripper(1)]
        where rot(3) is axis-angle, euler_xyz, etc. per ``rotation_repr``.

    The gripper dimension of each arm is passed through unchanged.
    """

    pos_dims: int = 3
    rotation_repr: str = "axis_angle"
    enabled: bool = True

    # single-arm slice width (pos + rot + gripper)
    arm_dims: int = 7

    def __post_init__(self):
        valid_reprs = ("axis_angle", "quaternion", "rotation_6d", "matrix", "euler_xyz")
        canon = _canonical_tcp_pose_rotation_repr(self.rotation_repr)
        if canon not in valid_reprs:
            raise ValueError(
                f"rotation_repr must be one of {valid_reprs} or 'euler', got {self.rotation_repr!r}"
            )
        if self.rotation_repr != canon:
            object.__setattr__(self, "rotation_repr", canon)

    def __call__(self, data: DataDict) -> DataDict:
        if "actions" not in data or not self.enabled:
            return data

        left_data = {
            "state":   data["state"][..., :self.arm_dims],
            "actions": data["actions"][..., :self.arm_dims],
        }
        right_data = {
            "state":   data["state"][..., self.arm_dims:2 * self.arm_dims],
            "actions": data["actions"][..., self.arm_dims:2 * self.arm_dims],
        }

        delta_fn = DeltaTCPPoseActions(pos_dims=self.pos_dims, rotation_repr=self.rotation_repr)
        left_data  = delta_fn(left_data)
        right_data = delta_fn(right_data)

        data["actions"] = np.concatenate([left_data["actions"], right_data["actions"]], axis=-1)
        return data


@dataclasses.dataclass(frozen=True)
class DualArmAbsoluteTCPPoseActions(DataTransformFn):
    """Inverse of DualArmDeltaTCPPoseActions — recovers absolute poses for both arms."""

    pos_dims: int = 3
    rotation_repr: str = "axis_angle"
    enabled: bool = True

    arm_dims: int = 7

    def __post_init__(self):
        valid_reprs = ("axis_angle", "quaternion", "rotation_6d", "matrix", "euler_xyz")
        canon = _canonical_tcp_pose_rotation_repr(self.rotation_repr)
        if canon not in valid_reprs:
            raise ValueError(
                f"rotation_repr must be one of {valid_reprs} or 'euler', got {self.rotation_repr!r}"
            )
        if self.rotation_repr != canon:
            object.__setattr__(self, "rotation_repr", canon)

    def __call__(self, data: DataDict) -> DataDict:
        if "actions" not in data or not self.enabled:
            return data

        left_data = {
            "state":   data["state"][..., :self.arm_dims],
            "actions": data["actions"][..., :self.arm_dims],
        }
        right_data = {
            "state":   data["state"][..., self.arm_dims:2 * self.arm_dims],
            "actions": data["actions"][..., self.arm_dims:2 * self.arm_dims],
        }

        abs_fn = AbsoluteTCPPoseActions(pos_dims=self.pos_dims, rotation_repr=self.rotation_repr)
        left_data  = abs_fn(left_data)
        right_data = abs_fn(right_data)

        data["actions"] = np.concatenate([left_data["actions"], right_data["actions"]], axis=-1)
        return data


@dataclasses.dataclass(frozen=True)
class ZeroArmActionDeltas(DataTransformFn):
    """Zero out the delta actions for specified arm(s) in a dual-arm setup.

    Use this when an arm is intentionally stationary during a task.
    Sensor noise in recorded data can produce tiny non-zero deltas that,
    after normalization with a very small std, cause large loss spikes.
    Zeroing prevents the model from learning to reproduce sensor noise artifacts,
    which would otherwise cause arm drift during deployment.

    The arm layout is assumed to be:
        actions[..., 0:arm_dims]            → left arm
        actions[..., arm_dims:2*arm_dims]   → right arm
    """

    arm_dims: int = 7
    zero_left: bool = False
    zero_right: bool = False

    def __call__(self, data: DataDict) -> DataDict:
        if "actions" not in data:
            return data
        actions = data["actions"].copy()
        if self.zero_left:
            actions[..., : self.arm_dims] = 0.0
        if self.zero_right:
            actions[..., self.arm_dims : 2 * self.arm_dims] = 0.0
        data["actions"] = actions
        return data


@dataclasses.dataclass(frozen=True)
class TokenizePrompt(DataTransformFn):
    tokenizer: _tokenizer.PaligemmaTokenizer
    discrete_state_input: bool = False

    def __call__(self, data: DataDict) -> DataDict:
        if (prompt := data.pop("prompt", None)) is None:
            raise ValueError("Prompt is required")

        if self.discrete_state_input:
            if (state := data.get("state", None)) is None:
                raise ValueError("State is required.")
        else:
            state = None

        if not isinstance(prompt, str):
            prompt = prompt.item()

        tokens, token_masks = self.tokenizer.tokenize(prompt, state)
        return {**data, "tokenized_prompt": tokens, "tokenized_prompt_mask": token_masks}


@dataclasses.dataclass(frozen=True)
class TokenizeFASTInputs(DataTransformFn):
    tokenizer: _tokenizer.FASTTokenizer

    def __call__(self, data: DataDict) -> DataDict:
        if (prompt := data.pop("prompt", None)) is None:
            raise ValueError("Prompt is required")

        if not isinstance(prompt, str):
            prompt = prompt.item()

        state, actions = data["state"], data.get("actions")
        tokens, token_mask, ar_mask, loss_mask = self.tokenizer.tokenize(prompt, state, actions)
        return {
            **data,
            "tokenized_prompt": tokens,
            "tokenized_prompt_mask": token_mask,
            "token_ar_mask": ar_mask,
            "token_loss_mask": loss_mask,
        }


@dataclasses.dataclass(frozen=True)
class ExtractFASTActions(DataTransformFn):
    tokenizer: _tokenizer.FASTTokenizer
    action_horizon: int
    action_dim: int

    def __call__(self, data: DataDict) -> DataDict:
        if "actions" not in data:
            return data
        # Model outputs are saved in "actions", but for FAST models they represent tokens.
        tokens = data.pop("actions")
        actions = self.tokenizer.extract_actions(tokens.astype(np.int32), self.action_horizon, self.action_dim)
        return {
            **data,
            "actions": actions,
        }


@dataclasses.dataclass(frozen=True)
class PromptFromLeRobotTask(DataTransformFn):
    """Extracts a prompt from the current LeRobot dataset task.

    Supports both single-dataset and multi-dataset training:
    - Single dataset: ``tasks`` is a mapping ``{task_index -> prompt}``.
    - Multi-dataset (``_LocalMultiDataset`` / ``MultiLeRobotDataset``): ``tasks`` is a list of
      such mappings, one per constituent dataset; the sample's ``dataset_index`` selects which
      mapping to use. This correctly handles overlapping task_index values across datasets.
    """

    # Either a single task mapping or one mapping per dataset.
    tasks: dict[int, str] | list[dict[int, str]]

    def __call__(self, data: DataDict) -> DataDict:
        if "task_index" not in data:
            raise ValueError('Cannot extract prompt without "task_index"')

        task_index = int(data["task_index"])

        if isinstance(self.tasks, list):
            if "dataset_index" not in data:
                raise ValueError('Cannot extract prompt for multi-dataset without "dataset_index"')
            dataset_index = int(data["dataset_index"])
            if dataset_index < 0 or dataset_index >= len(self.tasks):
                raise ValueError(f"{dataset_index=} out of range for tasks list of size {len(self.tasks)}")
            dataset_tasks = self.tasks[dataset_index]
        else:
            dataset_tasks = self.tasks

        if (prompt := dataset_tasks.get(task_index)) is None:
            raise ValueError(f"{task_index=} not found in task mapping: {dataset_tasks}")

        return {**data, "prompt": prompt}


@dataclasses.dataclass(frozen=True)
class PadStatesAndActions(DataTransformFn):
    """Zero-pads states and actions to the model action dimension."""

    model_action_dim: int

    def __call__(self, data: DataDict) -> DataDict:
        data["state"] = pad_to_dim(data["state"], self.model_action_dim, axis=-1)
        if "actions" in data:
            data["actions"] = pad_to_dim(data["actions"], self.model_action_dim, axis=-1)
        return data


def flatten_dict(tree: at.PyTree) -> dict:
    """Flatten a nested dictionary. Uses '/' as the separator."""
    return traverse_util.flatten_dict(tree, sep="/")


def unflatten_dict(tree: dict) -> at.PyTree:
    """Unflatten a flattened dictionary. Assumes that '/' was used as a separator."""
    return traverse_util.unflatten_dict(tree, sep="/")


def transform_dict(patterns: Mapping[str, str | None], tree: at.PyTree) -> at.PyTree:
    """Transform the structure of a nested dictionary using a set of patterns.

    The transformation is defined using the `patterns` dictionary. The keys are the
    input keys that should be matched and the values are the new names inside the output
    dictionary. If the value is None, the input key is removed.

    Both keys and values should represent flattened paths using '/' as the separator.
    Keys can be regular expressions and values can include backreferences to the
    matched groups (see `re.sub` for more details). Note that the regular expression
    must match the entire key.

    The order inside the `patterns` dictionary is important. Only the first pattern that
    matches the input key will be used.

    See unit tests for more examples.

    Args:
        patterns: A mapping from old keys to new keys.
        tree: The nested dictionary to transform.

    Returns:
        The transformed nested dictionary.
    """
    data = flatten_dict(tree)

    # Compile the patterns.
    compiled = {re.compile(k): v for k, v in patterns.items()}

    output = {}
    for k in data:
        for pattern, repl in compiled.items():
            if pattern.fullmatch(k):
                new_k = pattern.sub(repl, k, count=1) if repl is not None else None
                break
        else:
            # Use the original key if no match is found.
            new_k = k

        if new_k is not None:
            if new_k in output:
                raise ValueError(f"Key '{new_k}' already exists in output")
            output[new_k] = data[k]

    # Validate the output structure to make sure that it can be unflattened.
    names = sorted(output)
    for i in range(len(names) - 1):
        name, next_name = names[i : i + 2]
        if next_name.startswith(name + "/"):
            raise ValueError(f"Leaf '{name}' aliases a node of '{next_name}'")

    return unflatten_dict(output)


def apply_tree(
    tree: at.PyTree[T], selector: at.PyTree[S], fn: Callable[[T, S], T], *, strict: bool = False
) -> at.PyTree[T]:
    tree = flatten_dict(tree)
    selector = flatten_dict(selector)

    def transform(k: str, v: T) -> T:
        if k in selector:
            return fn(v, selector[k])
        return v

    if strict:
        for k in selector:
            if k not in tree:
                raise ValueError(f"Selector key {k} not found in tree")

    return unflatten_dict({k: transform(k, v) for k, v in tree.items()})


def pad_to_dim(x: np.ndarray, target_dim: int, axis: int = -1, value: float = 0.0) -> np.ndarray:
    """Pad an array to the target dimension with zeros along the specified axis."""
    current_dim = x.shape[axis]
    if current_dim < target_dim:
        pad_width = [(0, 0)] * len(x.shape)
        pad_width[axis] = (0, target_dim - current_dim)
        return np.pad(x, pad_width, constant_values=value)
    return x


def make_bool_mask(*dims: int) -> tuple[bool, ...]:
    """Make a boolean mask for the given dimensions.

    Example:
        make_bool_mask(2, -2, 2) == (True, True, False, False, True, True)
        make_bool_mask(2, 0, 2) == (True, True, True, True)

    Args:
        dims: The dimensions to make the mask for.

    Returns:
        A tuple of booleans.
    """
    result = []
    for dim in dims:
        if dim > 0:
            result.extend([True] * (dim))
        else:
            result.extend([False] * (-dim))
    return tuple(result)


def _assert_quantile_stats(norm_stats: at.PyTree[NormStats]) -> None:
    for k, v in flatten_dict(norm_stats).items():
        if v.q01 is None or v.q99 is None:
            raise ValueError(
                f"quantile stats must be provided if use_quantile_norm is True. Key {k} is missing q01 or q99."
            )
