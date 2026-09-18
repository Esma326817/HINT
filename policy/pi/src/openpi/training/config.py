"""HINT π training configs (attention / highlighting injection). See _CONFIGS."""

import abc
from collections.abc import Sequence
import dataclasses
import difflib
import logging
import pathlib
from typing import Any, Literal, Protocol, TypeAlias

import etils.epath as epath
import flax.nnx as nnx
from typing_extensions import override
import tyro

import openpi.models.model as _model
import openpi.models.pi0_config as pi0_config
import openpi.models.tokenizer as _tokenizer
import openpi.policies.piper_policy as piper_policy
import openpi.shared.download as _download
import openpi.shared.normalize as _normalize
import openpi.training.droid_rlds_dataset as droid_rlds_dataset
import openpi.training.optimizer as _optimizer
import openpi.training.weight_loaders as weight_loaders
import openpi.transforms as _transforms
ModelType: TypeAlias = _model.ModelType
# Work around a tyro issue with using nnx.filterlib.Filter directly.
Filter: TypeAlias = nnx.filterlib.Filter


@dataclasses.dataclass(frozen=True)
class AssetsConfig:
    """Determines the location of assets (e.g., norm stats) that will be used to set up the data pipeline.

    These assets will be replicated inside the checkpoint under the `assets/asset_id` directory.

    This can be used to load assets from a different checkpoint (e.g., base model checkpoint) or some other
    centralized location. For example, to load the norm stats for the Trossen robot from the base model checkpoint
    during fine-tuning, use:

    ```
    AssetsConfig(
        assets_dir="gs://openpi-assets/checkpoints/pi0_base/assets",
        asset_id="trossen",
    )
    ```
    """

    # Assets directory. If not provided, the config assets_dirs will be used. This is useful to load assets from
    # a different checkpoint (e.g., base model checkpoint) or some other centralized location.
    assets_dir: str | None = None

    # Asset id. If not provided, the repo id will be used. This allows users to reference assets that describe
    # different robot platforms.
    asset_id: str | None = None


@dataclasses.dataclass(frozen=True)
class DataConfig:
    # LeRobot repo id. If None, fake data will be created.
    # For single-dataset training, set this. For multi-dataset training, use ``repo_ids`` instead.
    repo_id: str | None = None
    # List of LeRobot repo ids for multi-dataset training. When set (non-empty), this takes precedence
    # over ``repo_id`` and a MultiLeRobotDataset (or per-dataset ConcatDataset) will be constructed.
    repo_ids: Sequence[str] | None = None
    # Optional sampling weights, one per entry in ``repo_ids``. When provided, a
    # WeightedRandomSampler is used so that dataset k is sampled with probability
    # ``dataset_weights[k] / sum(dataset_weights)`` regardless of its size.
    # If None (default), all datasets are sampled uniformly across frames
    # (i.e. probability proportional to dataset size, as in plain concatenation).
    dataset_weights: Sequence[float] | None = None
    # Directory within the assets directory containing the data assets.
    asset_id: str | None = None
    # Contains precomputed normalization stats. If None, normalization will not be performed.
    norm_stats: dict[str, _transforms.NormStats] | None = None

    # Used to adopt the inputs from a dataset specific format to a common format
    # which is expected by the data transforms.
    repack_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # Data transforms, typically include robot specific transformations. Will be applied
    # before the data is normalized. See `model.Observation` and `model.Actions` to learn about the
    # normalized data.
    data_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # Model specific transforms. Will be applied after the data is normalized.
    model_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # If true, will use quantile normalization. Otherwise, normal z-score normalization will be used.
    use_quantile_norm: bool = False

    # Names of keys that will be used by the data loader to generate the action sequence. The length of the
    # sequence is defined by the `action_horizon` field in the model config. This should be adjusted if your
    # LeRobot dataset is using different keys to represent the action.
    action_sequence_keys: Sequence[str] = ("actions",)

    # If true, will use the LeRobot dataset task to define the prompt.
    prompt_from_task: bool = False

    # Only used for RLDS data loader (ie currently only used for DROID).
    rlds_data_dir: str | None = None
    # Action space for DROID dataset.
    action_space: droid_rlds_dataset.DroidActionSpace | None = None
    # Path to the data filter file for DROID dataset
    filter_dict_path: str | None = None


class GroupFactory(Protocol):
    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        """Create a group."""


@dataclasses.dataclass(frozen=True)
class ModelTransformFactory(GroupFactory):
    """Creates model transforms for standard pi0 models."""

    # If provided, will determine the default prompt that be used by the model.
    default_prompt: str | None = None

    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        match model_config.model_type:
            case _model.ModelType.PI0:
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizePrompt(
                            _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                        ),
                        _transforms.PadStatesAndActions(model_config.action_dim),
                    ],
                )
            case _model.ModelType.PI05:
                assert isinstance(model_config, pi0_config.Pi0Config)
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizePrompt(
                            _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                            discrete_state_input=model_config.discrete_state_input,
                        ),
                        _transforms.PadStatesAndActions(model_config.action_dim),
                    ],
                )
            case _model.ModelType.PI0_FAST:
                tokenizer_cls = (
                    _tokenizer.FASTTokenizer
                    if model_config.fast_model_tokenizer is None
                    else model_config.fast_model_tokenizer
                )
                tokenizer_kwargs = (
                    {} if model_config.fast_model_tokenizer_kwargs is None else model_config.fast_model_tokenizer_kwargs
                )
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizeFASTInputs(
                            tokenizer_cls(model_config.max_token_len, **tokenizer_kwargs),
                        ),
                    ],
                    outputs=[
                        _transforms.ExtractFASTActions(
                            tokenizer_cls(model_config.max_token_len, **tokenizer_kwargs),
                            action_horizon=model_config.action_horizon,
                            action_dim=model_config.action_dim,
                        )
                    ],
                )


@dataclasses.dataclass(frozen=True)
class DataConfigFactory(abc.ABC):
    # The LeRobot repo id. Use this for single-dataset training.
    # Either ``repo_id`` or ``repo_ids`` (below) must be provided for non-fake configs.
    repo_id: str | None = None
    # Optional list of LeRobot repo ids for multi-dataset training.
    # If set (non-empty), this overrides ``repo_id`` and the data loader will build a
    # MultiLeRobotDataset that concatenates all datasets.
    repo_ids: Sequence[str] | None = None
    # Optional per-dataset sampling weights, one per entry in ``repo_ids``.
    # See ``DataConfig.dataset_weights`` for details.
    dataset_weights: Sequence[float] | None = None
    # Determines how the assets will be loaded.
    assets: AssetsConfig = dataclasses.field(default_factory=AssetsConfig)
    # Base config that will be updated by the factory.
    base_config: tyro.conf.Suppress[DataConfig | None] = None

    @abc.abstractmethod
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        """Create a data config."""

    def create_base_config(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repo_id = self.repo_id
        repo_ids = tuple(self.repo_ids) if self.repo_ids else None
        if repo_ids is not None and len(repo_ids) == 0:
            repo_ids = None
        # Validate dataset_weights.
        dataset_weights = tuple(self.dataset_weights) if self.dataset_weights is not None else None
        if dataset_weights is not None:
            if repo_ids is None:
                raise ValueError("`dataset_weights` was provided but `repo_ids` is empty. Set `repo_ids`.")
            if len(dataset_weights) != len(repo_ids):
                raise ValueError(
                    f"`dataset_weights` length ({len(dataset_weights)}) must match `repo_ids` length ({len(repo_ids)})."
                )
        # Prefer explicit asset_id; otherwise fall back to single repo_id or first of repo_ids.
        asset_id = self.assets.asset_id or repo_id or (repo_ids[0] if repo_ids else None)
        return dataclasses.replace(
            self.base_config or DataConfig(),
            repo_id=repo_id,
            repo_ids=repo_ids,
            dataset_weights=dataset_weights,
            asset_id=asset_id,
            norm_stats=self._load_norm_stats(epath.Path(self.assets.assets_dir or assets_dirs), asset_id),
            use_quantile_norm=model_config.model_type != ModelType.PI0,
        )

    def _load_norm_stats(self, assets_dir: epath.Path, asset_id: str | None) -> dict[str, _transforms.NormStats] | None:
        if asset_id is None:
            return None
        try:
            data_assets_dir = str(assets_dir / asset_id)
            norm_stats = _normalize.load(_download.maybe_download(data_assets_dir))
            logging.info(f"Loaded norm stats from {data_assets_dir}")
            return norm_stats
        except FileNotFoundError:
            logging.info(f"Norm stats not found in {data_assets_dir}, skipping.")
        return None


@dataclasses.dataclass(frozen=True)
class FakeDataConfig(DataConfigFactory):
    repo_id: str = "fake"

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        return DataConfig(repo_id=self.repo_id)


@dataclasses.dataclass(frozen=True)
class PiperDataConfig(DataConfigFactory):
    # If true, will convert joint dimensions to deltas with respect to the current state before passing to the model.
    # Gripper / ee_command dimensions will remain in absolute values (mask: 6 delta + 1 abs per arm).
    extra_delta_transform: bool = True
    # If provided, will be injected into the input data if the "prompt" key is not present.
    default_prompt: str | None = None
    # Optional per-camera source columns containing model-space ViT patch maps.
    semantic_grounding_keys: dict[str, str] | None = None
    # Preserve fractional wrist coverage while making small global targets reach
    # a peak grounding weight of one.
    global_semantic_grounding_max_normalize: bool = False

    # Repack transforms (LeRobot flat keys -> keys expected by PiperInputs).
    repack_transforms: tyro.conf.Suppress[_transforms.Group] = dataclasses.field(
        default=_transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/global": "images.global",
                        "observation/left_wrist": "images.left_wrist",
                        "observation/right_wrist": "images.right_wrist",
                        "observation/state": "state",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )
    )
    # Must match the action feature name in the LeRobot dataset (e.g. data_process/convertion/2arm_2grp_3cam.py uses "actions").
    action_sequence_keys: Sequence[str] = ("actions",)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        data_transforms = _transforms.Group(
            inputs=[
                piper_policy.PiperInputs(
                    model_type=model_config.model_type,
                    attention_injection=bool(
                        getattr(model_config, "attention_injection", False)
                        or getattr(model_config, "vit_attention_injection", False)
                    ),
                    global_semantic_grounding_max_normalize=self.global_semantic_grounding_max_normalize,
                )
            ],
            outputs=[piper_policy.PiperOutputs()],
        )
        if self.extra_delta_transform:
            delta_action_mask = _transforms.make_bool_mask(6, -1, 6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(model_config)

        repack_transforms = self.repack_transforms
        if self.semantic_grounding_keys is not None:
            if not repack_transforms.inputs or not isinstance(
                repack_transforms.inputs[0], _transforms.RepackTransform
            ):
                raise ValueError("semantic_grounding_keys requires Piper's standard RepackTransform")
            repack = repack_transforms.inputs[0]
            structure = dict(repack.structure)
            structure["semantic_grounding"] = {
                camera: self.semantic_grounding_keys[camera]
                for camera in ("global", "left_wrist", "right_wrist")
            }
            repack_transforms = _transforms.Group(
                inputs=(
                    _transforms.RepackTransform(structure),
                    *repack_transforms.inputs[1:],
                ),
                outputs=repack_transforms.outputs,
            )

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transforms,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=self.action_sequence_keys,
        )


@dataclasses.dataclass(frozen=True)
class TrainConfig:
    # Name of the config. Must be unique. Will be used to reference this config.
    name: tyro.conf.Suppress[str]
    # Project name.
    project_name: str = "openpi"
    # Experiment name. Will be used to name the metadata and checkpoint directories.
    exp_name: str = tyro.MISSING

    # Defines the model config. Some attributes (action_dim, action_horizon, and max_token_len) are shared by all models
    # -- see BaseModelConfig. Specific model implementations (e.g., Pi0Config) inherit from BaseModelConfig and may
    # define additional attributes.
    model: _model.BaseModelConfig = dataclasses.field(default_factory=pi0_config.Pi0Config)

    # A weight loader can optionally load (possibly partial) weights from disk after the model is initialized.
    weight_loader: weight_loaders.WeightLoader = dataclasses.field(default_factory=weight_loaders.NoOpWeightLoader)

    # Optional path to a PyTorch checkpoint to load weights from.
    pytorch_weight_path: str | None = None

    # Precision for PyTorch training.
    pytorch_training_precision: Literal["bfloat16", "float32"] = "bfloat16"

    lr_schedule: _optimizer.LRScheduleConfig = dataclasses.field(default_factory=_optimizer.CosineDecaySchedule)
    optimizer: _optimizer.OptimizerConfig = dataclasses.field(default_factory=_optimizer.AdamW)
    ema_decay: float | None = 0.99

    # Specifies which weights should be frozen.
    freeze_filter: tyro.conf.Suppress[Filter] = dataclasses.field(default_factory=nnx.Nothing)

    # Determines the data to be trained on.
    data: DataConfigFactory = dataclasses.field(default_factory=FakeDataConfig)

    # Base directory for config assets (e.g., norm stats).
    assets_base_dir: str = "./assets"
    # Base directory for checkpoints.
    checkpoint_base_dir: str = "./checkpoints"

    # Random seed that will be used by random generators during training.
    seed: int = 42
    # Global batch size.
    batch_size: int = 32
    # Number of workers to use for the data loader. Increasing this number will speed up data loading but
    # will increase memory and CPU usage.
    num_workers: int = 2
    # Number of train steps (batches) to run.
    num_train_steps: int = 30_000

    # How often (in steps) to log training metrics.
    log_interval: int = 100
    # How often (in steps) to save checkpoints.
    save_interval: int = 1000
    # If set, any existing checkpoints matching step % keep_period == 0 will not be deleted.
    keep_period: int | None = 5000

    # If true, will overwrite the checkpoint directory if it already exists.
    overwrite: bool = False
    # If true, will resume training from the last checkpoint.
    resume: bool = False

    # If true, will enable wandb logging.
    wandb_enabled: bool = True

    # Used to pass metadata to the policy server.
    policy_metadata: dict[str, Any] | None = None

    # If the value is greater than 1, FSDP will be enabled and shard across number of specified devices; overall
    # device memory will be reduced but training could potentially be slower.
    # eg. if total device is 4 and fsdp devices is 2; then the model will shard to 2 devices and run
    # data parallel between 2 groups of devices.
    fsdp_devices: int = 1

    @property
    def assets_dirs(self) -> pathlib.Path:
        """Get the assets directory for this config."""
        return (pathlib.Path(self.assets_base_dir) / self.name).resolve()

    @property
    def checkpoint_dir(self) -> pathlib.Path:
        """Get the checkpoint directory for this config."""
        if not self.exp_name:
            raise ValueError("--exp_name must be set")
        return (pathlib.Path(self.checkpoint_base_dir) / self.name / self.exp_name).resolve()

    @property
    def trainable_filter(self) -> nnx.filterlib.Filter:
        """Get the filter for the trainable parameters."""
        return nnx.All(nnx.Param, nnx.Not(self.freeze_filter))

    def __post_init__(self) -> None:
        if self.resume and self.overwrite:
            raise ValueError("Cannot resume and overwrite at the same time.")


# HINT Piper tasks use the paper's sort / spell / peg abbreviations (Table S1).
_CONFIGS = [
    TrainConfig(
        name="pi05_piper_peg_HINT",
        checkpoint_base_dir="/data/checkpoints/finetuned/pi05/piper",
        exp_name="v1",
        model=pi0_config.Pi0Config(
            pi05=True,
            vit_attention_injection=True,
            vit_attention_injection_layers=[6, 12, 14, 15, 16, 17, 18, 20, 24],
            vit_attention_alpha_mode="fixed",
            vit_attention_alpha=1.0,
            attention_injection=True,
            attention_injection_layers=[15, 16, 17, 18],
            attention_alpha_mode="fixed",
            attention_alpha=1.0,
        ),
        data=PiperDataConfig(
            repo_id="peg_in_hole/peg_in_hole_v4_5_both_merged",
            assets=AssetsConfig(asset_id="peg_in_hole/peg_in_hole_v4_5_both_merged"),
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=True,
            semantic_grounding_keys={
                "global": "global_semantic_grounding",
                "left_wrist": "left_wrist_semantic_grounding",
                "right_wrist": "right_wrist_semantic_grounding",
            },
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=5000,
            peak_lr=2e-5,
            decay_steps=50_000,
            decay_lr=2e-7,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1),
        ema_decay=0.999,
        weight_loader=weight_loaders.CheckpointWeightLoader("/data/checkpoints/base_models/pi05/pi05_base/params"),
        num_train_steps=80_000,
        batch_size=32,
        num_workers=32,
        save_interval=5000,
        fsdp_devices=2,
        keep_period=10000,
    ),

    TrainConfig(
        name="pi05_piper_spell_HINT",
        checkpoint_base_dir="/home/zjuhe/ckpt/finetuned/pi05/piper",
        exp_name="v1",
        model=pi0_config.Pi0Config(
            pi05=True,
            vit_attention_injection=True,
            vit_attention_injection_layers=[6, 12, 14, 15, 16, 17, 18, 20, 24],
            vit_attention_alpha_mode="fixed",
            vit_attention_alpha=1.0,
            attention_injection=True,
            attention_injection_layers=[15, 16, 17, 18],
            attention_alpha_mode="fixed",
            attention_alpha=1.0,
        ),
        data=PiperDataConfig(
            repo_id="/home/zjuhe/data/realworld/piper/piper_letter_v2_big_annotation_merged_224_both",
            assets=AssetsConfig(asset_id="piper_letter_v2_HINT"),
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=True,
            semantic_grounding_keys={
                "global": "global_semantic_grounding",
                "left_wrist": "left_wrist_semantic_grounding",
                "right_wrist": "right_wrist_semantic_grounding",
            },
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=5000,
            peak_lr=2e-5,
            decay_steps=50_000,
            decay_lr=2e-7,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1),
        ema_decay=0.999,
        weight_loader=weight_loaders.CheckpointWeightLoader("/home/zjuhe/ckpt/base_model/pi05/pi05_base/params"),
        num_train_steps=80_000,
        batch_size=32,
        num_workers=32,
        save_interval=5000,
        fsdp_devices=2,
        keep_period=10000,
    ),

    TrainConfig(
        name="pi05_piper_sort_mixed_HINT",
        checkpoint_base_dir="/home/zjuhe/ckpt/finetuned/pi05/piper",
        exp_name="v1",
        model=pi0_config.Pi0Config(
            pi05=True,
            vit_attention_injection=True,
            vit_attention_injection_layers=[6, 12, 14, 15, 16, 17, 18, 20, 24],
            vit_attention_alpha_mode="fixed",
            vit_attention_alpha=1.0,
            attention_injection=True,
            attention_injection_layers=[15, 16, 17, 18],
            attention_alpha_mode="fixed",
            attention_alpha=1.0,
        ),
        data=PiperDataConfig(
            repo_id="/home/zjuhe/data/realworld/piper/classify_mixed_both",
            assets=AssetsConfig(asset_id="classify_mixed_both"),
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=True,
            semantic_grounding_keys={
                "global": "global_semantic_grounding",
                "left_wrist": "left_wrist_semantic_grounding",
                "right_wrist": "right_wrist_semantic_grounding",
            },
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=5000,
            peak_lr=2e-5,
            decay_steps=50_000,
            decay_lr=2e-7,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1),
        ema_decay=0.999,
        weight_loader=weight_loaders.CheckpointWeightLoader("/home/zjuhe/ckpt/base_model/pi05/pi05_base/params"),
        num_train_steps=50_000,
        batch_size=32,
        num_workers=32,
        save_interval=5000,
        fsdp_devices=2,
        keep_period=10000,
    ),

    TrainConfig(
        name="pi05_piper_sort_HINT",
        checkpoint_base_dir="/home/zjuhe/ckpt/finetuned/pi05/piper",
        exp_name="v1",
        model=pi0_config.Pi0Config(
            pi05=True,
            vit_attention_injection=True,
            vit_attention_injection_layers=[6, 12, 14, 15, 16, 17, 18, 20, 24],
            vit_attention_alpha_mode="fixed",
            vit_attention_alpha=1.0,
            attention_injection=True,
            attention_injection_layers=[15, 16, 17, 18],
            attention_alpha_mode="fixed",
            attention_alpha=1.0,
        ),
        data=PiperDataConfig(
            repo_id="/home/zjuhe/data/realworld/piper/classify_merged_v1_v2_both",
            assets=AssetsConfig(asset_id="classify_merged_v1_v2_both"),
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=True,
            semantic_grounding_keys={
                "global": "global_semantic_grounding",
                "left_wrist": "left_wrist_semantic_grounding",
                "right_wrist": "right_wrist_semantic_grounding",
            },
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=5000,
            peak_lr=2e-5,
            decay_steps=50_000,
            decay_lr=2e-7,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1),
        ema_decay=0.999,
        weight_loader=weight_loaders.CheckpointWeightLoader("/home/zjuhe/ckpt/base_model/pi05/pi05_base/params"),
        num_train_steps=50_000,
        batch_size=32,
        num_workers=32,
        save_interval=5000,
        fsdp_devices=2,
        keep_period=10000,
    ),

    TrainConfig(
        name="debug",
        data=FakeDataConfig(),
        batch_size=2,
        model=pi0_config.Pi0Config(paligemma_variant="dummy", action_expert_variant="dummy"),
        save_interval=100,
        overwrite=True,
        exp_name="debug",
        num_train_steps=10,
        wandb_enabled=False,
    ),

    TrainConfig(
        name="debug_pi05",
        model=pi0_config.Pi0Config(pi05=True, paligemma_variant="dummy", action_expert_variant="dummy"),
        data=FakeDataConfig(),
        batch_size=2,
        num_train_steps=10,
        overwrite=True,
        exp_name="debug_pi05",
        wandb_enabled=False,
    ),
]

if len({config.name for config in _CONFIGS}) != len(_CONFIGS):
    raise ValueError("Config names must be unique.")
_CONFIGS_DICT = {config.name: config for config in _CONFIGS}


def cli() -> TrainConfig:
    return tyro.extras.overridable_config_cli({k: (k, v) for k, v in _CONFIGS_DICT.items()})


def get_config(config_name: str) -> TrainConfig:
    """Get a config by name."""
    if config_name not in _CONFIGS_DICT:
        closest = difflib.get_close_matches(config_name, _CONFIGS_DICT.keys(), n=1, cutoff=0.0)
        closest_str = f" Did you mean '{closest[0]}'? " if closest else ""
        raise ValueError(f"Config '{config_name}' not found.{closest_str}")

    return _CONFIGS_DICT[config_name]
