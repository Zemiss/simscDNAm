"""训练超参数及 JSON 配置读取。"""
from dataclasses import dataclass, asdict
from pathlib import Path
import json


@dataclass(frozen=True)
class TrainingConfig:
    SEED: int = 42
    TEST_FRACTION: float = 0.2
    INTERNAL_VALIDATION_FRACTION: float = 0.1
    LATENT_DIM: int = 32
    CONDITION_DIM: int = 32
    HIDDEN_DIM: int = 256
    DENOISER_BLOCKS: int = 3
    DIFFUSION_TIMESTEPS: int = 1000
    DDIM_SAMPLING_STEPS: int = 10
    DIFFUSION_LOSS_WEIGHT: float = 0.05
    DIFFUSION_PREDICTION: str = 'v'
    DIFFUSION_SCHEDULE: str = 'cosine'
    DROPOUT_RATE: float = 0.1
    BETA_TARGETS: tuple = (0.0001, 0.001, 0.1)
    FEATURE_BLOCK: int = 4096
    INPUT_ROW_BLOCK: int = 16
    GENERATION_ROW_BLOCK: int = 8
    TRAIN_BATCH_SIZE: int = 128
    VAE_PRETRAIN_EPOCHS: int = 80
    DIFFUSION_WARMUP_EPOCHS: int = 300
    JOINT_EPOCHS: int = 50
    JOINT_VAE_LR_FACTOR: float = 0.1
    SELECTION_INTERVAL: int = 5
    SELECTION_FEATURES: int = 1024
    DIFFUSION_PATIENCE: int = 40
    PATIENCE: int = 8
    EARLY_STOP_MIN_DELTA: float = 1e-05
    LEARNING_RATE: float = 0.001
    KL_WARMUP_EPOCHS: int = 10

    def __post_init__(self):
        positive = ("LATENT_DIM", "CONDITION_DIM", "HIDDEN_DIM", "DENOISER_BLOCKS",
                    "DIFFUSION_TIMESTEPS", "FEATURE_BLOCK", "INPUT_ROW_BLOCK",
                    "GENERATION_ROW_BLOCK", "TRAIN_BATCH_SIZE", "VAE_PRETRAIN_EPOCHS",
                    "DIFFUSION_WARMUP_EPOCHS", "JOINT_EPOCHS", "SELECTION_INTERVAL",
                    "SELECTION_FEATURES", "DIFFUSION_PATIENCE", "PATIENCE", "KL_WARMUP_EPOCHS")
        if any(not isinstance(getattr(self, key), int) or getattr(self, key) < 1 for key in positive):
            raise ValueError("维数、批大小和训练轮数必须是正整数")
        if not 1 <= self.DDIM_SAMPLING_STEPS <= self.DIFFUSION_TIMESTEPS:
            raise ValueError("DDIM 采样步数必须位于扩散日程范围内")
        if not self.BETA_TARGETS or any(beta <= 0 for beta in self.BETA_TARGETS):
            raise ValueError("至少需要一个正数 β 候选")
        if len(set(self.BETA_TARGETS)) != len(self.BETA_TARGETS):
            raise ValueError("β 候选不能重复")
        if not 0 < self.INTERNAL_VALIDATION_FRACTION < 1:
            raise ValueError("内部验证比例必须在 0 与 1 之间")
        if self.DIFFUSION_PREDICTION != "v" or self.DIFFUSION_SCHEDULE != "cosine":
            raise ValueError("模型使用 v 预测与 cosine 日程")

    @classmethod
    def from_json(cls, path):
        values = json.loads(Path(path).read_text(encoding="utf-8"))
        if "BETA_TARGETS" in values:
            values["BETA_TARGETS"] = tuple(values["BETA_TARGETS"])
        return cls(**values)

    def to_dict(self):
        return asdict(self)
