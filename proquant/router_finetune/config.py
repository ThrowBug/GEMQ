from dataclasses import dataclass


NORM_DISTILL_MODES = {
    "norm_ft": {
        "optimize_input_norm": True,
        "router_compensated": False,
    },
    "roco": {
        "optimize_input_norm": True,
        "router_compensated": True,
    },
}
RFT_TRAINERS = (
    "router_ft",
    "norm_ft",
    "roco",
)


@dataclass(frozen=True)
class DistillCEConfig:
    """Configuration for a joint teacher-soft-label distillation stage."""

    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    teacher_cache_dir: str
    rebuild_teacher_cache: bool

    @classmethod
    def from_args(cls, args):
        config = cls(
            epochs=args.rft_epochs,
            batch_size=args.rft_batch_size,
            learning_rate=args.rft_lr,
            weight_decay=args.rft_wd,
            teacher_cache_dir=args.rft_teacher_cache_dir,
            rebuild_teacher_cache=args.rft_rebuild_teacher_cache,
        )
        config.validate()
        return config

    def validate(self):
        if self.epochs <= 0:
            raise ValueError("--rft_epochs must be positive.")
        if self.batch_size <= 0:
            raise ValueError("--rft_batch_size must be positive.")
        if self.learning_rate <= 0.0:
            raise ValueError("--rft_lr must be positive.")
        if self.weight_decay < 0.0:
            raise ValueError("--rft_wd must be non-negative.")

    @property
    def needs_router_targets(self):
        return False

    @property
    def needs_output_targets(self):
        return True


