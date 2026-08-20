from dataclasses import dataclass


ROUTER_LOSS_TYPES = ("kd", "kd_tail", "l2", "l2_center")
RFT_TIMINGS = ("after_all_quantization", "after_each_layer_quantization")
RFT_TRAINERS = ("legacy_ce", "distill_ce", "layerwise_teacher")


@dataclass(frozen=True)
class DistillCEConfig:
    """Configuration for joint router fine-tuning with teacher soft labels."""

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


@dataclass(frozen=True)
class RouterFinetuneConfig:
    """Validated configuration for teacher-guided router fine-tuning."""

    timing: str
    router_loss: str
    router_alpha: float
    router_loss_weight: float
    output_kl_weight: float
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    teacher_cache_dir: str
    rebuild_teacher_cache: bool

    @classmethod
    def from_args(cls, args):
        config = cls(
            timing=args.rft_timing,
            router_loss=args.rft_router_loss,
            router_alpha=args.rft_router_alpha,
            router_loss_weight=args.rft_router_loss_weight,
            output_kl_weight=args.rft_output_kl_weight,
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
        if self.timing not in RFT_TIMINGS:
            raise ValueError(f"Unsupported router fine-tuning timing: {self.timing}")
        if self.router_loss not in ROUTER_LOSS_TYPES:
            raise ValueError(f"Unsupported router loss: {self.router_loss}")
        if not 0.0 <= self.router_alpha <= 1.0:
            raise ValueError("--rft_router_alpha must be in [0, 1].")
        if self.router_loss_weight < 0.0 or self.output_kl_weight < 0.0:
            raise ValueError("Router and output loss weights must be non-negative.")
        if self.router_loss_weight == 0.0 and self.output_kl_weight == 0.0:
            raise ValueError("At least one router fine-tuning loss weight must be positive.")
        if self.epochs <= 0:
            raise ValueError("--rft_epochs must be positive.")
        if self.batch_size <= 0:
            raise ValueError("--rft_batch_size must be positive.")
        if self.learning_rate <= 0.0:
            raise ValueError("--rft_lr must be positive.")
        if self.weight_decay < 0.0:
            raise ValueError("--rft_wd must be non-negative.")
        if self.timing == "after_each_layer_quantization" and self.output_kl_weight > 0.0:
            raise ValueError(
                "Output KL is not supported with --rft_timing after_each_layer_quantization. "
                "The current GPTQ loop offloads every suffix layer, so a differentiable final-output "
                "forward would require a separate suffix dispatch implementation. Use "
                "after_all_quantization or set --rft_output_kl_weight 0."
            )

    @property
    def needs_router_targets(self):
        return self.router_loss_weight > 0.0

    @property
    def needs_output_targets(self):
        return self.output_kl_weight > 0.0

    @property
    def is_router_only(self):
        return self.output_kl_weight == 0.0

    def effective_top_m(self, num_experts, top_k):
        if not 0 < top_k <= num_experts:
            raise ValueError(f"Expected 0 < top_k <= num_experts, got {top_k} and {num_experts}.")
        extra = int(self.router_alpha * (num_experts - top_k) + 0.5)
        return min(num_experts, top_k + extra)
