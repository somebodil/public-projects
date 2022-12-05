from dataclasses import dataclass, field
from typing import Optional, List

from transformers import TrainingArguments


@dataclass
class OurTrainingArguments(TrainingArguments):
    STRATEGY = 'steps'
    STRATEGY_STEPS = 250

    # Trainer Arguments --
    output_dir: str = field(default='./')
    overwrite_output_dir: bool = field(default=True)

    evaluation_strategy: str = field(default=STRATEGY)
    eval_steps: int = field(default=STRATEGY_STEPS)
    save_strategy: str = field(default=STRATEGY)
    save_steps: int = field(default=STRATEGY_STEPS)
    save_total_limit: int = field(default=2)
    logging_strategy: str = field(default=STRATEGY)
    logging_steps: int = field(default=STRATEGY_STEPS)
    load_best_model_at_end: bool = field(default=True)
    metric_for_best_model: str = field(default='eval_stsb_spearman')

    num_train_epochs: int = field(default=1)
    per_device_train_batch_size: int = field(default=32)

    seed: int = field(default=42)
    fp16: bool = field(default=True)

    learning_rate: float = field(default=4e-5)

    ray_scope: Optional[str] = field(default="all")

    # Trainer Arguments (WANT TO REMOVE) --

    do_train: bool = field(default=True)
    do_eval: bool = field(default=True)
    eval_transfer: bool = field(default=False)

    # Ray Tune Arguments --
    use_ray: bool = field(default=False)
    local_dir: str = field(default='./ray_results/')
    num_samples: int = field(default=1)  # Will be ignored because "tune.grid_search" is used
    metric_direction: str = field(default='maximize')  # Should be 'maximize' or 'minimize'
    max_concurrent_trials: int = field(default=0)
    cpus_per_trial: int = field(default=1)
    gpus_per_trial: int = field(default=1)
    tune_choice_seed: List[int] = field(
        default_factory=lambda: [
            42, 43, 44, 45
        ])
