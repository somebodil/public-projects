from dataclasses import dataclass, field
from typing import Optional, List

import transformers


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    STRATEGY = 'steps'
    STRATEGY_STEPS = 125

    # Trainer Arguments --
    output_dir: str = field(default='./')  # with ray-tune, we don't need this

    evaluation_strategy: str = field(default=STRATEGY)
    eval_steps: int = field(default=STRATEGY_STEPS)
    save_strategy: str = field(default=STRATEGY)
    save_steps: int = field(default=STRATEGY_STEPS)
    save_total_limit: int = field(default=2)
    logging_strategy: str = field(default=STRATEGY)
    logging_steps: int = field(default=STRATEGY_STEPS)
    load_best_model_at_end: bool = field(default=True)
    metric_for_best_model: str = field(default='eval_stsb_spearman')
    report_to: str = field(default='none')

    num_train_epochs: int = field(default=1)
    per_device_train_batch_size: int = field(default=64)
    per_device_eval_batch_size: int = field(default=64)

    learning_rate: float = field(default=3e-5)

    fp16: bool = field(default=False)

    ray_scope: Optional[str] = field(default="all")

    # Ray Tune Arguments --
    use_ray: bool = field(default=True)
    num_samples: int = field(default=1)
    metric_direction: str = field(default='maximize')  # Should be 'maximize' or 'minimize'
    max_concurrent_trials: int = field(default=0)
    cpus_per_trial: int = field(default=1)
    gpus_per_trial: int = field(default=1)
    tune_choice_learning_rate: List[float] = field(
        default_factory=lambda: [
            1e-3, 3e-3, 5e-3, 7e-3, 9e-3,
            1e-4, 3e-4, 5e-4, 7e-4, 9e-4,
            1e-5, 3e-5, 5e-5, 7e-5, 9e-5,
            1e-6, 3e-6, 5e-6, 7e-6, 9e-6,
        ]
    )
    tune_choice_seed: List[int] = field(
        default_factory=lambda: [
            42, 43, 44, 45, 46,
        ]
    )

    # Task Arguments --
    num_proc: int = field(default=2)

    model_name_or_path: str = field(default='bert-base-uncased')
    max_seq_length: int = field(default=32)

    train_file: str = field(default='./data/wiki1m_for_simcse.txt')
    pooler_type: str = field(default='cls')
    mlp_only_train: bool = field(default=True)

    temperature: float = field(default=0.05)

    num_augmentation: int = field(default=2)  # Should set more than 2
