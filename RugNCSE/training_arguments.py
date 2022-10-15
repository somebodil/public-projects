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

    fp16: bool = field(default=True)

    ray_scope: Optional[str] = field(default="all")

    # Ray Tune Arguments --
    num_samples: int = field(default=1)
    metric_direction: str = field(default='maximize')  # Should be 'maximize' or 'minimize'
    max_concurrent_trials: int = field(default=0)
    cpus_per_trial: int = field(default=1)
    gpus_per_trial: int = field(default=0.25)
    tune_choice_seed: List[int] = field(
        default_factory=lambda: [
            42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68,
            69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95,
            96, 97, 98, 99
        ])

    # Task Arguments --
    num_proc: int = field(default=8)

    model_name_or_path: str = field(default='bert-base-uncased')
    max_seq_length: int = field(default=32)

    train_file: str = field(default='./data/wiki1m_for_simcse.sample.txt')
    pooler_type: str = field(default='cls')
    mlp_only_train: bool = field(default=True)

    temperature: float = field(default=0.05)
