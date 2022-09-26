import logging
from dataclasses import dataclass, field
from typing import Union, List, Dict

import torch
import transformers
from datasets import load_dataset
from prettytable import PrettyTable
from transformers import (
    HfArgumentParser,
    TrainingArguments,
    set_seed,
    BertTokenizer, BertConfig, PreTrainedTokenizerBase, EarlyStoppingCallback
)

import util
from rugcse.models import BertForCL, RobertaForCL
from rugcse.trainers import CLTrainer

logger = logging.getLogger(__name__)


def print_table(task_names, scores):
    tb = PrettyTable()
    tb.field_names = task_names
    tb.add_row(scores)
    print(tb)


def print_senteval_results(results):
    task_names = []
    scores = []
    for task in ['STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'STSBenchmark', 'SICKRelatedness']:
        task_names.append(task)
        if task in results:
            if task in ['STS12', 'STS13', 'STS14', 'STS15', 'STS16']:
                scores.append('%.2f' % (results[task]['all']['spearman']['all'] * 100))
            else:
                scores.append('%.2f' % (results[task]['test']['spearman'].correlation * 100))
        else:
            scores.append('0.00')

    task_names.append('Avg.')
    scores.append('%.2f' % (sum([float(score) for score in scores]) / len(scores)))
    print_table(task_names, scores)


def main():
    # Logger & Parser --
    util.log_init(log_level=logging.INFO)
    parser = HfArgumentParser(TrainingArguments)
    training_args, unused_args = parser.parse_args_into_dataclasses(return_remaining_strings=True)
    util.log_args(training_args, unused_args)

    # Seed --
    set_seed(training_args.seed)

    # Prepare tokenizer, dataset (+ dataloader), model, loss function, optimizer, etc --
    tokenizer = BertTokenizer.from_pretrained(training_args.model_name_or_path)

    datasets = load_dataset(
        'text',
        data_files={'train': training_args.train_file},
        split='train'
    )
    column_names = datasets.column_names
    column_name = column_names[0]  # The only column name in unsup dataset

    def preprocess_function(examples):
        total = len(examples[column_name])  # Total len
        copied = examples[column_name] + examples[column_name]  # Repeat itself
        tokenized = tokenizer(copied, truncation=True, max_length=training_args.max_seq_length)

        result = {}
        for key in tokenized:
            result[key] = [[tokenized[key][i], tokenized[key][i + total]] for i in range(total)]

        return result

    train_dataset = datasets.map(
        preprocess_function,
        batched=True,
        remove_columns=column_names,
    )

    config = BertConfig.from_pretrained(training_args.model_name_or_path)

    if 'roberta' in training_args.model_name_or_path:
        model = RobertaForCL.from_pretrained(
            training_args.model_name_or_path,
            config=config,
            temperature=training_args.temperature,
            hard_negative_weight=training_args.hard_negative_weight,
            pooler_type=training_args.pooler_type,
            mlp_only_train=training_args.mlp_only_train,
        )
    elif 'bert' in training_args.model_name_or_path:
        model = BertForCL.from_pretrained(
            training_args.model_name_or_path,
            config=config,
            temperature=training_args.temperature,
            hard_negative_weight=training_args.hard_negative_weight,
            pooler_type=training_args.pooler_type,
            mlp_only_train=training_args.mlp_only_train,
        )
    else:
        raise NotImplementedError

    # Custom Data collator, because of data repeating in preprocess_function
    @dataclass
    class DataCollatorWithPadding:
        tokenizer: PreTrainedTokenizerBase
        padding = True

        def __call__(
                self, features: List[Dict[str, Union[List[int], List[List[int]], torch.Tensor]]]
        ) -> Dict[str, torch.Tensor]:

            special_keys = ['input_ids', 'attention_mask', 'token_type_ids']

            bs = len(features)
            if bs == 0:
                raise ValueError('Dataset is empty')

            num_sent = len(features[0]['input_ids'])

            # flat
            flat_features = []
            for feature in features:
                for i in range(num_sent):
                    flat_features.append({k: feature[k][i] if k in special_keys else feature[k] for k in feature})

            batch = self.tokenizer.pad(flat_features, padding=self.padding, return_tensors='pt')

            # un-flat
            batch = {
                k: batch[k].view(bs, num_sent, -1) if k in special_keys else batch[k].view(bs, num_sent, -1)[:, 0]
                for k in batch
            }

            if 'label' in batch:
                batch['labels'] = batch['label']
                del batch['label']
            if 'label_ids' in batch:
                batch['labels'] = batch['label_ids']
                del batch['label_ids']

            return batch

    data_collator = DataCollatorWithPadding(tokenizer)

    trainer = CLTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]
    )

    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    trainer.save_model()
    trainer.save_state()

    # Run sts tasks w/ best model & Print results (For test-set only, not dev)
    senteval_results = trainer.evaluate(while_training=False)
    print_senteval_results(senteval_results)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    STRATEGY = 'steps'
    STRATEGY_STEPS = 125

    # Trainer Arguments --
    output_dir: str = field(default='./output_dir')
    overwrite_output_dir: bool = field(default=True)

    evaluation_strategy: str = field(default=STRATEGY)
    eval_steps: int = field(default=STRATEGY_STEPS)
    save_strategy: str = field(default=STRATEGY)
    save_steps: int = field(default=STRATEGY_STEPS)
    save_total_limit: int = field(default=2)
    logging_strategy: str = field(default=STRATEGY)
    logging_first_step: bool = field(default=True)
    logging_steps: int = field(default=STRATEGY_STEPS)
    load_best_model_at_end: bool = field(default=True)
    metric_for_best_model: str = field(default='stsb_spearman')
    report_to: str = field(default='tensorboard')

    num_train_epochs: int = field(default=1)
    per_device_train_batch_size: int = field(default=64)
    per_device_eval_batch_size: int = field(default=64)

    learning_rate: float = field(default=3e-5)

    fp16: bool = field(default=True)

    # Non-Trainer Arguments --
    model_name_or_path: str = field(default='bert-base-uncased')
    max_seq_length: int = field(default=32)

    train_file: str = field(default='')  # depends on task_mode
    pooler_type: str = field(default='cls')
    mlp_only_train: bool = field(default=True)

    temperature: float = field(default=0.05)
    hard_negative_weight: float = field(default=0)

    # Modes --
    MODE_DATA_FULL = 'full-data'
    MODE_DATA_SAMPLE = 'sample-data'

    task_mode: List[str] = field(
        default_factory=lambda: [
            TrainingArguments.MODE_DATA_SAMPLE,
        ]
    )

    def __post_init__(self):
        super().__post_init__()

        if TrainingArguments.MODE_DATA_SAMPLE in self.task_mode:
            self.train_file = './data/wiki1m_for_simcse.sample.txt'
        elif TrainingArguments.MODE_DATA_FULL in self.task_mode:
            self.train_file = './data/wiki1m_for_simcse.txt'
        else:
            raise ValueError


if __name__ == '__main__':
    main()
