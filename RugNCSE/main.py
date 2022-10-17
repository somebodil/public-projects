import json
import logging
import os
import random
from dataclasses import dataclass
from typing import Union, List, Dict

import ray
import torch
from datasets import load_dataset
from prettytable import PrettyTable
from ray import tune
from ray.tune import CLIReporter, Callback
from ray.tune.schedulers import FIFOScheduler
from ray.tune.search import BasicVariantGenerator
from transformers import (
    HfArgumentParser,
    set_seed,
    BertTokenizer, BertConfig, PreTrainedTokenizerBase, EarlyStoppingCallback
)

import util
from rugncse.models import BertForCL, RobertaForCL
from rugncse.trainers import CLTrainer
from training_arguments import TrainingArguments

logger = logging.getLogger(__name__)


def format_senteval_results(results):
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

    tb = PrettyTable()
    tb.field_names = task_names
    tb.add_row(scores)
    return tb.__str__()


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

        sentences = examples[column_name] + examples[column_name]
        for i in range(2, training_args.num_augmentation):
            sentences += examples[column_name]

        tokenized = tokenizer(sentences, truncation=True, max_length=training_args.max_seq_length)

        result = {}
        for key in tokenized:  # Wrap
            result[key] = []
            for i in range(total):
                li = []
                for j in range(training_args.num_augmentation):
                    li_item = tokenized[key][i + j * total]
                    if j > 1:  # Because Anchor: j=0, Positive example: j=1
                        li_item = tokenized[key][i + j * total][1:-1]
                        random.shuffle(li_item)
                        li_item.insert(0, tokenized[key][i + j * total][0])
                        li_item.append(tokenized[key][i + j * total][-1])

                    li.append(li_item)
                result[key].append(li)
        return result

    train_dataset = datasets.map(
        preprocess_function,
        batched=True,
        remove_columns=column_names,
        num_proc=training_args.num_proc,
        load_from_cache_file=False,
    )

    config = BertConfig.from_pretrained(training_args.model_name_or_path)

    def model_init_impl(model_name_or_path=None):
        if not model_name_or_path:
            model_name_or_path = training_args.model_name_or_path

        if 'roberta' in training_args.model_name_or_path:
            return RobertaForCL.from_pretrained(
                model_name_or_path,
                config=config,
                training_args=training_args
            )
        elif 'bert' in training_args.model_name_or_path:
            return BertForCL.from_pretrained(
                model_name_or_path,
                config=config,
                training_args=training_args
            )
        else:
            raise NotImplementedError

    def model_init():
        return model_init_impl()

    model_init()  # for caching

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

    trainer = CLTrainer(
        model_init=model_init,
        args=training_args,
        data_collator=DataCollatorWithPadding(tokenizer),
        train_dataset=train_dataset,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]
    )

    if not training_args.use_ray:
        trainer.train()

    else:
        # Init
        ray.init(log_to_driver=False)

        # Callback code, to let ray-tune run trainer evaluate for test-set at the end of the trial
        class EndOfTrialCallback(Callback):
            def on_trial_complete(self, iteration: int, trials: List["Trial"], trial: "Trial", **info):
                # Save best model val-dataset accuracy
                with open(trial.logdir + os.sep + "trial_best_model_val_result.txt", "w") as fp:
                    # HF Trainer provided metric from "compute_objective" is saved always as "objective"
                    fp.write(str(trial.metric_analysis["objective"][training_args.metric_direction[:3]]))

                # Run and save best model for test-set accuracy
                checkpoints_dir = None
                for root, dirs, files in os.walk(trial.logdir):
                    for directory in dirs:
                        if directory == f"run-{trial.trial_id}":
                            checkpoints_dir = trial.logdir + os.sep + directory

                best_model_checkpoint = None
                if checkpoints_dir:
                    last_checkpoint = sorted(
                        os.listdir(checkpoints_dir),
                        key=lambda dirname: int(dirname.split('-')[1])
                    )[-1]

                    trainer_state_path = checkpoints_dir + os.sep + last_checkpoint + os.sep + "trainer_state.json"
                    with open(trainer_state_path) as fp:
                        trainer_state = json.load(fp)
                        best_model_checkpoint_tmp = trainer_state['best_model_checkpoint']
                        if "./" == best_model_checkpoint_tmp[:2]:
                            best_model_checkpoint_tmp = best_model_checkpoint_tmp[2:]

                        best_model_checkpoint = trial.logdir + os.sep + best_model_checkpoint_tmp

                if best_model_checkpoint:
                    local_trainer = CLTrainer(
                        model=model_init_impl(best_model_checkpoint),
                        args=training_args,
                        data_collator=DataCollatorWithPadding(tokenizer),
                        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]
                    )

                    with open(trial.logdir + os.sep + "trial_best_model_test_result.txt", "w") as fp:
                        fp.write(str(format_senteval_results(local_trainer.evaluate(while_training=False))))

        best_run = trainer.hyperparameter_search(
            backend="ray",
            hp_space=lambda _: {
                "seed": tune.grid_search(training_args.tune_list_seed),
                "learning_rate": tune.grid_search(training_args.tune_list_learning_rate),
                "alpha": tune.grid_search(training_args.tune_list_alpha)
            },
            compute_objective=lambda metrics: metrics[training_args.metric_for_best_model],
            n_trials=training_args.num_samples,
            direction=training_args.metric_direction,
            resources_per_trial={"cpu": training_args.cpus_per_trial, "gpu": training_args.gpus_per_trial},
            local_dir=training_args.local_dir,
            search_alg=BasicVariantGenerator(
                max_concurrent=training_args.max_concurrent_trials,
                constant_grid_search=True
            ),
            scheduler=FIFOScheduler(),
            progress_reporter=CLIReporter(  # Does not show best result
                metric_columns=[training_args.metric_for_best_model]
            ),
            log_to_file=True,
            callbacks=[EndOfTrialCallback()]
        )

        logger.info(f"best_run.run_id/best_run.hyperparameters: [{best_run.run_id}/{best_run.hyperparameters}]")


if __name__ == '__main__':
    main()
