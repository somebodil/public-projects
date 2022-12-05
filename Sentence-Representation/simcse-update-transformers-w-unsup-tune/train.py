import json
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional, Union, List, Dict, Tuple

import ray
import torch
import transformers
from datasets import load_dataset
from ray import tune
from ray.tune import Callback
from ray.tune.schedulers import FIFOScheduler
from ray.tune.search import BasicVariantGenerator
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_MASKED_LM_MAPPING,
    AutoConfig,
    AutoTokenizer,
    HfArgumentParser,
    default_data_collator,
    set_seed,
    BertForPreTraining
)
from transformers.tokenization_utils_base import PaddingStrategy, PreTrainedTokenizerBase
from transformers.trainer_utils import is_main_process

from core.models import RobertaForCL, BertForCL
from core.trainers import CLTrainer
from training_arguments import OurTrainingArguments

logger = logging.getLogger(__name__)
MODEL_CONFIG_CLASSES = list(MODEL_FOR_MASKED_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    # Huggingface's original arguments
    model_name_or_path: Optional[str] = field(
        default='prajjwal1/bert-small',
        metadata={
            "help": "The model checkpoint for weights initialization."
                    "Don't set if you want to train a model from scratch."
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
                    "with private models)."
        },
    )

    # SimCSE's arguments
    temp: float = field(
        default=0.05,
        metadata={
            "help": "Temperature for softmax."
        }
    )
    pooler_type: str = field(
        default="cls",
        metadata={
            "help": "What kind of pooler to use (cls, cls_before_pooler, avg, avg_top2, avg_first_last)."
        }
    )
    hard_negative_weight: float = field(
        default=0,
        metadata={
            "help": "The **logit** of weight for hard negatives (only effective if hard negatives are used)."
        }
    )
    do_mlm: bool = field(
        default=False,
        metadata={
            "help": "Whether to use MLM auxiliary objective."
        }
    )
    mlm_weight: float = field(
        default=0.1,
        metadata={
            "help": "Weight for MLM auxiliary objective (only effective if --do_mlm)."
        }
    )
    mlp_only_train: bool = field(
        default=True,
        metadata={
            "help": "Use MLP only during training"
        }
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    # Huggingface's original arguments. 
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=4,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )

    # SimCSE's arguments
    train_file: Optional[str] = field(
        default='data/wiki1m_for_simcse.txt',
        metadata={"help": "The training data file (.txt or .csv)."}
    )
    max_seq_length: Optional[int] = field(
        default=64,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
                    "than this will be truncated."
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
                    "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    mlm_probability: float = field(
        default=0.15,
        metadata={"help": "Ratio of tokens to mask for MLM (only effective if --do_mlm)"}
    )

    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`train_file` should be a csv, a json or a txt file."


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, OurTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if (
            os.path.exists(training_args.output_dir)
            and os.listdir(training_args.output_dir)
            and training_args.do_train
            and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty."
            "Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if is_main_process(training_args.local_rank) else logging.WARN,
    )

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column. You can easily tweak this
    # behavior (see below)
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    data_files = {}
    if data_args.train_file is not None:
        data_files["train"] = data_args.train_file
    extension = data_args.train_file.split(".")[-1]
    if extension == "txt":
        extension = "text"
    if extension == "csv":
        # YYH; removed cache_dir="./data/" because of hang
        datasets = load_dataset(
            extension, data_files=data_files, delimiter="\t" if "tsv" in data_args.train_file else ","
        )
    else:
        datasets = load_dataset(extension, data_files=data_files)  # YYH; removed cache_dir="./data/" because of hang

    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")

    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, **tokenizer_kwargs)
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, **tokenizer_kwargs)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    if model_args.model_name_or_path:
        if 'roberta' in model_args.model_name_or_path:
            model = RobertaForCL.from_pretrained(
                model_args.model_name_or_path,
                model_args,
                from_tf=bool(".ckpt" in model_args.model_name_or_path),
                config=config,
                cache_dir=model_args.cache_dir,
                revision=model_args.model_revision,
                use_auth_token=True if model_args.use_auth_token else None,
            )
        elif 'bert' in model_args.model_name_or_path:
            model = BertForCL.from_pretrained(
                model_args.model_name_or_path,
                model_args,
                from_tf=bool(".ckpt" in model_args.model_name_or_path),
                config=config,
                cache_dir=model_args.cache_dir,
                revision=model_args.model_revision,
                use_auth_token=True if model_args.use_auth_token else None,
            )
            if model_args.do_mlm:
                pretrained_model = BertForPreTraining.from_pretrained(model_args.model_name_or_path)
                model.lm_head.load_state_dict(pretrained_model.cls.predictions.state_dict())
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError

    model.resize_token_embeddings(len(tokenizer))

    # Prepare features
    column_names = datasets["train"].column_names
    sent2_cname = None
    if len(column_names) == 2:
        # Pair datasets
        sent0_cname = column_names[0]
        sent1_cname = column_names[1]
    elif len(column_names) == 3:
        # Pair datasets with hard negatives
        sent0_cname = column_names[0]
        sent1_cname = column_names[1]
        sent2_cname = column_names[2]
    elif len(column_names) == 1:
        # Unsupervised datasets
        sent0_cname = column_names[0]
        sent1_cname = column_names[0]
    else:
        raise NotImplementedError

    def prepare_features(examples):
        # padding = longest (default)
        #   If no sentence in the batch exceed the max length, then use
        #   the max sentence length in the batch, otherwise use the 
        #   max sentence length in the argument and truncate those that
        #   exceed the max length.
        # padding = max_length (when pad_to_max_length, for pressure test)
        #   All sentences are padded/truncated to data_args.max_seq_length.
        total = len(examples[sent0_cname])

        # Avoid "None" fields 
        for idx in range(total):
            if examples[sent0_cname][idx] is None:
                examples[sent0_cname][idx] = " "
            if examples[sent1_cname][idx] is None:
                examples[sent1_cname][idx] = " "

        sentences = examples[sent0_cname] + examples[sent1_cname]

        # If hard negative exists
        if sent2_cname is not None:
            for idx in range(total):
                if examples[sent2_cname][idx] is None:
                    examples[sent2_cname][idx] = " "
            sentences += examples[sent2_cname]

        sent_features = tokenizer(
            sentences,
            max_length=data_args.max_seq_length,
            truncation=True,
            padding="max_length" if data_args.pad_to_max_length else False,
        )

        features = {}
        if sent2_cname is not None:
            for key in sent_features:
                features[key] = [
                    [sent_features[key][i], sent_features[key][i + total], sent_features[key][i + total * 2]] for i in
                    range(total)]
        else:
            for key in sent_features:
                features[key] = [[sent_features[key][i], sent_features[key][i + total]] for i in range(total)]

        return features

    if training_args.do_train:
        train_dataset = datasets["train"].map(
            prepare_features,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
        )

    # Data collator
    @dataclass
    class OurDataCollatorWithPadding:

        tokenizer: PreTrainedTokenizerBase
        padding: Union[bool, str, PaddingStrategy] = True
        max_length: Optional[int] = None
        pad_to_multiple_of: Optional[int] = None
        mlm: bool = True
        mlm_probability: float = data_args.mlm_probability

        def __call__(self, features: List[Dict[str, Union[List[int], List[List[int]], torch.Tensor]]]) -> Dict[
            str, torch.Tensor]:
            special_keys = ['input_ids', 'attention_mask', 'token_type_ids', 'mlm_input_ids', 'mlm_labels']
            bs = len(features)
            if bs > 0:
                num_sent = len(features[0]['input_ids'])
            else:
                return
            flat_features = []
            for feature in features:
                for i in range(num_sent):
                    flat_features.append({k: feature[k][i] if k in special_keys else feature[k] for k in feature})

            batch = self.tokenizer.pad(
                flat_features,
                padding=self.padding,
                max_length=self.max_length,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_tensors="pt",
            )
            if model_args.do_mlm:
                batch["mlm_input_ids"], batch["mlm_labels"] = self.mask_tokens(batch["input_ids"])

            batch = {k: batch[k].view(bs, num_sent, -1) if k in special_keys else batch[k].view(bs, num_sent, -1)[:, 0]
                     for k in batch}

            if "label" in batch:
                batch["labels"] = batch["label"]
                del batch["label"]
            if "label_ids" in batch:
                batch["labels"] = batch["label_ids"]
                del batch["label_ids"]

            return batch

        def mask_tokens(
                self, inputs: torch.Tensor, special_tokens_mask: Optional[torch.Tensor] = None
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            """
            Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
            """
            inputs = inputs.clone()
            labels = inputs.clone()
            # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
            probability_matrix = torch.full(labels.shape, self.mlm_probability)
            if special_tokens_mask is None:
                special_tokens_mask = [
                    self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in
                    labels.tolist()
                ]
                special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
            else:
                special_tokens_mask = special_tokens_mask.bool()

            probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
            masked_indices = torch.bernoulli(probability_matrix).bool()
            labels[~masked_indices] = -100  # We only compute loss on masked tokens

            # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
            indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
            inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

            # 10% of the time, we replace masked input tokens with random word
            indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
            random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
            inputs[indices_random] = random_words[indices_random]

            # The rest of the time (10% of the time) we keep the masked input tokens unchanged
            return inputs, labels

    data_collator = default_data_collator if data_args.pad_to_max_length else OurDataCollatorWithPadding(tokenizer)

    # Training

    if not training_args.use_ray:
        trainer = CLTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset if training_args.do_train else None,
            tokenizer=tokenizer,
            data_collator=data_collator,
        )
        trainer.model_args = model_args

        if training_args.do_train:
            model_path = (
                model_args.model_name_or_path
                if (model_args.model_name_or_path is not None and os.path.isdir(model_args.model_name_or_path))
                else None
            )
            train_result = trainer.train(model_path=model_path)
            trainer.save_model()  # Saves the tokenizer too for easy upload

            output_train_file = os.path.join(training_args.output_dir, "train_results.txt")
            if trainer.is_world_process_zero():
                with open(output_train_file, "w") as writer:
                    logger.info("***** Train results *****")
                    for key, value in sorted(train_result.metrics.items()):
                        logger.info(f"  {key} = {value}")
                        writer.write(f"{key} = {value}\n")

                # Need to save the state, since Trainer.save_model saves only the tokenizer with the model
                trainer.state.save_to_json(os.path.join(training_args.output_dir, "trainer_state.json"))

        # Evaluation
        results = {}
        if training_args.do_eval:
            logger.info("*** Evaluate ***")
            results = trainer.evaluate(eval_senteval_transfer=True)

            output_eval_file = os.path.join(training_args.output_dir, "eval_results.txt")
            if trainer.is_world_process_zero():
                with open(output_eval_file, "w") as writer:
                    logger.info("***** Eval results *****")
                    for key, value in sorted(results.items()):
                        logger.info(f"  {key} = {value}")
                        writer.write(f"{key} = {value}\n")

        return results

    else:  # Assume training_args.do_train == true
        def model_init_impl(best_model_checkpoint=None):  # Assume only BERT is used
            return BertForCL.from_pretrained(
                model_args.model_name_or_path if best_model_checkpoint is None else best_model_checkpoint,
                model_args,
                from_tf=bool(".ckpt" in model_args.model_name_or_path),
                config=config,
                cache_dir=model_args.cache_dir,
                revision=model_args.model_revision,
                use_auth_token=True if model_args.use_auth_token else None,
            )

        def model_init():
            return model_init_impl()

        trainer = CLTrainer(
            model_init=model_init,
            args=training_args,
            train_dataset=train_dataset if training_args.do_train else None,
            tokenizer=tokenizer,
            data_collator=data_collator,
        )
        trainer.model_args = model_args

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
                        data_collator=data_collator
                    )

                    with open(trial.logdir + os.sep + "trial_best_model_test_result.json", "w") as fp:
                        fp.write(str(local_trainer.evaluate(eval_senteval_transfer=True)))

        best_run = trainer.hyperparameter_search(
            backend="ray",
            hp_space=lambda _: {
                "seed": tune.grid_search(training_args.tune_choice_seed),
            },
            compute_objective=lambda metrics: metrics[training_args.metric_for_best_model],
            n_trials=training_args.num_samples,
            direction=training_args.metric_direction,
            resources_per_trial={"cpu": training_args.cpus_per_trial, "gpu": training_args.gpus_per_trial},
            local_dir=training_args.local_dir,
            search_alg=BasicVariantGenerator(max_concurrent=training_args.max_concurrent_trials),
            scheduler=FIFOScheduler(),
            log_to_file=True,
            callbacks=[EndOfTrialCallback()]
        )

        logger.info(f"best_run.run_id/best_run.hyperparameters: [{best_run.run_id}/{best_run.hyperparameters}]")


if __name__ == "__main__":
    main()
