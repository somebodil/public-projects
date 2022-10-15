import os
import sys
from typing import List, Optional

import torch
from torch.utils.data.dataset import Dataset
from transformers import Trainer

# Absolute path to senteval
WORKING_DIR = os.path.dirname(os.path.abspath(__file__))
PATH_TO_SENTEVAL = WORKING_DIR + os.sep + '..' + os.sep + 'SentEval'
PATH_TO_SENTEVAL_DATA = WORKING_DIR + os.sep + '..' + os.sep + 'SentEval' + os.sep + 'data'
sys.path.insert(0, PATH_TO_SENTEVAL)
import senteval


class CLTrainer(Trainer):

    # Returns raw senteval result if while_training=False
    def evaluate(
            self,
            eval_dataset: Optional[Dataset] = None,
            ignore_keys: Optional[List[str]] = None,
            metric_key_prefix: str = 'eval',
            while_training: bool = True,
    ):

        # SentEval prepare and batcher
        def prepare(params, samples):
            return

        def batcher(params, batch):
            tokenizer = self.data_collator.tokenizer

            sentences = [' '.join(s) for s in batch]
            batch = tokenizer(
                sentences,
                return_tensors='pt',
                padding=True,
            )

            for k in batch:
                batch[k] = batch[k].to(self.args.device)

            with torch.no_grad():
                outputs = self.model(**batch, output_hidden_states=True, return_dict=True, sent_emb=True)
                pooler_output = outputs.pooler_output

            return pooler_output.cpu()

        # Set params for SentEval (fastmode)
        params = {
            'task_path': PATH_TO_SENTEVAL_DATA,
            'usepytorch': True,
            'kfold': 5,
            'classifier': {'nhid': 0, 'optim': 'rmsprop', 'batch_size': 128, 'tenacity': 3, 'epoch_size': 2}
        }

        se = senteval.engine.SE(params, batcher, prepare)
        if not while_training:
            tasks = ['STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'STSBenchmark', 'SICKRelatedness']  # All sts tasks
        else:
            tasks = ['STSBenchmark']

        self.model.eval()
        results = se.eval(tasks)

        if not while_training:
            return results
        else:
            stsb_spearman = results['STSBenchmark']['dev']['spearman'][0]
            align_loss = results['STSBenchmark']['dev']['align_loss']
            uniform_loss = results['STSBenchmark']['dev']['uniform_loss']

            metrics = {
                'eval_stsb_spearman': stsb_spearman,
                'eval_align_loss': align_loss,
                'eval_uniform_loss': uniform_loss,
            }

            self.log(metrics)
            return metrics
