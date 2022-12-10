#!/bin/bash

python evaluation.py \
  --pooler cls_before_pooler \
  "$@"
#  --model_name_or_path ./exp_result/my-unsup-core-bert-base-uncased \


