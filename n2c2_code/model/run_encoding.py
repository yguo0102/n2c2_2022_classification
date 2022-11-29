# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning the library models for sequence classification on GLUE (Bert, XLM, XLNet, RoBERTa, Albert, XLM-RoBERTa)."""


import dataclasses
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np

from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, EvalPrediction
from transformers import GlueDataTrainingArguments as DataTrainingArguments
from transformers import (
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
    LineByLineTextDataset,
    RobertaForSequenceClassification
)
from transformers.data.metrics import simple_accuracy, pearson_and_spearman
from sklearn.metrics import f1_score

from custom_dataset import CustomDataset
from custom_trainer import CustomTrainer

logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )


def conv_ex_args(remaining_args):
    ex_args = {}
    ex_args['train_file'] = remaining_args[1]
    ex_args['dev_file'] = remaining_args[3]
    ex_args['test_file'] = remaining_args[5]
    ex_args['metric'] = remaining_args[7]
    return ex_args

def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args, remaining_args = parser.parse_args_into_dataclasses(args=None, 
                return_remaining_strings=True, look_for_args_file=True)
        ex_args = conv_ex_args(remaining_args)


    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed
    set_seed(training_args.seed)

    # Get datasets
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )

    #file_path = ex_args['test_file']
    #test_dataset = LineByLineTextDataset(tokenizer=tokenizer, file_path=file_path, block_size=128)
    train_dataset = CustomDataset(data_args, ex_args, tokenizer=tokenizer)
    eval_dataset =  CustomDataset(data_args, ex_args, tokenizer=tokenizer, mode="dev")
    test_dataset =  CustomDataset(data_args, ex_args, tokenizer=tokenizer, mode="test")

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=len(train_dataset.get_labels()),
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
    )

    def compute_metrics(p: EvalPrediction) -> Dict:
        preds = np.argmax(p.predictions, axis=1)
        labels = p.label_ids
        metrics = {
            'acc' : simple_accuracy(preds, labels),
            'pearson' : pearson_and_spearman(preds, labels)['pearson'],
            'f1_macro_weighted' : f1_score(y_true=labels, y_pred=preds, average='weighted'),
            'f1_macro' : f1_score(y_true=labels, y_pred=preds, average='macro'),
            'f1_micro' : f1_score(y_true=labels, y_pred=preds, average='micro'),
            'pos_class_f1' : f1_score(y_true=labels, y_pred=preds) if ex_args['metric'] == 'pos_class_f1' else 0,
        }
        return metrics


    # Initialize our Trainer
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        eval_metric = ex_args['metric'],
    )

    trainer.output_emb(test_dataset=test_dataset, description='encoding',
            output_file=os.path.join(training_args.output_dir, ex_args['test_file']))


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
