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
"""PyTorch BERT model extensions from LHAMa """

import logging
import torch
from torch import nn

from .modeling_bert import (BertModel, BertForQuestionAnswering)


logger = logging.getLogger(__name__)


class LHAMaCnnBertForQuestionAnswering(BertForQuestionAnswering):
    """
    Extension of the BERT for Question Answering model from HuggingFace.
    Rather than a single linear layer on top of BERT, this model uses a sequence
    of convolutional layers before outputting the start/stop output expected
    for the SQUaD task.
    """

    def __init__(self, config, freeze_weights=False):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)

        if(freeze_weights):
            logger.info('Freezing weights for LHAMa CNN')
            # Freeze the BERT weights, i.e. Feature Extraction to reduce training time
            for name, param in self.bert.named_parameters():                
                if name.startswith('embeddings'):
                    param.requires_grad = False
        else:
            logger.info('Fine-tuning for LHAMa CNN')
        
        self.qa_outputs = nn.Sequential(
                                          nn.Linear(config.hidden_size, config.hidden_size*2),
                                          nn.ReLU(),
                                          nn.Linear(config.hidden_size*2, config.hidden_size),
                                          nn.ReLU(),
                                          nn.Linear(config.hidden_size, config.num_labels),
                                       )

        self.init_weights()


class LHAMaLstmBertForQuestionAnswering(BertForQuestionAnswering):
    """
    Extension of the BERT for Question Answering model from HuggingFace.
    Rather than a single linear layer on top of BERT, this model uses a sequence
    of recurrent layers (LSTM) before outputting the start/stop output expected
    for the SQUaD task.
    """

    def __init__(self, config, freeze_weights=False):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)

        if(freeze_weights):
            logger.info('Freezing weights for LHAMa LSTM')
            # Freeze the BERT weights, i.e. Feature Extraction to reduce training time
            for name, param in self.bert.named_parameters():                
                if name.startswith('embeddings'):
                    param.requires_grad = False
        else:
            logger.info('Fine-tuning for LHAMa LSTM')
        
        self.qa_outputs = nn.Sequential(
                                          nn.Linear(config.hidden_size, config.hidden_size*2),
                                          nn.ReLU(),
                                          nn.Linear(config.hidden_size*2, config.hidden_size),
                                          nn.ReLU(),
                                          nn.Linear(config.hidden_size, config.num_labels),
                                       )

        self.init_weights()
