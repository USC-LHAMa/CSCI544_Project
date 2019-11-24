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
from torch.nn import CrossEntropyLoss

from .modeling_bert import (BertModel, BertForQuestionAnswering)


logger = logging.getLogger(__name__)


class LHAMaCnnBertForQuestionAnswering(BertForQuestionAnswering):
    """
    Extension of the BERT for Question Answering model from HuggingFace.
    Rather than a single linear layer on top of BERT, this model uses a sequence
    of convolutional layers before outputting the start/stop output expected
    for the SQuAD task.
    """

    def __init__(self, config, freeze_weights=False, kernel_size=3, padding=1):
        super().__init__(config)
        self.num_labels     = config.num_labels
        self.bert           = BertModel(config)
        self.freeze_weights = freeze_weights
        self.kernel_size    = kernel_size
        self.padding        = padding

        if(self.freeze_weights):
            logger.info('Freezing weights for LHAMa CNN')
            # Freeze the BERT weights, i.e. Feature Extraction to reduce training time
            for name, param in self.bert.named_parameters():                
                if name.startswith('embeddings'):
                    param.requires_grad = False
        else:
            logger.info('Fine-tuning for LHAMa CNN')
        
        self.qa_outputs = nn.Sequential(
                                          nn.Conv2d(config.hidden_size // 2, config.hidden_size, self.kernel_size, padding=self.padding),
                                          nn.MaxPool2d(5),
                                          nn.ReLU(),
                                          nn.Conv2d(config.hidden_size, config.hidden_size, self.kernel_size, padding=self.padding),
                                          nn.MaxPool2d(5),
                                          nn.ReLU(),
                                          nn.Linear(config.hidden_size, config.num_labels),
                                       )

        self.init_weights()


class LHAMaLstmBertForQuestionAnswering(BertForQuestionAnswering):
    """
    Extension of the BERT for Question Answering model from HuggingFace.
    Rather than a single linear layer on top of BERT, this model uses a sequence
    of recurrent layers (LSTM) before outputting the start/stop output expected
    for the SQuAD task.
    """

    def __init__(self, config, freeze_weights=False, num_layers=1):
        super().__init__(config)
        self.num_labels     = config.num_labels
        self.bert           = BertModel(config)
        self.freeze_weights = freeze_weights
        self.num_layers     = num_layers

        if(self.freeze_weights):
            logger.info('Freezing weights for LHAMa LSTM')
            # Freeze the BERT weights, i.e. Feature Extraction to reduce training time
            for name, param in self.bert.named_parameters():                
                if name.startswith('embeddings'):
                    param.requires_grad = False
        else:
            logger.info('Fine-tuning for LHAMa LSTM')
        
        self.lstm = nn.LSTM(batch_first=True, input_size=config.hidden_size,
                            hidden_size=config.hidden_size*2, num_layers=self.num_layers)
        self.linear = nn.Linear(config.hidden_size*2, config.num_labels)
        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, inputs_embeds=None,
                start_positions=None, end_positions=None):

        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask,
                            inputs_embeds=inputs_embeds)

        sequence_output = outputs[0]
        output, (h_n, c_n) = self.lstm(sequence_output)
        print('output: {}'.format(output.shape))
        print('h_n: {}'.format(h_n.shape))
        print('c_n: {}'.format(c_n.shape))
        # Get the output of the LSTM and send only the last layer to linear
        logits = self.linear(output[:,:,-2:])
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        print('logits: {}'.format(logits.shape))
        print('start_logits: {}'.format(start_logits.shape))
        print('end_logits: {}'.format(start_logits.shape))

        outputs = (start_logits, end_logits,) + outputs[2:]
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2
            outputs = (total_loss,) + outputs

        return outputs  # (loss), start_logits, end_logits, (hidden_states), (attentions)
