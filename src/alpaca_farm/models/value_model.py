# Copyright 2023 The Alpaca Team
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

from typing import Dict

import torch
import transformers
from torch import Tensor, nn
from transformers.utils.generic import ModelOutput

from .. import common


class ValueConfig(transformers.PretrainedConfig):
    model_type = "reward_model"

    # Huggingface doesn't allow non-kwargs for `__init__`.
    def __init__(self, backbone_model_name_or_path=None, **kwargs):
        super(ValueConfig, self).__init__(**kwargs)
        self.backbone_model_name_or_path = backbone_model_name_or_path
        self._name_or_path = backbone_model_name_or_path


class ValueModelOutput(ModelOutput):
    values: Tensor = None


class ValueModel(transformers.PreTrainedModel):
    config_class = ValueConfig

    def __init__(self, config: ValueConfig, **kwargs):
        super(ValueModel, self).__init__(config)
        self.backbone_model = common.make_generative_lm(config.backbone_model_name_or_path, **kwargs)
        hidden_size = common.get_transformer_hidden_size(self.backbone_model)
        reward_head = nn.Linear(hidden_size, 1) # keep the name so that we can load from reward ckpt
        torch.nn.init.zeros_(reward_head.bias)
        self.reward_head = reward_head.to(next(self.backbone_model.parameters()).device)

    # def forward(self, input_ids, attention_mask=None, return_dict=True, **kwargs):
    #     # We only compute the values and don't compute the logistic regression loss in this function so that it's
    #     # easier to use for later stages of reranking / RL training.
    #     outputs = self.backbone_model.model(
    #         input_ids=input_ids, attention_mask=attention_mask, return_dict=True, **kwargs
    #     )
    #     last_hidden_state = outputs.last_hidden_state
    #     last_hidden_state_at_the_end = last_hidden_state[:, -1, :]
    #     # TODO(lxuechen): Make returning values at all positions and last_hidden_state an option.
    #     values = self.value_head(last_hidden_state_at_the_end).squeeze(-1)
    #     return ValueModelOutput(values=values) if return_dict else (values,)

    def forward(self, queries: Tensor, query_attn_masks: Tensor, responses: Tensor) -> Dict[str, Tensor]:
        sequences = torch.cat([queries, responses], dim=1)
        sequence_attn_masks = sequences.ne(self.base_tokenizer.pad_token_id)

        inputs = self.backbone_model.prepare_inputs_for_generation(
            input_ids=sequences,
            attention_mask=sequence_attn_masks,
            use_cache=False,
        )
        outputs = self.backbone_model.model(**inputs, return_dict=True) # (B, Lq+Lr, V)
        # value[t]: \hat{V}(sequences_{:t-1}); must align with `_estimate_advantage`.
        last_hidden_state = outputs.last_hidden_state[:, queries.size(1) - 1 : -1] # (B, Lr, V)
        values = self.reward_head(last_hidden_state).squeeze(-1) # (B, Lr)
        return dict(values=values)

    def forward2(self, input_ids):
        '''
        input_ids: (B, Lq+Lr)
        return: values (B)
        '''
        attn_masks = input_ids.ne(self.base_tokenizer.pad_token_id)
        # inputs = self.base_model.prepare_inputs_for_generation(
        #     input_ids=input_ids,
        #     attention_mask=attn_masks,
        #     use_cache=False,
        # )
        outputs = self.backbone_model.model(input_ids=input_ids, attention_mask=attn_masks, return_dict=True) # (B, Lq+Lr, V)
        last_hidden_state = outputs.last_hidden_state[:, -1, :] # (B, V)
        values = self.reward_head(last_hidden_state).squeeze(-1) # (B)
        # print(input_ids, last_hidden_state, values)
        return values
