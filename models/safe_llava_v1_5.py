import torch
import torch.nn as nn
from transformers import LlavaForConditionalGeneration
from typing import List, Optional, Tuple, Union
from dataclasses import dataclass

import torch
import torch.utils.checkpoint
from torch import nn

from transformers.activations import ACT2FN
from transformers.modeling_outputs import ModelOutput
from transformers.models.llava.configuration_llava import LlavaConfig


from .honeybee.loc_proj import CAbstractor

NUM_SAFE_TOKENS = 64
SAFE_HIDDEN_SIZE = 4096
SAFE_EMBED_DIM = 4096
num_safety_classes = 4
num_other_classes = 7

@dataclass
class LlavaCausalLMOutputWithPast(ModelOutput):
    """
    Base class for Llava causal language model (or autoregressive) outputs.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss (for next-token prediction).
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`)

            Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
            `past_key_values` input) to speed up sequential decoding.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        image_hidden_states (`tuple(torch.FloatTensor)`, *optional*):
            Tuple of `torch.FloatTensor` (one for the output of the image embeddings, `(batch_size, num_images,
            sequence_length, hidden_size)`.

            image_hidden_states of the model produced by the vision encoder, and optionally by the perceiver
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[List[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    image_hidden_states: Optional[Tuple[torch.FloatTensor]] = None


class LlavaMultiModalProjector(nn.Module):
    def __init__(self, config: LlavaConfig):
        super().__init__()

        self.linear_1 = nn.Linear(config.vision_config.hidden_size, config.text_config.hidden_size, bias=True)
        self.act = ACT2FN[config.projector_hidden_act]
        self.linear_2 = nn.Linear(config.text_config.hidden_size, config.text_config.hidden_size, bias=True)

    def forward(self, image_features):
        hidden_states = self.linear_1(image_features)
        hidden_states = self.act(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states


class HoneyBeeProjector(nn.Module):
    def __init__(self, config: LlavaConfig):
        super().__init__()
        config.num_eos_tokens = None
        config.pos_emb = False
        config.prenorm = False
        config.depth = 3
        config.mlp_depth = 2
        config.num_queries = 512 + 64
        # config.pos_emb = False
        config.encoder_hidden_size = config.vision_config.hidden_size
        config.hidden_size = config.vision_config.hidden_size + NUM_SAFE_TOKENS
        config.output_hidden_size = config.text_config.hidden_size
        self.proj = CAbstractor(config, config.vision_config.hidden_size + NUM_SAFE_TOKENS, config.text_config.hidden_size)

    def forward(self, image_features):
        hidden_states = self.proj(image_features)
        return hidden_states
    

class CrossAttentionClassifier(nn.Module):
    def __init__(self, text_feature_dim, num_safety_classes, num_other_classes):
        super().__init__()
        
        self.cross_attention = nn.MultiheadAttention(embed_dim=text_feature_dim, num_heads=8)
        
        self.safety_head = nn.Linear(text_feature_dim, num_safety_classes)
        self.other_head = nn.Linear(text_feature_dim, num_other_classes)
        
    def forward(self, text_features, image_features, safety_feature, attention_mask, safety_labels=None, other_labels=None):
        combined_features = torch.cat((safety_feature, image_features), dim=1)
        key_padding_mask = attention_mask == 0
        
        # Cross attention between text features and combined features
        #if text_features.dtype != combined_features.dtype:
        #    combined_features = combined_features.to(torch.bfloat16)
        # print(combined_features.dtype, text_features.dtype)
        attn_output, _ = self.cross_attention(query=text_features.permute(1, 0, 2),
                                              key=combined_features.permute(1, 0, 2),
                                              value=combined_features.permute(1, 0, 2))
                                              #key_padding_mask=expanded_key_padding_mask)
                                              
        #print(expanded_key_padding_mask.sum())
        safety_logits = self.safety_head(attn_output[0])
        other_logits = self.other_head(attn_output[0])
        
        loss = 0.0
        if safety_labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss_safety = loss_fct(safety_logits.view(-1, num_safety_classes), safety_labels.view(-1))
            loss += loss_safety
        
        if other_labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss_other = loss_fct(other_logits.view(-1, num_other_classes), other_labels.view(-1))
            loss += loss_other
        
        return loss, safety_logits, other_logits
    

class SafeLlavaModel(LlavaForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        # self.new_tokens = nn.Embedding(new_token_num, config.hidden_size)
        self.safety_query_tokens = nn.Parameter(
            torch.zeros(1, NUM_SAFE_TOKENS , SAFE_HIDDEN_SIZE), requires_grad = True
        )
        self.safety_query_tokens.data.normal_(mean=0.0, std=0.02)

        self.safety_query_tokens_2 = nn.Parameter(torch.zeros(1, NUM_SAFE_TOKENS , SAFE_HIDDEN_SIZE),\
                                                requires_grad = True)
        self.safety_query_tokens_2.data.normal_(mean=0.0, std=0.02)

        self.safe_mm_projector = HoneyBeeProjector(config)

        self.safe_part = CrossAttentionClassifier(SAFE_HIDDEN_SIZE, num_safety_classes, num_other_classes)
        #self.language_model.config._attn_implementation = 'flash'
        self.post_init()

    #def forward(self, input_ids, **kwargs):
        #outputs = super().forward(input_ids, **kwargs)
    def forward(
            self,
            input_ids: torch.LongTensor = None,
            pixel_values: torch.FloatTensor = None,
            return_cls_only: bool = False,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            vision_feature_layer: Optional[int] = None,
            vision_feature_select_strategy: Optional[str] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            safety_labels: Optional[torch.LongTensor] = None,
            rt_types: Optional[List[int]] = None,
            #img_paths: Optional[List[str]]  = None,
        ) -> Union[Tuple, LlavaCausalLMOutputWithPast]:        
        #return outputs
        #print(input_ids.shape)
        #print(pixel_values.shape)
        #print(safety_labels)
        #print(rt_types)
        #print(self.safety_query_tokens[0][31][1000:1005])
        #print(self.safety_query_tokens_2[0][31][1000:1005])
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        vision_feature_layer = (
            vision_feature_layer if vision_feature_layer is not None else self.config.vision_feature_layer
        )
        vision_feature_select_strategy = (
            vision_feature_select_strategy
            if vision_feature_select_strategy is not None
            else self.config.vision_feature_select_strategy
        )
        if inputs_embeds is None:
            # 1. Extra the input embeddings
            inputs_embeds = self.get_input_embeddings()(input_ids) #.to(torch.bfloat16)
            # 2. Merge text and images
            if pixel_values is not None and input_ids.shape[1] != 1:
                image_outputs = self.vision_tower(pixel_values, output_hidden_states=True)
                # this is not memory efficient at all (output_hidden_states=True) will save all the hidden stated.
                selected_image_feature = image_outputs.hidden_states[vision_feature_layer]
                if vision_feature_select_strategy == "default":
                    selected_image_feature = selected_image_feature[:, 1:]
                elif vision_feature_select_strategy == "full":
                    selected_image_feature = selected_image_feature
                else:
                    raise ValueError(
                        f"Unexpected select feature strategy: {self.config.vision_feature_select_strategy}"
                    )
                
                image_features = self.multi_modal_projector(selected_image_feature)
                local_safe_image_features = self.safe_mm_projector(selected_image_feature)
                #print(image_features.shape, local_safe_image_features.shape)
                #print(image_features.shape) # torch.Size([B, 576, 4096])
                #print(inputs_embeds.shape, attention_mask.shape, labels.shape, position_ids) 
                #safety_feature = self.safe_projection(self.safety_query_tokens) # (1,10,4096)
                #batch_size = image_features.size(0)
                #safety_feature = safety_query_tokens.expand(batch_size, -1, -1)
                
                # print(safety_feature.shape)
                
                #####################################################
                #####################################################
                ###### Safe cross attention + classification ########
                #
                batch_size = inputs_embeds.size(0)
                safety_feature = self.safety_query_tokens.expand(batch_size, -1, -1)
                
                loss_safe, safety_logits, other_logits = self.safe_part(inputs_embeds, local_safe_image_features, \
                                                                   safety_feature, attention_mask, safety_labels, rt_types)
                
                loss_safe *= 10
                if return_cls_only:
                    return loss_safe, safety_logits, other_logits
                #####################################################
                #####################################################
                #####################################################
                
                """
                inputs_embeds, attention_mask, labels, position_ids = self._merge_input_ids_with_image_features(
                    image_features, inputs_embeds, input_ids, attention_mask, labels
                )
                """
                batch_size = image_features.size(0)
                safety_feature_2 = self.safety_query_tokens_2.expand(batch_size, -1, -1).to(image_features.dtype)
                #print(safety_feature.dtype,image_features.shape,inputs_embeds.shape)
                #print(torch.cat((safety_feature, image_features), dim=1).dtype,inputs_embeds.dtype)
                #print(input_ids)
                inputs_embeds, attention_mask, labels, position_ids = self._merge_input_ids_with_image_features(
                    torch.cat((safety_feature_2, image_features), dim=1), inputs_embeds, input_ids, attention_mask, labels
                )
                if labels is None:
                    labels = torch.full_like(attention_mask, self.config.ignore_index).to(torch.long)
            else:
                # In case input_ids.shape[1] == 1 & pixel_values==None & past_key_values != None, we are in the case of
                # generation with cache
                if past_key_values is not None and pixel_values is not None and input_ids.shape[1] == 1:
                    # Retrieve the first layer to inspect the logits and mask out the hidden states
                    # that are set to 0
                    first_layer_past_key_value = past_key_values[0][0][:, :, :, 0]
                    # Sum all dimensions of head_dim (-2) to avoid random errors such as: https://github.com/huggingface/transformers/pull/28032#issuecomment-1863691941
                    batch_index, non_attended_tokens = torch.where(first_layer_past_key_value.float().sum(-2) == 0)
                    # Get the target length
                    target_seqlen = first_layer_past_key_value.shape[-1] + 1
                    extended_attention_mask = torch.ones(
                        (attention_mask.shape[0], target_seqlen - attention_mask.shape[1]),
                        dtype=attention_mask.dtype,
                        device=attention_mask.device,
                    )
                    # Filter out only the tokens that can be un-attended, this can happen
                    # if one uses Llava + Fused modules where the cache on the
                    # first iteration is already big enough, or if one passes custom cache
                    valid_indices = non_attended_tokens < extended_attention_mask.size(-1)
                    new_batch_index = batch_index[valid_indices]
                    new_non_attended_tokens = non_attended_tokens[valid_indices]
                    # Zero-out the places where we don't need to attend
                    extended_attention_mask[new_batch_index, new_non_attended_tokens] = 0
                    attention_mask = torch.cat((attention_mask, extended_attention_mask), dim=1)
                    position_ids = torch.sum(attention_mask, dim=1).unsqueeze(-1) - 1
        #print(attention_mask)
        #print(output_hidden_states)
        outputs = self.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        logits = outputs[0]
        # print(safety_labels)

        #print(combined_features.shape)
        #print(safety_labels.shape)
        #print(inputs_embeds.shape, input_ids.shape)
        #print(logits.shape)
        loss_llm = None
        if labels is not None:
            # Shift so that tokens < n predict n
            if attention_mask is not None:
                shift_attention_mask = attention_mask[..., 1:]
                shift_logits = logits[..., :-1, :][shift_attention_mask.to(logits.device) != 0].contiguous()
                shift_labels = labels[..., 1:][shift_attention_mask.to(labels.device) != 0].contiguous()
            else:
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            #print(shift_logits.view(-1, shift_logits.size(-1)).shape)
            #print(shift_labels.view(-1).shape)
            loss_llm = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1).to(shift_logits.device)
            )
        # print(shift_logits.view(-1, shift_logits.size(-1)).shape, shift_labels.view(-1).shape)
        # print(loss_cls, loss)
        # print(labels.shape,safety_labels.shape,combined_features.shape)
        if safety_labels is None:
            loss = loss_llm
        else:
            loss = loss_llm + loss_safe
            # print(loss_ll m, loss_safe)
        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output
        return LlavaCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )