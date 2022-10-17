import torch
import torch.nn as nn
from transformers.modeling_outputs import SequenceClassifierOutput, BaseModelOutputWithPoolingAndCrossAttentions
from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertModel
from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel, RobertaModel


class MLPLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = self.activation(x)
        return x


class Similarity(nn.Module):
    def __init__(self, temperature):
        super().__init__()
        self.temperature = temperature
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return self.cos(x, y) / self.temperature


class Pooler(nn.Module):
    """
    Parameter-free poolers to get the sentence embedding
    'cls': [CLS] representation with BERT/RoBERTa's MLP pooler.
    'cls_before_pooler': [CLS] representation without the original MLP pooler.
    'avg': average of the last layers' hidden states at each token.
    'avg_top2': average of the last two layers.
    'avg_first_last': average of the first and the last layers.
    """

    TYPE_FIRST_LAST = 'avg_first_last'
    TYPE_TOP2 = 'avg_top2'
    TYPE_AVG = 'avg'
    TYPE_CLS = 'cls'
    TYPE_ALL = [TYPE_CLS, TYPE_AVG, TYPE_TOP2, TYPE_FIRST_LAST]

    def __init__(self, pooler_type):
        super().__init__()
        self.pooler_type = pooler_type
        assert self.pooler_type in Pooler.TYPE_ALL, 'unrecognized pooling type %s' % self.pooler_type

    def forward(self, attention_mask, outputs):
        last_hidden = outputs.last_hidden_state
        hidden_states = outputs.hidden_states

        if self.pooler_type == Pooler.TYPE_CLS:
            return last_hidden[:, 0]
        elif self.pooler_type == Pooler.TYPE_AVG:
            return (last_hidden * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
        elif self.pooler_type == Pooler.TYPE_FIRST_LAST:
            a = hidden_states[0]
            b = hidden_states[-1]
            pooled_result = ((a + b) / 2.0 * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
            return pooled_result
        elif self.pooler_type == Pooler.TYPE_TOP2:
            a = hidden_states[-2]
            b = hidden_states[-1]
            pooled_result = ((a + b) / 2.0 * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
            return pooled_result
        else:
            raise NotImplementedError


def cl_forward(
        cls,
        encoder,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
):
    return_dict = return_dict if return_dict is not None else cls.config.use_return_dict

    batch_size = input_ids.size(0)
    num_sent = input_ids.size(1)

    # Flatten input for encoding
    input_ids = input_ids.view((-1, input_ids.size(-1)))  # (bs * num_sent, len)
    attention_mask = attention_mask.view((-1, attention_mask.size(-1)))  # (bs * num_sent len)
    if token_type_ids is not None:
        token_type_ids = token_type_ids.view((-1, token_type_ids.size(-1)))  # (bs * num_sent, len)

    # Get raw embeddings
    outputs = encoder(
        input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        position_ids=position_ids,
        head_mask=head_mask,
        inputs_embeds=inputs_embeds,
        output_attentions=output_attentions,
        output_hidden_states=True if cls.pooler_type in [Pooler.TYPE_TOP2, Pooler.TYPE_FIRST_LAST] else False,
        return_dict=True,
    )

    # Pooling
    pooler_output = cls.pooler(attention_mask, outputs)
    pooler_output = pooler_output.view((batch_size, num_sent, pooler_output.size(-1)))  # (bs, num_sent, hidden)

    # If using 'cls', we add an extra MLP layer
    # (same as BERT's original implementation) over the representation.
    if cls.pooler_type == Pooler.TYPE_CLS:
        pooler_output = cls.mlp(pooler_output)

    # Calculate loss
    z1, z2 = pooler_output[:, 0], pooler_output[:, 1]
    cos_sim = cls.sim(z1.unsqueeze(1), z2.unsqueeze(0))
    for i in range(2, pooler_output.shape[1]):
        zi = pooler_output[:, i]
        z1_zi_cos = cls.sim(z1.unsqueeze(1), zi.unsqueeze(0))
        z1_zi_cos *= cls.alpha
        cos_sim = torch.cat([cos_sim, z1_zi_cos], 1)

    labels = torch.arange(cos_sim.size(0)).long().to(cls.device)
    loss_fct = nn.CrossEntropyLoss()

    loss = loss_fct(cos_sim, labels)

    if not return_dict:
        output = (cos_sim,) + outputs[2:]
        return ((loss,) + output) if loss is not None else output

    return SequenceClassifierOutput(
        loss=loss,
        logits=cos_sim,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )


def sent_emb_forward(
        cls,
        encoder,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
):
    return_dict = return_dict if return_dict is not None else cls.config.use_return_dict

    outputs = encoder(
        input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        position_ids=position_ids,
        head_mask=head_mask,
        inputs_embeds=inputs_embeds,
        output_attentions=output_attentions,
        output_hidden_states=True if cls.pooler_type in [Pooler.TYPE_TOP2, Pooler.TYPE_FIRST_LAST] else False,
        return_dict=True,
    )

    pooler_output = cls.pooler(attention_mask, outputs)
    if cls.pooler_type == Pooler.TYPE_CLS and not cls.mlp_only_train:
        pooler_output = cls.mlp(pooler_output)

    if not return_dict:
        return (outputs[0], pooler_output) + outputs[2:]

    return BaseModelOutputWithPoolingAndCrossAttentions(
        pooler_output=pooler_output,
        last_hidden_state=outputs.last_hidden_state,
        hidden_states=outputs.hidden_states,
    )


def cl_init(cls, config):
    cls.pooler = Pooler(cls.pooler_type)
    if cls.pooler_type == Pooler.TYPE_CLS:
        cls.mlp = MLPLayer(config)
    cls.sim = Similarity(temperature=cls.temperature)
    cls.post_init()


class BertForCL(BertPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r'position_ids']

    def __init__(self, config, training_args=None):
        super().__init__(config)

        self.alpha = training_args.alpha
        self.temperature = training_args.temperature
        self.pooler_type = training_args.pooler_type
        self.mlp_only_train = training_args.mlp_only_train
        self.num_augmentation = training_args.num_augmentation

        self.bert = BertModel(config, add_pooling_layer=False)

        cl_init(self, config)

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            sent_emb=False,
    ):
        if sent_emb:
            return sent_emb_forward(
                self,
                self.bert,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        else:
            return cl_forward(
                self,
                self.bert,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )


class RobertaForCL(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r'position_ids']

    def __init__(self, config, training_args=None):
        super().__init__(config)

        self.alpha = training_args.alpha
        self.temperature = training_args.temperature
        self.pooler_type = training_args.pooler_type
        self.mlp_only_train = training_args.mlp_only_train
        self.num_augmentation = training_args.num_augmentation

        self.roberta = RobertaModel(config, add_pooling_layer=False)

        cl_init(self, config)

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            sent_emb=False,
    ):
        if sent_emb:
            return sent_emb_forward(
                self,
                self.roberta,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        else:
            return cl_forward(
                self,
                self.roberta,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
