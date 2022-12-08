import torch
import torch.nn as nn
from transformers.modeling_outputs import SequenceClassifierOutput, BaseModelOutputWithPoolingAndCrossAttentions
from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertModel, BertLMPredictionHead
from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel, RobertaModel, RobertaLMHead


class MLPLayer(nn.Module):
    """
    Head for getting sentence representations over RoBERTa/BERT's CLS representation.
    """

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = self.activation(x)

        return x


class Similarity(nn.Module):
    """
    Dot product or cosine similarity
    """

    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return self.cos(x, y) / self.temp


class Pooler(nn.Module):
    """
    Parameter-free poolers to get the sentence embedding
    'cls': [CLS] representation with BERT/RoBERTa's MLP pooler.
    'cls_before_pooler': [CLS] representation without the original MLP pooler.
    'avg': average of the last layers' hidden states at each token.
    'avg_top2': average of the last two layers.
    'avg_first_last': average of the first and the last layers.
    """

    def __init__(self, pooler_type):
        super().__init__()
        self.pooler_type = pooler_type
        assert self.pooler_type in ["cls", "cls_before_pooler", "avg", "avg_top2",
                                    "avg_first_last"], "unrecognized pooling type %s" % self.pooler_type

    def forward(self, attention_mask, outputs):
        last_hidden = outputs.last_hidden_state
        pooler_output = outputs.pooler_output
        hidden_states = outputs.hidden_states

        if self.pooler_type in ['cls_before_pooler', 'cls']:
            return last_hidden[:, 0]
        elif self.pooler_type == "avg":
            return ((last_hidden * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1))
        elif self.pooler_type == "avg_first_last":
            first_hidden = hidden_states[0]
            last_hidden = hidden_states[-1]
            pooled_result = ((first_hidden + last_hidden) / 2.0 * attention_mask.unsqueeze(-1)).sum(
                1) / attention_mask.sum(-1).unsqueeze(-1)
            return pooled_result
        elif self.pooler_type == "avg_top2":
            second_last_hidden = hidden_states[-2]
            last_hidden = hidden_states[-1]
            pooled_result = ((last_hidden + second_last_hidden) / 2.0 * attention_mask.unsqueeze(-1)).sum(
                1) / attention_mask.sum(-1).unsqueeze(-1)
            return pooled_result
        else:
            raise NotImplementedError


def cl_init(cls, config):
    """
    Contrastive learning class init function.
    """
    cls.pooler_type = cls.model_args.pooler_type
    cls.pooler = Pooler(cls.model_args.pooler_type)
    if cls.model_args.pooler_type == "cls":
        cls.mlp = MLPLayer(config)
    cls.sim = Similarity(temp=cls.model_args.temp)
    cls.init_weights()


def encode(
        cls,
        encoder,
        input_ids,
        attention_mask,
        token_type_ids,
        position_ids,
        head_mask,
        inputs_embeds,
        output_attentions,
        return_dict
):
    return_dict = return_dict if return_dict is not None else cls.config.use_return_dict

    batch_size = input_ids.size(0)

    # Number of sentences in one instance
    # 2: pair instance; 3: pair instance with a hard negative
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
        output_hidden_states=True if cls.model_args.pooler_type in ['avg_top2', 'avg_first_last'] else False,
        return_dict=True,
    )

    # Pooling
    pooler_output = cls.pooler(attention_mask, outputs)
    pooler_output = pooler_output.view((batch_size, num_sent, pooler_output.size(-1)))  # (bs, num_sent, hidden)

    # If using "cls", we add an extra MLP layer (same as BERT's original implementation) over the representation.
    # if cls.pooler_type == "cls":
    #     pooler_output = cls.mlp(pooler_output)

    # Separate representation
    z1, z2 = pooler_output[:, 0], pooler_output[:, 1]

    # Hard negative
    z3 = None
    if num_sent == 3:
        z3 = pooler_output[:, 2]

    # Torch.cat z1 z2
    t1 = z1.unsqueeze(1).repeat(1, batch_size, 1)
    t2 = z2.unsqueeze(0).repeat(batch_size, 1, 1)
    z_cat = torch.cat((t1, t2, (t1 - t2).abs(), (t1 * t2)), dim=-1)
    return z1, z2, z3, z_cat, outputs, batch_size, num_sent, return_dict


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
        mlm_input_ids=None,
        mlm_labels=None,
):
    cls.try_switch()
    cls.classifier.to(cls.device)

    if cls.is_classifier_train_time:
        with torch.no_grad():
            z1, z2, z3, z_cat, outputs, batch_size, num_sent, return_dict = encode(
                cls,
                encoder,
                input_ids,
                attention_mask,
                token_type_ids,
                position_ids,
                head_mask,
                inputs_embeds,
                output_attentions,
                return_dict
            )

        # predict
        predict = cls.classifier(z_cat)

        # Labels
        labels = torch.eye(batch_size).long().to(cls.device)

        # reshape
        predict = predict.reshape(batch_size * batch_size, 2)
        labels = labels.reshape(batch_size * batch_size)

        class_weight = torch.FloatTensor([1 / (batch_size - 1), 1]).to(cls.device)
        loss_fct = nn.CrossEntropyLoss(weight=class_weight)

        loss = loss_fct(predict, labels)
        loss_val = loss.item()

        print(f"\nclassifier training loss: {loss_val}")

        if loss_val < cls.training_args.classifier_loss_limit:
            cls.force_switch()

        if not return_dict:
            output = (predict,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output
        return SequenceClassifierOutput(
            loss=loss,
            logits=predict,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    else:
        z1, z2, z3, z_cat, outputs, batch_size, num_sent, return_dict = encode(
            cls,
            encoder,
            input_ids,
            attention_mask,
            token_type_ids,
            position_ids,
            head_mask,
            inputs_embeds,
            output_attentions,
            return_dict
        )

        cos_sim = cls.sim(z1.unsqueeze(1), z2.unsqueeze(0))

        # Hard negative
        if num_sent >= 3:
            z1_z3_cos = cls.sim(z1.unsqueeze(1), z3.unsqueeze(0))
            cos_sim = torch.cat([cos_sim, z1_z3_cos], 1)

        with torch.no_grad():
            predict = cls.classifier(z_cat)
            labels = predict.argmax(-1)

            if not ((batch_size - cls.training_args.pseudo_label_window_range)
                    <= labels.sum().item()
                    <= (batch_size + cls.training_args.pseudo_label_window_range)):
                print(f"\npseudo-label: [{labels.sum().item()}], use original label")
                labels = torch.arange(cos_sim.size(0)).long().to(cls.device)
                cls.force_switch()

            else:
                print(f"\npseudo-label: [{labels.sum().item()}, {labels.sum(dim=-1).tolist()}], use pseudo-label")
                labels = labels.transpose(-2, -1).divide(labels.sum(dim=-1)).transpose(-2, -1)
                labels[labels != labels] = 0

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


def sentemb_forward(
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
        output_hidden_states=True if cls.pooler_type in ['avg_top2', 'avg_first_last'] else False,
        return_dict=True,
    )

    pooler_output = cls.pooler(attention_mask, outputs)
    # if cls.pooler_type == "cls" and not cls.model_args.mlp_only_train:
    #     pooler_output = cls.mlp(pooler_output)

    if not return_dict:
        return (outputs[0], pooler_output) + outputs[2:]

    return BaseModelOutputWithPoolingAndCrossAttentions(
        pooler_output=pooler_output,
        last_hidden_state=outputs.last_hidden_state,
        hidden_states=outputs.hidden_states,
    )


class BertForCL(BertPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config, *args, **kargs):
        super().__init__(config)
        self.model_args = args[0]
        self.training_args = args[1]
        self.bert = BertModel(config, add_pooling_layer=False)

        # YYH --
        self.classifier = None
        self.classifier_input_size = config.hidden_size * 4
        self.classifier_model_init()

        self.is_classifier_train_time = True
        self.classifier_counter = 0
        self.encoder_counter = 0
        # --

        if self.model_args.do_mlm:
            self.lm_head = BertLMPredictionHead(config)

        cl_init(self, config)

    def classifier_model_init(self):
        if not self.classifier:
            hidden_size = 512
            output_size = 2
            self.classifier = nn.Sequential(
                nn.Linear(self.classifier_input_size, hidden_size),
                nn.LeakyReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.LeakyReLU(),
                nn.Linear(hidden_size, output_size),
                nn.LeakyReLU(),
            )

    def force_switch(self):
        if self.is_classifier_train_time:
            self.classifier_counter += (
                    self.training_args.train_classifier_interval
                    - (self.classifier_counter % self.training_args.train_classifier_interval)
            )
        else:
            self.encoder_counter += (
                    self.training_args.train_encoder_interval
                    - (self.encoder_counter % self.training_args.train_encoder_interval)
            )

        self.is_classifier_train_time = not self.is_classifier_train_time

    def _try_switch(self, counter, interval):
        if counter % interval == 0:
            self.is_classifier_train_time = not self.is_classifier_train_time

    def try_switch(self):
        if self.is_classifier_train_time:
            self.classifier_counter += 1
            self._try_switch(self.classifier_counter, self.training_args.train_classifier_interval)

        else:
            self.encoder_counter += 1
            self._try_switch(self.encoder_counter, self.training_args.train_encoder_interval)

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
            mlm_input_ids=None,
            mlm_labels=None,
    ):
        if sent_emb:
            return sentemb_forward(
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
                mlm_input_ids=mlm_input_ids,
                mlm_labels=mlm_labels,
            )


class RobertaForCL(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config, *args, **kwargs):
        super().__init__(config)
        self.model_args = args[0]
        self.roberta = RobertaModel(config, add_pooling_layer=False)

        if self.model_args.do_mlm:
            self.lm_head = RobertaLMHead(config)

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
            mlm_input_ids=None,
            mlm_labels=None,
    ):
        if sent_emb:
            return sentemb_forward(
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
                mlm_input_ids=mlm_input_ids,
                mlm_labels=mlm_labels,
            )
