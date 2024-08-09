from abc import ABC
import torch.nn
from transformers import PretrainedConfig, AutoModel, AutoModelForCausalLM, EncoderDecoderModel, EncoderDecoderConfig, \
    AutoConfig, PreTrainedModel, AutoModelForSequenceClassification, AlbertModel, Data2VecAudioModel, RobertaModel
import warnings
import enum
from typing import Optional, Tuple, Union
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import Seq2SeqLMOutput, BaseModelOutput, SequenceClassifierOutput
from transformers.models.encoder_decoder.modeling_encoder_decoder import DEPRECATION_WARNING, shift_tokens_right
import torch.nn.functional as F
from collections import Counter

from settings import EMPATHY_CLASSIFIER_MODELS_PATH
from utils.models import MultiTaskModel, BaseMultiTaskOutput, T5EncoderClassifier


class ModelType(enum.Enum):
    roberta_shared = 'roberta_shared'
    roberta_gpt2 = 'roberta_gpt2'
    roberta_dialogpt = 'roberta_dialogpt'


class RobertaShared(EncoderDecoderModel, ABC):
    """
        it isn't necessary to make a new class,
        this class is written to change lately on initial and forward functions
        """

    def __init__(self, bos_token_id=0, eos_token_id=2, pad_token_id=50266, config: PretrainedConfig = None,
                 embedding_tokens_len=50267,
                 *inputs, **kwargs):
        """
        set encoder and decoder for Roberta-Roberta (shared weights) seq2seq model
        :param config:
        :param inputs: use as args
        :param kwargs:
        """

        config_encoder = AutoConfig.from_pretrained('roberta-base')
        config_decoder = AutoConfig.from_pretrained('roberta-base')

        config_decoder.is_decoder = True
        config_decoder.add_cross_attention = True
        config_decoder.max_length = 64
        config_decoder.min_length = 2

        encoder = AutoModel.from_config(config=config_encoder)
        decoder = AutoModelForCausalLM.from_config(config=config_decoder)

        if embedding_tokens_len:
            encoder.resize_token_embeddings(embedding_tokens_len)
            decoder.resize_token_embeddings(embedding_tokens_len)

        if config is None:
            config = EncoderDecoderConfig.from_encoder_decoder_configs(encoder_config=encoder.config,
                                                                       decoder_config=decoder.config)

        config.decoder_start_token_id = bos_token_id
        config.eos_token_id = eos_token_id
        config.pad_token_id = pad_token_id

        # sensible parameters for beam search
        # set decoding params
        config.max_length = 64
        config.min_length = 2
        config.early_stopping = True
        config.no_repeat_ngram_size = 3
        config.length_penalty = 2.0
        config.num_beams = 4
        config.vocab_size = config.encoder.vocab_size

        config.tie_encoder_decoder = True
        super().__init__(config=config, encoder=encoder, decoder=decoder, *inputs, **kwargs)


class Roberta2GPT2(EncoderDecoderModel, ABC):
    """
    it isn't necessary to make a new class,
    this class is written to change lately on initial and forward functions
    """

    def __init__(self, bos_token_id=0, eos_token_id=2, pad_token_id=50266,
                 config: PretrainedConfig = None, embedding_tokens_len=50267,
                 *inputs, **kwargs):
        """
        set encoder and decoder for Roberta-GPT2 seq2seq model
        :param config:
        :param inputs:
        :param kwargs:
        """
        config_encoder = AutoConfig.from_pretrained('roberta-base')
        config_decoder = AutoConfig.from_pretrained('gpt2')

        config_decoder.is_decoder = True
        config_decoder.add_cross_attention = True
        config_decoder.max_length = 64
        config_decoder.min_length = 2

        encoder = AutoModel.from_config(config=config_encoder)
        decoder = AutoModelForCausalLM.from_config(config=config_decoder)

        if embedding_tokens_len:
            encoder.resize_token_embeddings(embedding_tokens_len)
            decoder.resize_token_embeddings(embedding_tokens_len)

        if config is None:
            config = EncoderDecoderConfig.from_encoder_decoder_configs(encoder_config=encoder.config,
                                                                       decoder_config=decoder.config)
        config.decoder_start_token_id = bos_token_id
        config.eos_token_id = eos_token_id
        config.pad_token_id = pad_token_id

        # sensible parameters for beam search
        # set decoding params
        config.max_length = 64
        config.min_length = 2
        config.early_stopping = True
        config.no_repeat_ngram_size = 3
        config.length_penalty = 2.0
        config.num_beams = 4
        config.vocab_size = config.encoder.vocab_size
        super().__init__(config=config, encoder=encoder, decoder=decoder, *inputs, **kwargs)


class ExampleEncoder(PreTrainedModel):

    def __init__(self, config=PretrainedConfig(), *args, **kwargs):
        super().__init__(config=config, *args, **kwargs)
        self.encoder = AlbertModel.from_pretrained("albert-base-v2")

    def forward(self,
                example_0_input_ids: Optional[torch.LongTensor] = None,
                example_0_attention_mask: Optional[torch.FloatTensor] = None,
                example_0_token_type_ids: Optional[torch.FloatTensor] = None,
                example_1_input_ids: Optional[torch.LongTensor] = None,
                example_1_attention_mask: Optional[torch.FloatTensor] = None,
                example_1_token_type_ids: Optional[torch.FloatTensor] = None,
                example_2_input_ids: Optional[torch.LongTensor] = None,
                example_2_attention_mask: Optional[torch.FloatTensor] = None,
                example_2_token_type_ids: Optional[torch.FloatTensor] = None,
                example_3_input_ids: Optional[torch.LongTensor] = None,
                example_3_attention_mask: Optional[torch.FloatTensor] = None,
                example_3_token_type_ids: Optional[torch.FloatTensor] = None,
                example_4_input_ids: Optional[torch.LongTensor] = None,
                example_4_attention_mask: Optional[torch.FloatTensor] = None,
                example_4_token_type_ids: Optional[torch.FloatTensor] = None,
                *args, **kwargs):
        """

        :param example_0_input_ids:
        :param example_0_attention_mask:
        :param example_0_token_type_ids:
        :param example_1_input_ids:
        :param example_1_attention_mask:
        :param example_1_token_type_ids:
        :param example_2_input_ids:
        :param example_2_attention_mask:
        :param example_2_token_type_ids:
        :param example_3_input_ids:
        :param example_3_attention_mask:
        :param example_3_token_type_ids:
        :param example_4_input_ids:
        :param example_4_attention_mask:
        :param example_4_token_type_ids:
        :param args:
        :param kwargs:
        :return:
        """
        all_encoded = list()

        if example_0_input_ids is not None:
            encoded_1 = self.encoder(input_ids=example_0_input_ids,
                                     attention_mask=example_0_attention_mask,
                                     token_type_ids=example_0_token_type_ids)[0]
            all_encoded.append(encoded_1)

        if example_1_input_ids is not None:
            encoded_2 = self.encoder(input_ids=example_1_input_ids,
                                     attention_mask=example_1_attention_mask,
                                     token_type_ids=example_1_token_type_ids)[0]
            all_encoded.append(encoded_2)

        if example_2_input_ids is not None:
            encoded_3 = self.encoder(input_ids=example_2_input_ids,
                                     attention_mask=example_2_attention_mask,
                                     token_type_ids=example_2_token_type_ids)[0]
            all_encoded.append(encoded_3)

        if example_3_input_ids is not None:
            encoded_4 = self.encoder(input_ids=example_3_input_ids,
                                     attention_mask=example_3_attention_mask,
                                     token_type_ids=example_3_token_type_ids)[0]
            all_encoded.append(encoded_4)

        if example_4_input_ids is not None:
            encoded_5 = self.encoder(input_ids=example_4_input_ids,
                                     attention_mask=example_4_attention_mask,
                                     token_type_ids=example_4_token_type_ids)[0]
            all_encoded.append(encoded_5)

        if len(all_encoded) == 0:
            return None

        return torch.sum(torch.stack(all_encoded), dim=0)


class KnowledgesEncoder(PreTrainedModel):

    def __init__(self, kwn_embedding_tokens_len=50266, config: PretrainedConfig = PretrainedConfig(), *args, **kwargs):
        super().__init__(config=config, *args, **kwargs)
        print(kwn_embedding_tokens_len, 'kwn_embedding_tokens_len')
        self.encoder = AlbertModel.from_pretrained("albert-base-v2")
        self.encoder.resize_token_embeddings(kwn_embedding_tokens_len)
        self.social_event_attention = torch.nn.MultiheadAttention(embed_dim=768, num_heads=8, dropout=0.2)
        self.social_entity_attention = torch.nn.MultiheadAttention(embed_dim=768, num_heads=8, dropout=0.2)
        self.norm_layer = torch.nn.LayerNorm(768)

    def forward(self,
                react_rel_input_ids: Optional[torch.LongTensor] = None,
                react_rel_attention_mask: Optional[torch.FloatTensor] = None,
                react_rel_token_type_ids: Optional[torch.FloatTensor] = None,
                social_rel_input_ids: Optional[torch.LongTensor] = None,
                social_rel_attention_mask: Optional[torch.FloatTensor] = None,
                social_rel_token_type_ids: Optional[torch.FloatTensor] = None,
                event_rel_input_ids: Optional[torch.LongTensor] = None,
                event_rel_attention_mask: Optional[torch.FloatTensor] = None,
                event_rel_token_type_ids: Optional[torch.FloatTensor] = None,
                entity_rel_input_ids: Optional[torch.LongTensor] = None,
                entity_rel_attention_mask: Optional[torch.FloatTensor] = None,
                entity_rel_token_type_ids: Optional[torch.FloatTensor] = None,
                *args, **kwargs):
        """

        :param react_rel_input_ids:
        :param react_rel_attention_mask:
        :param react_rel_token_type_ids:
        :param social_rel_input_ids:
        :param social_rel_attention_mask:
        :param social_rel_token_type_ids:
        :param event_rel_input_ids:
        :param event_rel_attention_mask:
        :param event_rel_token_type_ids:
        :param entity_rel_input_ids:
        :param entity_rel_attention_mask:
        :param entity_rel_token_type_ids:
        :param args:
        :param kwargs:
        :return:
        """

        encoded_react_knw = self.encoder(input_ids=react_rel_input_ids,
                                         attention_mask=react_rel_attention_mask,
                                         token_type_ids=react_rel_token_type_ids)[0]

        encoded_social_knw = self.encoder(input_ids=social_rel_input_ids,
                                          attention_mask=social_rel_attention_mask,
                                          token_type_ids=social_rel_token_type_ids)[0]

        encoded_event_knw = self.encoder(input_ids=event_rel_input_ids,
                                         attention_mask=event_rel_attention_mask,
                                         token_type_ids=event_rel_token_type_ids)[0]

        encoded_entity_knw = self.encoder(input_ids=entity_rel_input_ids,
                                          attention_mask=entity_rel_attention_mask,
                                          token_type_ids=entity_rel_token_type_ids)[0]

        social_event_attention_output, _ = self.social_event_attention(query=encoded_social_knw,
                                                                       key=encoded_event_knw,
                                                                       value=encoded_event_knw)

        social_entity_attention_output, _ = self.social_entity_attention(query=encoded_social_knw,
                                                                         key=encoded_entity_knw,
                                                                         value=encoded_entity_knw)

        return self.norm_layer(torch.sum(torch.stack([encoded_social_knw,
                                                      social_event_attention_output,
                                                      social_entity_attention_output,
                                                      encoded_react_knw]), dim=0))


class TextualResponseGenerator(EncoderDecoderModel, ABC):

    def __init__(self, special_token_dict: dict, bos_token_id=0, eos_token_id=2, pad_token_id=50266,
                 config: PretrainedConfig = None, embedding_tokens_len=50267,
                 kwn_embedding_tokens_len=50266, empathy_loss_weight=0.1, main_loss_weight=1, div_loss_weight=1.5,
                 *inputs, **kwargs):
        """
        set encoder and decoder for Roberta-DialoGPT seq2seq model
        :param config:
        :param inputs:
        :param kwargs:
        """
        config_encoder = AutoConfig.from_pretrained('roberta-base')
        config_decoder = AutoConfig.from_pretrained('microsoft/DialoGPT-small')

        config_decoder.is_decoder = True
        config_decoder.add_cross_attention = True
        config_decoder.max_new_tokens = 64
        config_decoder.min_length = 2

        encoder = AutoModel.from_config(config=config_encoder)
        decoder = AutoModelForCausalLM.from_config(config=config_decoder)

        if embedding_tokens_len:
            encoder.resize_token_embeddings(embedding_tokens_len)
            decoder.resize_token_embeddings(embedding_tokens_len)

        if config is None:
            config = EncoderDecoderConfig.from_encoder_decoder_configs(encoder_config=encoder.config,
                                                                       decoder_config=decoder.config)
        config.decoder_start_token_id = bos_token_id
        config.eos_token_id = eos_token_id
        config.pad_token_id = pad_token_id

        # sensible parameters for beam search
        # set decoding params
        config.max_new_tokens = 64
        config.min_length = 2
        config.early_stopping = True
        config.no_repeat_ngram_size = 3
        config.length_penalty = 2.0
        config.num_beams = 4
        config.vocab_size = config.encoder.vocab_size
        super().__init__(config=config, encoder=encoder, decoder=decoder, *inputs, **kwargs)

        self.knowledge_encoder = KnowledgesEncoder(kwn_embedding_tokens_len=kwn_embedding_tokens_len)
        self.example_encoders = ExampleEncoder()
        self.norm_layer = torch.nn.LayerNorm(768)

        # loss
        self.empathy_loss_weight = empathy_loss_weight
        self.main_loss_weight = main_loss_weight
        self.div_loss_weight = div_loss_weight

        self.special_token_dict = special_token_dict
        self.word_freq = torch.zeros(self.config.vocab_size)
        self.criterion = torch.nn.NLLLoss(ignore_index=config.pad_token_id, reduction="sum")

        # models for losses
        self.empathy_classifier_model1 = T5EncoderClassifier("base", 'roberta-base', 2, 0)
        self.empathy_classifier_model1.load_state_dict(torch.load(f"{EMPATHY_CLASSIFIER_MODELS_PATH}/saved/empathy/1619600015/model.pt"))
        for param in self.empathy_classifier_model1.parameters():
            param.requires_grad = False

        self.empathy_classifier_model2 = T5EncoderClassifier("base", 'roberta-base', 2, 0)
        self.empathy_classifier_model2.load_state_dict(torch.load(f"{EMPATHY_CLASSIFIER_MODELS_PATH}/saved/empathy/1619600805/model.pt"))
        for param in self.empathy_classifier_model2.parameters():
            param.requires_grad = False

        self.empathy_classifier_model3 = T5EncoderClassifier("base", 'roberta-base', 2, 0)
        self.empathy_classifier_model3.load_state_dict(torch.load(f"{EMPATHY_CLASSIFIER_MODELS_PATH}/saved/empathy/1619601340/model.pt"))
        for param in self.empathy_classifier_model3.parameters():
            param.requires_grad = False

    def forward(self,
                input_ids: Optional[torch.LongTensor] = None,
                attention_mask: Optional[torch.FloatTensor] = None,
                decoder_input_ids: Optional[torch.LongTensor] = None,
                decoder_attention_mask: Optional[torch.BoolTensor] = None,
                encoder_outputs: Optional[Tuple[torch.FloatTensor]] = None,
                past_key_values: Tuple[Tuple[torch.FloatTensor]] = None,
                inputs_embeds: Optional[torch.FloatTensor] = None,
                decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
                labels: Optional[torch.LongTensor] = None,
                use_cache: Optional[bool] = None,
                output_attentions: Optional[bool] = None,
                output_hidden_states: Optional[bool] = None,
                return_dict: Optional[bool] = None,
                react_rel_input_ids: Optional[torch.LongTensor] = None,
                react_rel_attention_mask: Optional[torch.FloatTensor] = None,
                react_rel_token_type_ids: Optional[torch.FloatTensor] = None,
                social_rel_input_ids: Optional[torch.LongTensor] = None,
                social_rel_attention_mask: Optional[torch.FloatTensor] = None,
                social_rel_token_type_ids: Optional[torch.FloatTensor] = None,
                event_rel_input_ids: Optional[torch.LongTensor] = None,
                event_rel_attention_mask: Optional[torch.FloatTensor] = None,
                event_rel_token_type_ids: Optional[torch.FloatTensor] = None,
                entity_rel_input_ids: Optional[torch.LongTensor] = None,
                entity_rel_attention_mask: Optional[torch.FloatTensor] = None,
                entity_rel_token_type_ids: Optional[torch.FloatTensor] = None,
                example_0_input_ids: Optional[torch.LongTensor] = None,
                example_0_attention_mask: Optional[torch.FloatTensor] = None,
                example_0_token_type_ids: Optional[torch.FloatTensor] = None,
                example_1_input_ids: Optional[torch.LongTensor] = None,
                example_1_attention_mask: Optional[torch.FloatTensor] = None,
                example_1_token_type_ids: Optional[torch.FloatTensor] = None,
                example_2_input_ids: Optional[torch.LongTensor] = None,
                example_2_attention_mask: Optional[torch.FloatTensor] = None,
                example_2_token_type_ids: Optional[torch.FloatTensor] = None,
                example_3_input_ids: Optional[torch.LongTensor] = None,
                example_3_attention_mask: Optional[torch.FloatTensor] = None,
                example_3_token_type_ids: Optional[torch.FloatTensor] = None,
                example_4_input_ids: Optional[torch.LongTensor] = None,
                example_4_attention_mask: Optional[torch.FloatTensor] = None,
                example_4_token_type_ids: Optional[torch.FloatTensor] = None,
                **kwargs) -> Union[Tuple, Seq2SeqLMOutput]:

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        kwargs_encoder = {argument: value for argument, value in kwargs.items() if not argument.startswith("decoder_")}

        kwargs_decoder = {
            argument[len("decoder_"):]: value for argument, value in kwargs.items() if argument.startswith("decoder_")
        }

        if encoder_outputs is None:
            encoder_outputs = self.encode_context(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=True,
                **kwargs_encoder
            )

            encoded_knowledge = self.knowledge_encoder(
                react_rel_input_ids=react_rel_input_ids,
                react_rel_attention_mask=react_rel_attention_mask,
                react_rel_token_type_ids=react_rel_token_type_ids,
                social_rel_input_ids=social_rel_input_ids,
                social_rel_attention_mask=social_rel_attention_mask,
                social_rel_token_type_ids=social_rel_token_type_ids,
                event_rel_input_ids=event_rel_input_ids,
                event_rel_attention_mask=event_rel_attention_mask,
                event_rel_token_type_ids=event_rel_token_type_ids,
                entity_rel_input_ids=entity_rel_input_ids,
                entity_rel_attention_mask=entity_rel_attention_mask,
                entity_rel_token_type_ids=entity_rel_token_type_ids
            )

            encoded_examples = self.example_encoders(
                example_0_input_ids=example_0_input_ids,
                example_0_attention_mask=example_0_attention_mask,
                example_0_token_type_ids=example_0_token_type_ids,
                example_1_input_ids=example_1_input_ids,
                example_1_attention_mask=example_1_attention_mask,
                example_1_token_type_ids=example_1_token_type_ids,
                example_2_input_ids=example_2_input_ids,
                example_2_attention_mask=example_2_attention_mask,
                example_2_token_type_ids=example_2_token_type_ids,
                example_3_input_ids=example_3_input_ids,
                example_3_attention_mask=example_3_attention_mask,
                example_3_token_type_ids=example_3_token_type_ids,
                example_4_input_ids=example_4_input_ids,
                example_4_attention_mask=example_4_attention_mask,
                example_4_token_type_ids=example_4_token_type_ids,
            )

            last_hidden_state = encoder_outputs.last_hidden_state + encoded_knowledge

            if encoded_examples is not None:
                last_hidden_state = last_hidden_state + encoded_examples

            encoder_outputs['last_hidden_state'] = self.norm_layer(last_hidden_state)

        elif isinstance(encoder_outputs, tuple):
            encoder_outputs = BaseModelOutput(*encoder_outputs)

        encoder_hidden_states = encoder_outputs[0]

        # optionally project encoder_hidden_states
        if (
                self.encoder.config.hidden_size != self.decoder.config.hidden_size
                and self.decoder.config.cross_attention_hidden_size is None
        ):
            encoder_hidden_states = self.enc_to_dec_proj(encoder_hidden_states)

        if (labels is not None) and (decoder_input_ids is None and decoder_inputs_embeds is None):
            decoder_input_ids = shift_tokens_right(
                labels, self.config.pad_token_id, self.config.decoder_start_token_id
            )

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            use_cache=use_cache,
            past_key_values=past_key_values,
            return_dict=return_dict,
            **kwargs_decoder,
        )

        # Compute loss independent from decoder (as some shift the logits inside them)
        loss = self.compute_loss(labels=labels, decoder_outputs=decoder_outputs, return_dict=return_dict)

        if not return_dict:
            if loss is not None:
                return (loss,) + decoder_outputs + encoder_outputs
            else:
                return decoder_outputs + encoder_outputs

        return Seq2SeqLMOutput(
            loss=loss,
            logits=decoder_outputs.logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

    def compute_face_loss(self, labels=None, logits=None):
        """
        the source of code is https://github.com/Sahandfer/CEM
        :param labels:
        :param logits:
        :return:
        """

        def clean_preds(logit):
            pred = torch.argmax(logit, dim=2)
            return pred.tolist()

        def update_frequency(preds):
            curr = Counter()
            for pred in preds:
                curr.update(pred)
            for k, v in curr.items():
                if k not in self.special_token_dict.values():
                    self.word_freq[k] += v

        def calc_weight():
            RF = self.word_freq / self.word_freq.sum()
            a = -1 / RF.max()
            weight = a * RF + 1
            weight = weight / weight.sum() * len(weight)

            return torch.FloatTensor(weight)

        if labels is not None:
            preds = clean_preds(logits)
            update_frequency(preds)
            self.criterion.weight = calc_weight()
            no_pad_label = labels[labels != -100]
            target_tokens = no_pad_label.long().sum().item()
            div_loss = self.criterion(
                logits.contiguous().view(-1, logits.size(-1)),
                labels.contiguous().view(-1),
            )
            div_loss /= target_tokens

            return div_loss

        return None

    def compute_empathy_loss(self, context_input_ids=None, labels=None, logits=None, response_mask=None):
        if response_mask is None:
            response_mask = torch.ones(labels.size())

        # compute empathy loss
        self.empathy_classifier_model1.eval()
        self.empathy_classifier_model2.eval()
        self.empathy_classifier_model3.eval()

        empathy1_preds = self.empathy_classifier_model1.output_from_logits(context_input_ids=context_input_ids,
                                                                           decoded_logits=logits,
                                                                           response_mask=response_mask)
        empathy2_preds = self.empathy_classifier_model2.output_from_logits(context_input_ids=context_input_ids,
                                                                           decoded_logits=logits,
                                                                           response_mask=response_mask)
        empathy3_preds = self.empathy_classifier_model3.output_from_logits(context_input_ids=context_input_ids,
                                                                           decoded_logits=logits,
                                                                           response_mask=response_mask)
        empathy1_labels = torch.ones((empathy1_preds.size()[0]))
        empathy2_labels = torch.ones((empathy2_preds.size()[0]))
        empathy3_labels = torch.ones((empathy3_preds.size()[0]))

        loss_fct = CrossEntropyLoss()
        empathy1_loss = loss_fct(empathy1_preds, empathy1_labels)
        empathy2_loss = loss_fct(empathy2_preds, empathy2_labels)
        empathy3_loss = loss_fct(empathy3_preds, empathy3_labels)

        return empathy1_loss + empathy2_loss + empathy3_loss

    def compute_loss(self, context_input_ids=None, context_attention_mask=None,
                     labels=None, decoder_outputs=None, response_mask=None, return_dict=True):
        """
        compute loss
        :param labels:
        :param context_input_ids:
        :param context_attention_mask:
        :param response_mask:
        :param decoder_outputs:
        :param return_dict:
        :return:
        """
        loss = None
        if labels is not None:
            warnings.warn(DEPRECATION_WARNING, FutureWarning)
            logits = decoder_outputs.logits if return_dict else decoder_outputs[0]
            loss_fct = CrossEntropyLoss()
            main_loss = loss_fct(logits.reshape(-1, self.decoder.config.vocab_size), labels.view(-1))

            face_loss = self.compute_face_loss(labels=labels, logits=logits)

            empathy_loss = self.compute_empathy_loss(context_input_ids=context_input_ids,
                                                     labels=labels,
                                                     logits=logits,
                                                     response_mask=response_mask)
            # aggregate losses
            loss = self.main_loss_weight * main_loss + \
                   self.empathy_loss_weight * empathy_loss + \
                   self.div_loss_weight * face_loss

        return loss

    def encode_context(self,
                       input_ids: Optional[torch.LongTensor] = None,
                       attention_mask: Optional[torch.FloatTensor] = None,
                       inputs_embeds: Optional[torch.FloatTensor] = None,
                       output_attentions: Optional[bool] = None,
                       output_hidden_states: Optional[bool] = None,
                       return_dict: Optional[bool] = None,
                       **kwargs):
        """
        encode context
        :param input_ids:
        :param attention_mask:
        :param inputs_embeds:
        :param output_attentions:
        :param output_hidden_states:
        :param return_dict:
        :param kwargs:
        :return:
        """

        return self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


class EmotionRoberta2DialoGPT(MultiTaskModel):

    def __init__(self,
                 config: PretrainedConfig = PretrainedConfig(),
                 *inputs,
                 **kwargs):
        kwargs['special_token_dict'] = kwargs.get('special_token_dict', None)
        kwargs['empathy_loss_weight'] = kwargs.get('empathy_loss_weight', 0.1)
        kwargs['main_loss_weight'] = kwargs.get('main_loss_weight', 1)
        kwargs['div_loss_weight'] = kwargs.get('div_loss_weight', 1.5)
        kwargs['bos_token_id'] = kwargs.get('bos_token_id', 0)
        kwargs['eos_token_id'] = kwargs.get('eos_token_id', 2)
        kwargs['pad_token_id'] = kwargs.get('pad_token_id', 50266)
        kwargs['embedding_tokens_len'] = kwargs.get('embedding_tokens_len', 50267)
        kwargs['kwn_embedding_tokens_len'] = kwargs.get('kwn_embedding_tokens_len', 50266)
        kwargs['num_labels'] = kwargs.get('num_labels', 32)
        kwargs['encoder_decoder_config'] = kwargs.get('encoder_decoder_config', None)
        super().__init__(config=config, *inputs, **kwargs)
        self.config.pad_token_id = kwargs.get('pad_token_id', 50266)

    def initial_models(self, **kwargs) -> dict:
        """
        initial models
        :param kwargs: the arguments that is pasted by __init__ function
        :return: a dictionary that shows task_id and its task (model)
            Example:
            {
            "task_1": self.response_generator,
            "task_2": self.classifier,
            ...
            }

            WARNING: if you doesn't put tasks correctly in dict the model doesn't run the task for training and testing
        """
        self.encoder_decoder = TextualResponseGenerator(bos_token_id=kwargs['bos_token_id'],
                                                        eos_token_id=kwargs['eos_token_id'],
                                                        pad_token_id=kwargs['pad_token_id'],
                                                        config=kwargs['encoder_decoder_config'],
                                                        embedding_tokens_len=kwargs['embedding_tokens_len'],
                                                        kwn_embedding_tokens_len=kwargs['kwn_embedding_tokens_len'],
                                                        div_loss_weight=kwargs['div_loss_weight'],
                                                        empathy_loss_weight=kwargs['empathy_loss_weight'],
                                                        main_loss_weight=kwargs['main_loss_weight'],
                                                        special_token_dict=kwargs['special_token_dict'])

        self.emotion_classifier = AutoModelForSequenceClassification.from_pretrained("roberta-base",
                                                                                     num_labels=kwargs['num_labels'])

        return {
            'response_generator': self.encoder_decoder,
            'emotion_classifier': self.emotion_classifier
        }

    def get_generative_task_id(self):
        """
        get the task id of generative task. if there is no generative task return None
        :return:
        """
        return 'response_generator'

    def set_shared_layers(self):
        """
        make the specific layers shared
        :return:
        """
        encoder = getattr(self.encoder_decoder, 'encoder')
        setattr(self.emotion_classifier, 'roberta', encoder)

    def get_arg_forward_settings(self) -> dict:
        """
        get a dictionary of information about forward function's arguments based on tasks.
        this dictionary is used for passing or mapping function's inputs
        :return: dictionary with below format
        {
            "task_id": {
                "arg_name_in_forward_func": "arg_name of forward functions of task model"
            }
        }

        Example:
        {
            "task_1": {
                "input_ids": "input_ids",
                "attention_mask": "attention_mask",
                "labels_1": "labels",
            },

            "task_2": {
                "input_ids": "input_ids",
                "attention_mask": "attention_mask",
                "labels_2": "labels",
            }
        }

        """
        return {
            'response_generator': {
                'input_ids': 'input_ids',
                'attention_mask': 'attention_mask',
                "decoder_input_ids": 'decoder_input_ids',
                'decoder_attention_mask': 'decoder_attention_mask',
                'encoder_outputs': 'encoder_outputs',
                'past_key_values': 'past_key_values',
                'inputs_embeds': 'inputs_embeds',
                'decoder_inputs_embeds': 'decoder_inputs_embeds',
                'labels': 'labels',
                'use_cache': 'use_cache',
                'output_attentions': 'output_attentions',
                'output_hidden_states': 'output_hidden_states',
                'react_rel_input_ids': 'react_rel_input_ids',
                'react_rel_attention_mask': 'react_rel_attention_mask',
                'react_rel_token_type_ids': 'react_rel_token_type_ids',
                'social_rel_input_ids': 'social_rel_input_ids',
                'social_rel_attention_mask': 'social_rel_attention_mask',
                'social_rel_token_type_ids': 'social_rel_token_type_ids',
                'event_rel_input_ids': 'event_rel_input_ids',
                'event_rel_attention_mask': 'event_rel_attention_mask',
                'event_rel_token_type_ids': 'event_rel_token_type_ids',
                'entity_rel_input_ids': 'entity_rel_input_ids',
                'entity_rel_attention_mask': 'entity_rel_attention_mask',
                'entity_rel_token_type_ids': 'entity_rel_token_type_ids',
                'example_0_input_ids': 'example_0_input_ids',
                'example_0_attention_mask': 'example_0_attention_mask',
                'example_0_token_type_ids': 'example_0_token_type_ids',
                'example_1_input_ids': 'example_1_input_ids',
                'example_1_attention_mask': 'example_1_attention_mask',
                'example_1_token_type_ids': 'example_1_token_type_ids',
                'example_2_input_ids': 'example_2_input_ids',
                'example_2_attention_mask': 'example_2_attention_mask',
                'example_2_token_type_ids': 'example_2_token_type_ids',
                'example_3_input_ids': 'example_3_input_ids',
                'example_3_attention_mask': 'example_3_attention_mask',
                'example_3_token_type_ids': 'example_3_token_type_ids',
                'example_4_input_ids': 'example_4_input_ids',
                'example_4_attention_mask': 'example_4_attention_mask',
                'example_4_token_type_ids': 'example_4_token_type_ids'
            },

            'emotion_classifier': {
                'input_ids': 'input_ids',
                'attention_mask': 'attention_mask',
                'token_type_ids': 'token_type_ids',
                'position_ids': 'position_ids',
                'head_mask': 'head_mask',
                'inputs_embeds': 'inputs_embeds',
                'emotion_labels': 'labels',
                'output_attentions': 'output_attentions',
                'output_hidden_states': 'output_hidden_states'
            }
        }

    def get_encoder(self):
        """
        get encoder of model if model is encoderdecoder model
        :return:
        """
        return self.TASK_CONFIG[self.get_generative_task_id()].get_encoder()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,
                emotion_labels=None, labels=None, react_rel_input_ids=None,
                react_rel_attention_mask=None, react_rel_token_type_ids=None, social_rel_input_ids=None,
                social_rel_attention_mask=None, social_rel_token_type_ids=None, event_rel_input_ids=None,
                event_rel_attention_mask=None, event_rel_token_type_ids=None, entity_rel_input_ids=None,
                entity_rel_attention_mask=None, entity_rel_token_type_ids=None,
                example_0_input_ids=None, example_0_attention_mask=None, example_0_token_type_ids=None,
                example_1_input_ids=None, example_1_attention_mask=None, example_1_token_type_ids=None,
                example_2_input_ids=None, example_2_attention_mask=None, example_2_token_type_ids=None,
                example_3_input_ids=None, example_3_attention_mask=None, example_3_token_type_ids=None,
                example_4_input_ids=None, example_4_attention_mask=None, example_4_token_type_ids=None,
                **kwargs) -> BaseMultiTaskOutput:
        """

        :param input_ids:
        :param attention_mask:
        :param token_type_ids:
        :param emotion_labels:
        :param labels:
        :param react_rel_input_ids:
        :param react_rel_attention_mask:
        :param react_rel_token_type_ids:
        :param social_rel_input_ids:
        :param social_rel_attention_mask:
        :param social_rel_token_type_ids:
        :param event_rel_input_ids:
        :param event_rel_attention_mask:
        :param event_rel_token_type_ids:
        :param entity_rel_input_ids:
        :param entity_rel_attention_mask:
        :param entity_rel_token_type_ids:
        :param example_0_input_ids:
        :param example_0_attention_mask:
        :param example_0_token_type_ids:
        :param example_1_input_ids:
        :param example_1_attention_mask:
        :param example_1_token_type_ids:
        :param example_2_input_ids:
        :param example_2_attention_mask:
        :param example_2_token_type_ids:
        :param example_3_input_ids:
        :param example_3_attention_mask:
        :param example_3_token_type_ids:
        :param example_4_input_ids:
        :param example_4_attention_mask:
        :param example_4_token_type_ids:
        :param kwargs:
        :return:
        """
        return super().forward(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                               emotion_labels=emotion_labels, labels=labels, react_rel_input_ids=react_rel_input_ids,
                               react_rel_attention_mask=react_rel_attention_mask,
                               react_rel_token_type_ids=react_rel_token_type_ids,
                               social_rel_input_ids=social_rel_input_ids,
                               social_rel_attention_mask=social_rel_attention_mask,
                               social_rel_token_type_ids=social_rel_token_type_ids,
                               event_rel_input_ids=event_rel_input_ids,
                               event_rel_attention_mask=event_rel_attention_mask,
                               event_rel_token_type_ids=event_rel_token_type_ids,
                               entity_rel_input_ids=entity_rel_input_ids,
                               entity_rel_attention_mask=entity_rel_attention_mask,
                               entity_rel_token_type_ids=entity_rel_token_type_ids,
                               example_0_input_ids=example_0_input_ids,
                               example_0_attention_mask=example_0_attention_mask,
                               example_0_token_type_ids=example_0_token_type_ids,
                               example_1_input_ids=example_1_input_ids,
                               example_1_attention_mask=example_1_attention_mask,
                               example_1_token_type_ids=example_1_token_type_ids,
                               example_2_input_ids=example_2_input_ids,
                               example_2_attention_mask=example_2_attention_mask,
                               example_2_token_type_ids=example_2_token_type_ids,
                               example_3_input_ids=example_3_input_ids,
                               example_3_attention_mask=example_3_attention_mask,
                               example_3_token_type_ids=example_3_token_type_ids,
                               example_4_input_ids=example_4_input_ids,
                               example_4_attention_mask=example_4_attention_mask,
                               example_4_token_type_ids=example_4_token_type_ids,
                               **kwargs)


class TextAudioIntegrator(PreTrainedModel):
    """
    source https://github.com/yuntaeyang/TelME
    """

    def __init__(self, hidden_size=768, beta_shift=1e-1, dropout_prob=0.2, num_head=3,
                 config=PretrainedConfig(), *args, **kwargs):
        super().__init__(config=config, *args, **kwargs)

        self.TEXT_DIM = 768
        self.ACOUSTIC_DIM = 768
        self.multihead_attn = torch.nn.MultiheadAttention(self.ACOUSTIC_DIM, num_head)

        self.W_hav = torch.nn.Linear(self.ACOUSTIC_DIM + self.TEXT_DIM, self.TEXT_DIM)

        self.W_av = torch.nn.Linear(self.ACOUSTIC_DIM, self.TEXT_DIM)

        self.beta_shift = beta_shift

        self.LayerNorm = torch.nn.LayerNorm(hidden_size)
        self.AV_LayerNorm = torch.nn.LayerNorm(self.ACOUSTIC_DIM)
        self.dropout = torch.nn.Dropout(dropout_prob)

    def forward(self, text_embedding=None, acoustic_embedding=None):
        """
        integrate text embed with acoustic embed using Attention based modality Shifting Fusion
        (idea of TelME: Teacher-leading Multimodal Fusion Network for Emotion Recognition in Conversation)
        :param text_embedding:
        :param acoustic_embedding:
        :return:
        """
        eps = 1e-6

        # pooled input
        text_embedding_mean_pool = torch.mean(text_embedding, dim=1, keepdim=True)
        acoustic_embedding_mean_pool = torch.mean(acoustic_embedding, dim=1, keepdim=True)

        # apply self-attention on acoustic embeddings
        new_nv = self.multihead_attn(acoustic_embedding_mean_pool, acoustic_embedding_mean_pool,
                                     acoustic_embedding_mean_pool)[0] + acoustic_embedding_mean_pool
        att_audio_embedding = self.dropout(self.AV_LayerNorm(new_nv))

        # shifting text embeddings
        weight_av = F.relu(self.W_hav(torch.cat((att_audio_embedding, text_embedding_mean_pool), dim=-1)))

        h_m = weight_av * self.W_av(att_audio_embedding)

        em_norm = text_embedding_mean_pool.norm(2, dim=-1)
        hm_norm = h_m.norm(2, dim=-1)

        hm_norm_ones = torch.ones(hm_norm.shape, requires_grad=True)
        hm_norm = torch.where(hm_norm == 0, hm_norm_ones, hm_norm)

        thresh_hold = (em_norm / (hm_norm + eps)) * self.beta_shift

        ones = torch.ones(thresh_hold.shape, requires_grad=True)

        alpha = torch.min(thresh_hold, ones)
        alpha = alpha.unsqueeze(dim=-1)

        acoustic_vis_embedding = alpha * h_m

        embedding_output = self.dropout(
            self.LayerNorm(acoustic_vis_embedding + text_embedding_mean_pool)
        )

        return embedding_output


class MultiModelEmotionClassifier(PreTrainedModel):

    def __init__(self, num_classes: int, embedding_tokens_len=50267, config=PretrainedConfig(), *args, **kwargs):
        super().__init__(config=config, *args, **kwargs)
        self.num_classes = num_classes
        self.roberta = RobertaModel.from_pretrained('roberta-base')
        self.roberta.resize_token_embeddings(embedding_tokens_len)

        self.data2vec_audio = Data2VecAudioModel.from_pretrained("facebook/data2vec-audio-base-960h")

        self.text_audio_integrator = TextAudioIntegrator()

        self.W = torch.nn.Linear(768, num_classes)

    def forward(self, input_ids=None, attention_mask=None, audio_input_values=None, audio_attention_mask=None,
                return_dict=True, labels=None, *args, **kwargs):
        """
        :param input_ids:
        :param attention_mask:
        :param audio_input_values:
        :param audio_attention_mask:
        :param labels:
        :param return_dict:
        :return:
        """
        text_embed = self.roberta(input_ids=input_ids, attention_mask=attention_mask,
                                  return_dict=True).last_hidden_state
        audio_embed = self.data2vec_audio(input_values=audio_input_values, attention_mask=audio_attention_mask,
                                          return_dict=True).last_hidden_state
        embedding_output = self.text_audio_integrator(text_embedding=text_embed, acoustic_embedding=audio_embed)
        logits = self.W(embedding_output)
        loss = None

        if labels is not None:
            # compute loss
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_classes), labels.view(-1))

        # prepare output
        if return_dict:
            return SequenceClassifierOutput(
                loss=loss,
                logits=logits,
            )
        else:
            output = (logits,)
            return ((loss,) + output) if loss is not None else output


class MultiModalResponseGenerator(TextualResponseGenerator, ABC):

    def __init__(self, special_token_dict: dict, bos_token_id=0, eos_token_id=2, pad_token_id=50266,
                 config: PretrainedConfig = None, embedding_tokens_len=50267,
                 kwn_embedding_tokens_len=50266, empathy_loss_weight=0.1, main_loss_weight=1, div_loss_weight=1.5,
                 *inputs, **kwargs):
        """
        set encoder and decoder for Roberta-DialoGPT seq2seq model
        :param config:
        :param inputs:
        :param kwargs:
        """
        super().__init__(special_token_dict=special_token_dict, empathy_loss_weight=empathy_loss_weight,
                         main_loss_weight=main_loss_weight, div_loss_weight=div_loss_weight, config=config,
                         bos_token_id=bos_token_id, eos_token_id=eos_token_id,
                         pad_token_id=pad_token_id, embedding_tokens_len=embedding_tokens_len,
                         kwn_embedding_tokens_len=kwn_embedding_tokens_len, *inputs, **kwargs)

        self.data2vec_audio = Data2VecAudioModel.from_pretrained("facebook/data2vec-audio-base-960h")
        self.text_audio_integrator = TextAudioIntegrator()

    def forward(self,
                audio_input_values=None,
                audio_attention_mask=None,
                input_ids: Optional[torch.LongTensor] = None,
                attention_mask: Optional[torch.FloatTensor] = None,
                decoder_input_ids: Optional[torch.LongTensor] = None,
                decoder_attention_mask: Optional[torch.BoolTensor] = None,
                encoder_outputs: Optional[Tuple[torch.FloatTensor]] = None,
                past_key_values: Tuple[Tuple[torch.FloatTensor]] = None,
                inputs_embeds: Optional[torch.FloatTensor] = None,
                decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
                labels: Optional[torch.LongTensor] = None,
                use_cache: Optional[bool] = None,
                output_attentions: Optional[bool] = None,
                output_hidden_states: Optional[bool] = None,
                return_dict: Optional[bool] = None,
                react_rel_input_ids: Optional[torch.LongTensor] = None,
                react_rel_attention_mask: Optional[torch.FloatTensor] = None,
                react_rel_token_type_ids: Optional[torch.FloatTensor] = None,
                social_rel_input_ids: Optional[torch.LongTensor] = None,
                social_rel_attention_mask: Optional[torch.FloatTensor] = None,
                social_rel_token_type_ids: Optional[torch.FloatTensor] = None,
                event_rel_input_ids: Optional[torch.LongTensor] = None,
                event_rel_attention_mask: Optional[torch.FloatTensor] = None,
                event_rel_token_type_ids: Optional[torch.FloatTensor] = None,
                entity_rel_input_ids: Optional[torch.LongTensor] = None,
                entity_rel_attention_mask: Optional[torch.FloatTensor] = None,
                entity_rel_token_type_ids: Optional[torch.FloatTensor] = None,
                example_0_input_ids: Optional[torch.LongTensor] = None,
                example_0_attention_mask: Optional[torch.FloatTensor] = None,
                example_0_token_type_ids: Optional[torch.FloatTensor] = None,
                example_1_input_ids: Optional[torch.LongTensor] = None,
                example_1_attention_mask: Optional[torch.FloatTensor] = None,
                example_1_token_type_ids: Optional[torch.FloatTensor] = None,
                example_2_input_ids: Optional[torch.LongTensor] = None,
                example_2_attention_mask: Optional[torch.FloatTensor] = None,
                example_2_token_type_ids: Optional[torch.FloatTensor] = None,
                example_3_input_ids: Optional[torch.LongTensor] = None,
                example_3_attention_mask: Optional[torch.FloatTensor] = None,
                example_3_token_type_ids: Optional[torch.FloatTensor] = None,
                example_4_input_ids: Optional[torch.LongTensor] = None,
                example_4_attention_mask: Optional[torch.FloatTensor] = None,
                example_4_token_type_ids: Optional[torch.FloatTensor] = None,
                **kwargs) -> Union[Tuple, Seq2SeqLMOutput]:
        """

        :param audio_input_values:
        :param audio_attention_mask:
        :param input_ids:
        :param attention_mask:
        :param decoder_input_ids:
        :param decoder_attention_mask:
        :param encoder_outputs:
        :param past_key_values:
        :param inputs_embeds:
        :param decoder_inputs_embeds:
        :param labels:
        :param use_cache:
        :param output_attentions:
        :param output_hidden_states:
        :param return_dict:
        :param react_rel_input_ids:
        :param react_rel_attention_mask:
        :param react_rel_token_type_ids:
        :param social_rel_input_ids:
        :param social_rel_attention_mask:
        :param social_rel_token_type_ids:
        :param event_rel_input_ids:
        :param event_rel_attention_mask:
        :param event_rel_token_type_ids:
        :param entity_rel_input_ids:
        :param entity_rel_attention_mask:
        :param entity_rel_token_type_ids:
        :param example_0_input_ids:
        :param example_0_attention_mask:
        :param example_0_token_type_ids:
        :param example_1_input_ids:
        :param example_1_attention_mask:
        :param example_1_token_type_ids:
        :param example_2_input_ids:
        :param example_2_attention_mask:
        :param example_2_token_type_ids:
        :param example_3_input_ids:
        :param example_3_attention_mask:
        :param example_3_token_type_ids:
        :param example_4_input_ids:
        :param example_4_attention_mask:
        :param example_4_token_type_ids:
        :param kwargs:
        :return:
        """
        kwargs['audio_input_values'] = audio_input_values
        kwargs['audio_attention_mask'] = audio_attention_mask
        return super().forward(input_ids=input_ids, attention_mask=attention_mask,  decoder_input_ids=decoder_input_ids,
                               decoder_attention_mask=decoder_attention_mask, encoder_outputs=encoder_outputs,
                               past_key_values=past_key_values, inputs_embeds=inputs_embeds, labels=labels,
                               decoder_inputs_embeds=decoder_inputs_embeds, use_cache=use_cache, return_dict=return_dict,
                               output_attentions=output_attentions, output_hidden_states=output_hidden_states,
                               react_rel_input_ids=react_rel_input_ids,
                               react_rel_attention_mask=react_rel_attention_mask,
                               react_rel_token_type_ids=react_rel_token_type_ids,
                               social_rel_input_ids=social_rel_input_ids,
                               social_rel_attention_mask=social_rel_attention_mask,
                               social_rel_token_type_ids=social_rel_token_type_ids,
                               event_rel_input_ids=event_rel_input_ids,
                               event_rel_attention_mask=event_rel_attention_mask,
                               event_rel_token_type_ids=event_rel_token_type_ids,
                               entity_rel_input_ids=entity_rel_input_ids,
                               entity_rel_attention_mask=entity_rel_attention_mask,
                               entity_rel_token_type_ids=entity_rel_token_type_ids,
                               example_0_input_ids=example_0_input_ids,
                               example_0_attention_mask=example_0_attention_mask,
                               example_0_token_type_ids=example_0_token_type_ids,
                               example_1_input_ids=example_1_input_ids,
                               example_1_attention_mask=example_1_attention_mask,
                               example_1_token_type_ids=example_1_token_type_ids,
                               example_2_input_ids=example_2_input_ids,
                               example_2_attention_mask=example_2_attention_mask,
                               example_2_token_type_ids=example_2_token_type_ids,
                               example_3_input_ids=example_3_input_ids,
                               example_3_attention_mask=example_3_attention_mask,
                               example_3_token_type_ids=example_3_token_type_ids,
                               example_4_input_ids=example_4_input_ids,
                               example_4_attention_mask=example_4_attention_mask,
                               example_4_token_type_ids=example_4_token_type_ids, **kwargs)

    def encode_context(self,
                       input_ids: Optional[torch.LongTensor] = None,
                       attention_mask: Optional[torch.FloatTensor] = None,
                       inputs_embeds: Optional[torch.FloatTensor] = None,
                       output_attentions: Optional[bool] = None,
                       output_hidden_states: Optional[bool] = None,
                       return_dict: Optional[bool] = None,
                       audio_input_values=None,
                       audio_attention_mask=None,
                       **kwargs):
        """
        encode acoustic and textual context
        :param input_ids:
        :param attention_mask:
        :param inputs_embeds:
        :param output_attentions:
        :param output_hidden_states:
        :param return_dict:
        :param audio_input_values:
        :param audio_attention_mask:
        :param kwargs:
        :return:
        """
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        encoded_audio_output = self.data2vec_audio(
            input_values=audio_input_values,
            attention_mask=audio_attention_mask,
            return_dict=True,
        )

        # integrate encoded with audio

        integrated_output = self.text_audio_integrator(text_embedding=encoder_outputs.last_hidden_state,
                                                       acoustic_embedding=encoded_audio_output.last_hidden_state)

        encoder_outputs['last_hidden_state'] = integrated_output
        return encoder_outputs

