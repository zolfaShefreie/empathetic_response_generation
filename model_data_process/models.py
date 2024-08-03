from abc import ABC
import torch.nn
from transformers import PretrainedConfig, AutoModel, AutoModelForCausalLM, EncoderDecoderModel, EncoderDecoderConfig, \
    AutoConfig, PreTrainedModel, AutoModelForSequenceClassification, AlbertModel, Data2VecAudioModel, RobertaModel
import warnings
import enum
from typing import Optional, Tuple, Union
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import Seq2SeqLMOutput, BaseModelOutput, SequenceClassifierOutput, \
    BaseModelOutputWithPoolingAndCrossAttentions
from transformers.models.encoder_decoder.modeling_encoder_decoder import DEPRECATION_WARNING, shift_tokens_right
import torch.nn.functional as F


from utils.models import MultiTaskModel, BaseMultiTaskOutput


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

        return torch.mean(torch.stack([encoded_social_knw,
                                       social_event_attention_output,
                                       social_entity_attention_output]), dim=0) + encoded_react_knw


class TextualContextEncoder(PreTrainedModel):

    def __init__(self, config: PretrainedConfig = PretrainedConfig(), embedding_tokens_len=50267,
                 kwn_embedding_tokens_len=50266,
                 *inputs, **kwargs):

        super().__init__(config=config, *inputs, **kwargs)

        self.config.hidden_size = 768
        self.config.is_decoder = False

        self.roberta = RobertaModel.from_pretrained('roberta-base')
        if embedding_tokens_len:
            self.roberta.resize_token_embeddings(embedding_tokens_len)
        self.knowledge_encoder = KnowledgesEncoder(kwn_embedding_tokens_len=kwn_embedding_tokens_len)
        self.example_encoders = ExampleEncoder()
        self.norm_layer = torch.nn.LayerNorm(768)

    def forward(self,
                input_ids: Optional[torch.LongTensor] = None,
                attention_mask: Optional[torch.FloatTensor] = None,
                token_type_ids: Optional[torch.Tensor] = None,
                past_key_values: Tuple[Tuple[torch.FloatTensor]] = None,
                inputs_embeds: Optional[torch.FloatTensor] = None,
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
                **kwargs):

        encoder_outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            token_type_ids=token_type_ids,
            past_key_values=past_key_values,
            return_dict=True,
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

        last_hidden_state = torch.sum(torch.stack([encoder_outputs.last_hidden_state,
                                                   encoded_knowledge]), dim=0)
        if encoded_examples is not None:
            last_hidden_state = torch.sum(torch.stack([encoder_outputs.last_hidden_state,
                                                       encoded_examples]), dim=0)

        last_hidden_state = self.norm_layer(last_hidden_state)

        if not return_dict:
            return (last_hidden_state, encoder_outputs.pooler_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=last_hidden_state,
            pooler_output=encoder_outputs.pooler_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )


class TextualResponseGenerator(EncoderDecoderModel, ABC):
    """
    it isn't necessary to make a new class,
    this class is written to change lately on initial and forward functions
    """

    def __init__(self, bos_token_id=0, eos_token_id=2, pad_token_id=50266,
                 config: PretrainedConfig = None, embedding_tokens_len=50267,
                 kwn_embedding_tokens_len=50266,
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

        encoder = TextualContextEncoder(embedding_tokens_len=embedding_tokens_len,
                                        kwn_embedding_tokens_len=kwn_embedding_tokens_len)
        config.encoder = encoder.config

        super().__init__(config=config, encoder=encoder, decoder=decoder, *inputs, **kwargs)

        # self.encoder = TextualContextEncoder(embedding_tokens_len=embedding_tokens_len,
        #                                      kwn_embedding_tokens_len=kwn_embedding_tokens_len)
        # self.config.encoder = self.encoder.config

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
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
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
                example_4_token_type_ids=example_4_token_type_ids,
                **kwargs_encoder,
            )

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
        loss = None
        if labels is not None:
            warnings.warn(DEPRECATION_WARNING, FutureWarning)
            logits = decoder_outputs.logits if return_dict else decoder_outputs[0]
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.reshape(-1, self.decoder.config.vocab_size), labels.view(-1))

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


class EmotionRoberta2DialoGPT(MultiTaskModel):

    def __init__(self,
                 config: PretrainedConfig = PretrainedConfig(),
                 *inputs,
                 **kwargs):
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
                                                        kwn_embedding_tokens_len=kwargs['kwn_embedding_tokens_len'])

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
        setattr(self.emotion_classifier, 'roberta', encoder.roberta)

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

        # apply self-attention on acoustic embeddings
        new_nv = self.multihead_attn(acoustic_embedding, acoustic_embedding, acoustic_embedding)[0] + acoustic_embedding
        att_audio_embedding = self.dropout(self.AV_LayerNorm(new_nv))

        # shifting text embeddings
        weight_av = F.relu(self.W_hav(torch.cat((att_audio_embedding, text_embedding), dim=-1)))

        h_m = weight_av * self.W_av(att_audio_embedding)

        em_norm = text_embedding.norm(2, dim=-1)
        hm_norm = h_m.norm(2, dim=-1)

        hm_norm_ones = torch.ones(hm_norm.shape, requires_grad=True)
        hm_norm = torch.where(hm_norm == 0, hm_norm_ones, hm_norm)

        thresh_hold = (em_norm / (hm_norm + eps)) * self.beta_shift

        ones = torch.ones(thresh_hold.shape, requires_grad=True)

        alpha = torch.min(thresh_hold, ones)
        alpha = alpha.unsqueeze(dim=-1)

        acoustic_vis_embedding = alpha * h_m

        embedding_output = self.dropout(
            self.LayerNorm(acoustic_vis_embedding + text_embedding)
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
                                  return_dict=True).last_hidden_state[:, -1, :]
        audio_embed = self.data2vec_audio(input_values=audio_input_values, attention_mask=audio_attention_mask,
                                          return_dict=True).last_hidden_state[:, 0, :]
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
