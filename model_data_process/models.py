from abc import ABC
import torch.nn
from transformers import PretrainedConfig, AutoModel, AutoModelForCausalLM, EncoderDecoderModel, EncoderDecoderConfig,\
    AutoConfig, PreTrainedModel, AutoModelForSequenceClassification, AlbertModel
import enum
from typing import Optional

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


class KnowledgesEncoder(PreTrainedModel):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.encoder = AlbertModel.from_pretrained("albert-base-v2")
        self.social_event_attention = torch.nn.MultiheadAttention(embed_dim=768, num_heads=10, dropout=0.2)
        self.social_entity_attention = torch.nn.MultiheadAttention(embed_dim=768, num_heads=10, dropout=0.2)

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
                                       social_entity_attention_output]), dim=-1) + encoded_react_knw


class Roberta2DialoGPT(EncoderDecoderModel, ABC):
    """
    it isn't necessary to make a new class,
    this class is written to change lately on initial and forward functions
    """

    def __init__(self, bos_token_id=0, eos_token_id=2, pad_token_id=50266,
                 config: PretrainedConfig = None, embedding_tokens_len=50267,
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


class EmotionRoberta2DialoGPT(MultiTaskModel):

    def __init__(self,
                 config: PretrainedConfig = PretrainedConfig(),
                 *inputs,
                 **kwargs):
        kwargs['bos_token_id'] = kwargs.get('bos_token_id', 0)
        kwargs['eos_token_id'] = kwargs.get('eos_token_id', 2)
        kwargs['pad_token_id'] = kwargs.get('pad_token_id', 50266)
        kwargs['embedding_tokens_len'] = kwargs.get('embedding_tokens_len', 50267)
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
        self.encoder_decoder = Roberta2DialoGPT(bos_token_id=kwargs['bos_token_id'],
                                                eos_token_id=kwargs['eos_token_id'],
                                                pad_token_id=kwargs['pad_token_id'],
                                                config=kwargs['encoder_decoder_config'],
                                                embedding_tokens_len=kwargs['embedding_tokens_len'])

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
                emotion_labels=None, labels=None,  **kwargs) -> BaseMultiTaskOutput:
        """
        rewrite this function to show its arguments and avoid empty batch Error
        :param input_ids:
        :param attention_mask:
        :param token_type_ids:
        :param emotion_labels:
        :param labels:
        :param kwargs:
        :return:
        """
        return super().forward(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                               emotion_labels=emotion_labels, labels=labels, **kwargs)
