from abc import ABC
from transformers import PretrainedConfig, AutoModel, AutoModelForCausalLM, EncoderDecoderModel, EncoderDecoderConfig,\
    AutoConfig
import enum


class ModelType(enum.Enum):
    roberta_shared = 'roberta_shared'
    roberta_gpt2 = 'roberta_gpt2'


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

    def forward(
                self,
                input_ids=None,
                attention_mask=None,
                decoder_input_ids=None,
                decoder_attention_mask=None,
                encoder_outputs=None,
                past_key_values=None,
                inputs_embeds=None,
                decoder_inputs_embeds=None,
                labels=None,
                use_cache=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None,
                **kwargs):
        return super().forward(input_ids=input_ids,
                                   attention_mask=attention_mask,
                                   decoder_input_ids=decoder_input_ids,
                                   decoder_attention_mask=decoder_attention_mask,
                                   encoder_outputs=encoder_outputs,
                                   past_key_values=past_key_values,
                                   inputs_embeds=inputs_embeds,
                                   decoder_inputs_embeds=decoder_inputs_embeds,
                                   labels=labels,
                                   use_cache=use_cache,
                                   output_attentions=output_attentions,
                                   output_hidden_states=output_hidden_states,
                                   return_dict=return_dict,
                                   **kwargs
                                   )

