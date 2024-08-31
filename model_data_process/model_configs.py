from transformers import PretrainedConfig, AutoConfig, AutoModel, AutoModelForCausalLM, EncoderDecoderConfig


class KnowledgeEncoderConfig(PretrainedConfig):
    model_type = "knowledge_encoder"

    def __init__(self, kwn_embedding_tokens_len=30016, social_event_num_heads=8, social_event_dropout=0.2,
                 social_entity_num_heads=8, social_entity_dropout=0.2, **kwargs):
        super().__init__(**kwargs)
        self.kwn_embedding_tokens_len = kwn_embedding_tokens_len
        self.social_event_num_heads = social_event_num_heads
        self.social_event_dropout = social_event_dropout
        self.social_entity_num_heads = social_entity_num_heads
        self.social_entity_dropout = social_entity_dropout


class TextualResponseGeneratorConfig(EncoderDecoderConfig, KnowledgeEncoderConfig):
    model_type = 'textual_response_generator'

    def __init__(self, special_token_dict: dict, bos_token_id=0, eos_token_id=2, pad_token_id=50266,
                 embedding_tokens_len=50267, empathy_loss_weight=0.1, main_loss_weight=1, div_loss_weight=1.5,
                 ** kwargs):
        encoder_decoder_args = self.initial_encoder_decoder(embedding_tokens_len=embedding_tokens_len)
        EncoderDecoderConfig.__init__(self, **encoder_decoder_args)
        KnowledgeEncoderConfig.__init__(self, **kwargs)
        self.decoder_start_token_id = bos_token_id
        self.special_token_dict = special_token_dict
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.embedding_tokens_len = embedding_tokens_len
        self.empathy_loss_weight = empathy_loss_weight
        self.main_loss_weight = main_loss_weight
        self.div_loss_weight = div_loss_weight

        # sensible parameters for beam search
        # set decoding params
        self.max_new_tokens = 64
        self.min_length = 2
        self.early_stopping = True
        self.no_repeat_ngram_size = 3
        self.length_penalty = 2.0
        self.num_beams = 4
        self.vocab_size = self.encoder.vocab_size
        self.is_encoder_decoder = True

    def initial_encoder_decoder(self, embedding_tokens_len):
        result = dict()

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

        decoder.config.is_decoder = True
        decoder.config.add_cross_attention = True

        result['encoder'] = encoder.config.to_dict()
        result['decoder'] = decoder.config.to_dict()
        return result


class EmotionRoberta2DialoGPTConfig(TextualResponseGeneratorConfig):

    def __init__(self, num_labels=32, **kwargs):
        super().__init__(**kwargs)
        self.num_labels = num_labels


class TextAudioIntegratorConfig(PretrainedConfig):
    model_type = 'text_audio_integrator'

    def __init__(self, hidden_size_integrator=768, beta_shift_integrator=1e-1, dropout_prob_integrator=0.2,
                 num_head_integrator=3, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size_integrator = hidden_size_integrator
        self.beta_shift_integrator = beta_shift_integrator
        self.dropout_prob_integrator = dropout_prob_integrator
        self.num_head_integrator = num_head_integrator


class MultiModelEmotionClassifierConfig(TextAudioIntegratorConfig):
    model_type = "text_audio_emo_classifier"

    def __init__(self, text_audio_emo_num_classes=7, embedding_tokens_len=50267, **kwargs):
        super().__init__(**kwargs)
        self.text_audio_emo_num_classes = text_audio_emo_num_classes
        self.embedding_tokens_len = embedding_tokens_len


class MultiModalResponseGeneratorConfig(TextualResponseGeneratorConfig, TextAudioIntegratorConfig):

    def __init__(self, **kwargs):
        TextualResponseGeneratorConfig.__init__(self, **kwargs)
        TextAudioIntegratorConfig.__init__(self, **kwargs)
        self.decoder_start_token_id = self.bos_token_id
        self.is_encoder_decoder = True
