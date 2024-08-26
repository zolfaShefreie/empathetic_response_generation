from model_data_process.models import MultiModalResponseGenerator, TextualResponseGenerator, EmotionRoberta2DialoGPT, \
    MultiModelEmotionClassifier
from model_data_process.dataset import BiMEmpDialoguesDataset, EmpatheticDialoguesDataset, MELDDataset
from utils.preprocessing import Pipeline, ConversationFormatter, ConversationTokenizer, TextCleaner, ToTensor, \
    ToNumpy, ToLong, KnowledgeFormatter, KnowledgeTokenizer, FilterSample, PreProcessEncoderDecoderInputDictVersion, \
    ExampleTokenizer, AudioFeatureExtractor
from settings import DEFAULT_SAVE_DIR_PREFIX, HUB_ACCESS_TOKEN, MELD_DATASET_PATH, \
    HUB_BIMODEL_ID, HUB_BIMODEL_PRIVATE_REPO, BMEDIALOGUES_PATH, HUB_EMO_TEXT_MODEL_ID, HUB_EMO_TEXT_PRIVATE_REPO, \
    HUB_CLASSIFIER_MODEL_ID, HUB_CLASSIFIER_PRIVATE_REPO, HUB_TEXT_MODEL_ID, HUB_TEXT_PRIVATE_REPO
from utils.metrics import Metrics
from utils.trainer import Seq2SeqTrainerMultiLoss, MultiTaskTrainer

from transformers import RobertaTokenizer, AlbertTokenizer, AutoFeatureExtractor, Trainer


class MultiModalResponseGeneratorConfig:

    def __init__(self, source_max_len: int = 300, target_max_len: int = 100, *args, **kwargs):
        """
        initial of class obj
        :param source_max_len:
        :param target_max_len:
        """
        self.DatasetClass = BiMEmpDialoguesDataset
        self.ModelClass = MultiModalResponseGenerator
        self.TrainerClass = Seq2SeqTrainerMultiLoss

        self.CONVERSATION_TOKENIZER = ConversationTokenizer(tokenizer=RobertaTokenizer.from_pretrained("roberta-base"),
                                                            source_max_len=source_max_len,
                                                            label_max_len=target_max_len,
                                                            new_special_tokens={
                                                               'pad_token': '<pad>'
                                                            },
                                                            last_utter_key_name='last_utter',
                                                            history_key_name='history',
                                                            gen_label_key_name='label',
                                                            context_ids_key_name='input_ids',
                                                            context_mask_key_name='attention_mask',
                                                            context_token_type_key_name='token_type_ids',
                                                            gen_label_ids_key_name='labels')

        self.KNOWLEDGE_TOKENIZER = KnowledgeTokenizer(tokenizer=AlbertTokenizer.from_pretrained("albert-base-v2"),
                                                      react_key_name='react_rel',
                                                      social_rel_key_name='social_rel',
                                                      event_rel_key_name='event_rel',
                                                      entity_rel_key_name='entity_rel',
                                                      max_len=source_max_len,
                                                      use_special_tokens=True)

        self.EXAMPLE_TOKENIZER = ExampleTokenizer(tokenizer=AlbertTokenizer.from_pretrained("albert-base-v2"),
                                                  example_key_name='examples',
                                                  max_len=source_max_len)

        self.TRANSFORMS = Pipeline(functions=[
            TextCleaner(texts_key_name='history'),
            ConversationFormatter(history_key_name='history',
                                  gen_label_key_name='label',
                                  last_utter_key_name='last_utter',
                                  utter_sep=' '),
            KnowledgeFormatter(social_rel_key_name='social_rel',
                               event_rel_key_name='event_rel',
                               entity_rel_key_name='entity_rel',
                               react_rel_key_name='react_rel',
                               use_special_tokens=True),
            ToNumpy(unwanted_keys=['audio']),
            self.CONVERSATION_TOKENIZER,
            self.KNOWLEDGE_TOKENIZER,
            self.EXAMPLE_TOKENIZER,
            AudioFeatureExtractor(
                feature_extractor=AutoFeatureExtractor.from_pretrained("facebook/data2vec-audio-base-960h"),
                audio_key_name='audio',
                result_prefix_key_name='audio'),
            FilterSample(wanted_keys=['input_ids', 'attention_mask', 'token_type_ids', 'labels', 'audio_input_values',
                                      'audio_attention_mask'] +
                                     [f"{rel_name}_{suffix}" for rel_name in
                                      ['react_rel', 'social_rel', 'event_rel', 'entity_rel']
                                      for suffix in ['input_ids', 'attention_mask', 'token_type_ids']] +
                                     [f"example_{i}_{suffix}" for i in range(0, 5)
                                      for suffix in ['input_ids', 'attention_mask', 'token_type_ids']]),
            ToTensor(),
            PreProcessEncoderDecoderInputDictVersion(tokenizer=self.CONVERSATION_TOKENIZER.tokenizer,
                                                     gen_label_key_name='labels'),
            ToLong(wanted_list=['input_ids', 'attention_mask', 'token_type_ids', 'labels', 'audio_attention_mask'] +
                               [f"{rel_name}_{suffix}" for rel_name in ['react_rel', 'social_rel', 'event_rel',
                                                                        'entity_rel']
                                for suffix in ['input_ids', 'attention_mask', 'token_type_ids']] +
                               [f"example_{i}_{suffix}" for i in range(0, 5)
                                for suffix in ['input_ids', 'attention_mask', 'token_type_ids']]),
        ])

    def dataset_args(self, split: str = 'train'):
        """
        get dataset new object args
        :return:
        """
        return {'transform': self.TRANSFORMS, 'dataset_dir': BMEDIALOGUES_PATH, 'split': split}

    def model_args(self, div_loss_weight=1.5, main_loss_weight=1, empathy_loss_weight=0.1):
        """
        get model new object args
        :return:
        """
        return {'embedding_tokens_len': len(self.CONVERSATION_TOKENIZER.tokenizer),
                'special_token_dict': {each: self.CONVERSATION_TOKENIZER.tokenizer(each, add_special_tokens=False)['input_ids'][0]
                                       for each in self.CONVERSATION_TOKENIZER.tokenizer.all_special_tokens},
                'bos_token_id': self.CONVERSATION_TOKENIZER.tokenizer.bos_token_id,
                'eos_token_id': self.CONVERSATION_TOKENIZER.tokenizer.eos_token_id,
                'pad_token_id': self.CONVERSATION_TOKENIZER.tokenizer.pad_token_id,
                'kwn_embedding_tokens_len': len(self.KNOWLEDGE_TOKENIZER.tokenizer),
                'div_loss_weight': div_loss_weight,
                'main_loss_weight': main_loss_weight,
                'empathy_loss_weight': empathy_loss_weight}

    @staticmethod
    def hub_args():
        """
        :return:
        """
        return {
            'hub_model_id': HUB_BIMODEL_ID,
            'hub_private_repo': HUB_BIMODEL_PRIVATE_REPO,
            'hub_token': HUB_ACCESS_TOKEN
        }

    @staticmethod
    def default_save_dir():
        """
        :return:
        """
        return f"{DEFAULT_SAVE_DIR_PREFIX}/empathetic_spoken_chatbot"

    def metric_func(self):
        """
        :return:
        """
        return Metrics(tokenizer=self.CONVERSATION_TOKENIZER.tokenizer, task_list=['text_generator', ]).compute


class TextualResponseGeneratorConfig:

    def __init__(self, source_max_len: int = 300, target_max_len: int = 100, *args, **kwargs):
        """

        :param source_max_len:
        :param target_max_len:
        """
        self.DatasetClass = BiMEmpDialoguesDataset
        self.ModelClass = TextualResponseGenerator
        self.TrainerClass = Seq2SeqTrainerMultiLoss

        self.CONVERSATION_TOKENIZER = ConversationTokenizer(tokenizer=RobertaTokenizer.from_pretrained("roberta-base"),
                                                            source_max_len=source_max_len,
                                                            label_max_len=target_max_len,
                                                            new_special_tokens={
                                                                'pad_token': '<pad>'
                                                            },
                                                            last_utter_key_name='last_utter',
                                                            history_key_name='history',
                                                            gen_label_key_name='label',
                                                            context_ids_key_name='input_ids',
                                                            context_mask_key_name='attention_mask',
                                                            context_token_type_key_name='token_type_ids',
                                                            gen_label_ids_key_name='labels')

        self.KNOWLEDGE_TOKENIZER = KnowledgeTokenizer(tokenizer=AlbertTokenizer.from_pretrained("albert-base-v2"),
                                                      react_key_name='react_rel',
                                                      social_rel_key_name='social_rel',
                                                      event_rel_key_name='event_rel',
                                                      entity_rel_key_name='entity_rel',
                                                      max_len=source_max_len,
                                                      use_special_tokens=True)

        self.EXAMPLE_TOKENIZER = ExampleTokenizer(tokenizer=AlbertTokenizer.from_pretrained("albert-base-v2"),
                                                  example_key_name='examples',
                                                  max_len=source_max_len)

        self.TRANSFORMS = Pipeline(functions=[
            TextCleaner(texts_key_name='history'),
            ConversationFormatter(history_key_name='history',
                                  gen_label_key_name='label',
                                  last_utter_key_name='last_utter',
                                  utter_sep=' '),
            KnowledgeFormatter(social_rel_key_name='social_rel',
                               event_rel_key_name='event_rel',
                               entity_rel_key_name='entity_rel',
                               react_rel_key_name='react_rel',
                               use_special_tokens=True),
            ToNumpy(),
            self.CONVERSATION_TOKENIZER,
            self.KNOWLEDGE_TOKENIZER,
            self.EXAMPLE_TOKENIZER,
            FilterSample(wanted_keys=['input_ids', 'attention_mask', 'token_type_ids', 'labels', ] +
                                     [f"{rel_name}_{suffix}" for rel_name in
                                      ['react_rel', 'social_rel', 'event_rel', 'entity_rel']
                                      for suffix in ['input_ids', 'attention_mask', 'token_type_ids']] +
                                     [f"example_{i}_{suffix}" for i in range(0, 5)
                                      for suffix in ['input_ids', 'attention_mask', 'token_type_ids']]),
            ToTensor(),
            PreProcessEncoderDecoderInputDictVersion(tokenizer=self.CONVERSATION_TOKENIZER.tokenizer,
                                                     gen_label_key_name='labels'),
            ToLong(wanted_list=['input_ids', 'attention_mask', 'token_type_ids', 'labels', ] +
                               [f"{rel_name}_{suffix}" for rel_name in ['react_rel', 'social_rel', 'event_rel',
                                                                        'entity_rel']
                                for suffix in ['input_ids', 'attention_mask', 'token_type_ids']] +
                               [f"example_{i}_{suffix}" for i in range(0, 5)
                                for suffix in ['input_ids', 'attention_mask', 'token_type_ids']]),
        ])

    def dataset_args(self, split: str = 'train'):
        """
        without split
        :return:
        """
        return {'transform': self.TRANSFORMS, 'dataset_dir': BMEDIALOGUES_PATH, 'split': split, 'include_audio': False}

    def model_args(self, div_loss_weight=1.5, main_loss_weight=1, empathy_loss_weight=0.1):
        """
        get model new object args
        :return:
        """
        return {'embedding_tokens_len': len(self.CONVERSATION_TOKENIZER.tokenizer),
                'special_token_dict': {each: self.CONVERSATION_TOKENIZER.tokenizer(each, add_special_tokens=False)['input_ids'][0]
                                       for each in self.CONVERSATION_TOKENIZER.tokenizer.all_special_tokens},
                'bos_token_id': self.CONVERSATION_TOKENIZER.tokenizer.bos_token_id,
                'eos_token_id': self.CONVERSATION_TOKENIZER.tokenizer.eos_token_id,
                'pad_token_id': self.CONVERSATION_TOKENIZER.tokenizer.pad_token_id,
                'kwn_embedding_tokens_len': len(self.KNOWLEDGE_TOKENIZER.tokenizer),
                'div_loss_weight': div_loss_weight,
                'main_loss_weight': main_loss_weight,
                'empathy_loss_weight': empathy_loss_weight}

    @staticmethod
    def hub_args():
        """
        :return:
        """
        return {
            'hub_model_id': HUB_TEXT_MODEL_ID,
            'hub_private_repo': HUB_TEXT_PRIVATE_REPO,
            'hub_token': HUB_ACCESS_TOKEN
        }

    @staticmethod
    def default_save_dir():
        """
        :return:
        """
        return f"{DEFAULT_SAVE_DIR_PREFIX}/empathetic_chatbot"

    def metric_func(self):
        """
        :return:
        """
        return Metrics(tokenizer=self.CONVERSATION_TOKENIZER.tokenizer, task_list=['text_generator', ]).compute


class EmotionalTextualResponseGeneratorConfig:

    def __init__(self, source_max_len: int = 300, target_max_len: int = 100, *args, **kwargs):
        """

        :param source_max_len:
        :param target_max_len:
        """
        self.DatasetClass = EmpatheticDialoguesDataset
        self.ModelClass = EmotionRoberta2DialoGPT
        self.TrainerClass = MultiTaskTrainer

        self.CONVERSATION_TOKENIZER = ConversationTokenizer(tokenizer=RobertaTokenizer.from_pretrained("roberta-base"),
                                                            source_max_len=source_max_len,
                                                            label_max_len=target_max_len,
                                                            new_special_tokens={
                                                                'pad_token': '<pad>'
                                                            },
                                                            last_utter_key_name='last_utter',
                                                            history_key_name='history',
                                                            gen_label_key_name='label',
                                                            context_ids_key_name='input_ids',
                                                            context_mask_key_name='attention_mask',
                                                            context_token_type_key_name='token_type_ids',
                                                            gen_label_ids_key_name='labels')

        self.KNOWLEDGE_TOKENIZER = KnowledgeTokenizer(tokenizer=AlbertTokenizer.from_pretrained("albert-base-v2"),
                                                      react_key_name='react_rel',
                                                      social_rel_key_name='social_rel',
                                                      event_rel_key_name='event_rel',
                                                      entity_rel_key_name='entity_rel',
                                                      max_len=source_max_len,
                                                      use_special_tokens=True)

        self.EXAMPLE_TOKENIZER = ExampleTokenizer(tokenizer=AlbertTokenizer.from_pretrained("albert-base-v2"),
                                                  example_key_name='examples',
                                                  max_len=source_max_len)

        self.TRANSFORMS = Pipeline(functions=[
            TextCleaner(texts_key_name='history'),
            ConversationFormatter(history_key_name='history',
                                  gen_label_key_name='label',
                                  last_utter_key_name='last_utter',
                                  utter_sep=' '),
            KnowledgeFormatter(social_rel_key_name='social_rel',
                               event_rel_key_name='event_rel',
                               entity_rel_key_name='entity_rel',
                               react_rel_key_name='react_rel',
                               use_special_tokens=True),
            ToNumpy(),
            self.CONVERSATION_TOKENIZER,
            self.KNOWLEDGE_TOKENIZER,
            self.EXAMPLE_TOKENIZER,
            FilterSample(wanted_keys=['input_ids', 'attention_mask', 'token_type_ids', 'labels', 'emotion_labels'] +
                                     [f"{rel_name}_{suffix}" for rel_name in
                                      ['react_rel', 'social_rel', 'event_rel', 'entity_rel']
                                      for suffix in ['input_ids', 'attention_mask', 'token_type_ids']] +
                                     [f"example_{i}_{suffix}" for i in range(0, 5)
                                      for suffix in ['input_ids', 'attention_mask', 'token_type_ids']]),
            ToTensor(),
            PreProcessEncoderDecoderInputDictVersion(tokenizer=self.CONVERSATION_TOKENIZER.tokenizer,
                                                     gen_label_key_name='labels'),
            ToLong(wanted_list=['input_ids', 'attention_mask', 'token_type_ids', 'labels', 'emotion_labels'] +
                               [f"{rel_name}_{suffix}" for rel_name in ['react_rel', 'social_rel', 'event_rel',
                                                                        'entity_rel']
                                for suffix in ['input_ids', 'attention_mask', 'token_type_ids']] +
                               [f"example_{i}_{suffix}" for i in range(0, 5)
                                for suffix in ['input_ids', 'attention_mask', 'token_type_ids']]),
        ])

    def dataset_args(self, split: str = 'train'):
        """
        :return:
        """
        return {'split': split, 'transform': self.TRANSFORMS}

    def model_args(self, div_loss_weight=1.5, main_loss_weight=1, empathy_loss_weight=0.1):
        """
        :param div_loss_weight:
        :param main_loss_weight:
        :param empathy_loss_weight:
        :return:
        """
        return {'embedding_tokens_len': len(self.CONVERSATION_TOKENIZER.tokenizer),
                'special_token_dict': {each: self.CONVERSATION_TOKENIZER.tokenizer(each, add_special_tokens=False)['input_ids'][0]
                                       for each in self.CONVERSATION_TOKENIZER.tokenizer.all_special_tokens},
                'bos_token_id': self.CONVERSATION_TOKENIZER.tokenizer.bos_token_id,
                'eos_token_id': self.CONVERSATION_TOKENIZER.tokenizer.eos_token_id,
                'pad_token_id': self.CONVERSATION_TOKENIZER.tokenizer.pad_token_id,
                'num_labels': 32,
                'kwn_embedding_tokens_len': len(self.KNOWLEDGE_TOKENIZER.tokenizer),
                'div_loss_weight': div_loss_weight,
                'main_loss_weight': main_loss_weight,
                'empathy_loss_weight': empathy_loss_weight}

    @staticmethod
    def hub_args():
        """
        :return:
        """
        return {
            'hub_model_id': HUB_EMO_TEXT_MODEL_ID,
            'hub_private_repo': HUB_EMO_TEXT_PRIVATE_REPO,
            'hub_token': HUB_ACCESS_TOKEN
        }

    @staticmethod
    def default_save_dir():
        """
        :return:
        """
        return f"{DEFAULT_SAVE_DIR_PREFIX}/emotional_empathetic_chatbot"

    def metric_func(self):
        """
        :return:
        """
        return Metrics(tokenizer=self.CONVERSATION_TOKENIZER.tokenizer,
                       task_list=['text_generator', 'classifier']).compute


class MultiModelEmotionClassifierConfig:

    def __init__(self, source_max_len: int = 300, *args, **kwargs):

        self.DatasetClass = MELDDataset
        self.ModelClass = MultiModelEmotionClassifier
        self.TrainerClass = Trainer

        self.CONVERSATION_TOKENIZER = ConversationTokenizer(tokenizer=RobertaTokenizer.from_pretrained("roberta-base"),
                                                            source_max_len=source_max_len,
                                                            new_special_tokens={
                                                               'additional_special_tokens': [
                                                                   ConversationFormatter.SPECIAL_TOKEN_SPLIT_UTTERANCE, ],
                                                               'pad_token': '[PAD]'},
                                                            last_utter_key_name='last_utter',
                                                            history_key_name='history',
                                                            gen_label_key_name='label',
                                                            context_ids_key_name='input_ids',
                                                            context_mask_key_name='attention_mask',
                                                            context_token_type_key_name='token_type_ids',
                                                            gen_label_ids_key_name=None)

        self.TRANSFORMS = Pipeline(functions=[
            TextCleaner(texts_key_name='history'),
            ConversationFormatter(history_key_name='history',
                                  last_utter_key_name='last_utter'),
            ToNumpy(unwanted_keys=['audio']),
            self.CONVERSATION_TOKENIZER,
            AudioFeatureExtractor(
                feature_extractor=AutoFeatureExtractor.from_pretrained("facebook/data2vec-audio-base-960h"),
                audio_key_name='audio',
                result_prefix_key_name='audio'),
            FilterSample(wanted_keys=['input_ids', 'attention_mask', 'token_type_ids', 'audio_input_values',
                                      'audio_attention_mask', 'labels', ]),
            ToTensor(),
            ToLong(wanted_list=['input_ids', 'attention_mask', 'token_type_ids', 'labels', 'audio_attention_mask']),
        ])

    def dataset_args(self, split: str = 'train'):
        """
        without split
        :return:
        """
        return {'split': split, 'transform': self.TRANSFORMS, 'dataset_path': MELD_DATASET_PATH}

    def model_args(self):
        """
        get model new object args
        :return:
        """
        return {'num_classes': 7, 'embedding_tokens_len': len(self.CONVERSATION_TOKENIZER.tokenizer)}

    @staticmethod
    def hub_args():
        """
        :return:
        """
        return {
            'hub_model_id': HUB_CLASSIFIER_MODEL_ID,
            'hub_private_repo': HUB_CLASSIFIER_PRIVATE_REPO,
            'hub_token': HUB_ACCESS_TOKEN
        }

    @staticmethod
    def default_save_dir():
        """
        :return:
        """
        return f"{DEFAULT_SAVE_DIR_PREFIX}/emotion_recognition"

    def metric_func(self):
        """
        :return:
        """
        return Metrics(tokenizer=self.CONVERSATION_TOKENIZER.tokenizer, task_list=['classifier', ]).compute

