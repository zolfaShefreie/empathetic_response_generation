from transformers.trainer_utils import PredictionOutput

from model_data_process.model_configs import EmotionRoberta2DialoGPTConfig, MultiModalResponseGeneratorConfig, \
    MultiModelEmotionClassifierConfig, TextualResponseGeneratorConfig
from model_data_process.models import MultiModalResponseGenerator, TextualResponseGenerator, EmotionRoberta2DialoGPT, \
    MultiModelEmotionClassifier
from model_data_process.dataset import BiMEmpDialoguesDataset, EmpatheticDialoguesDataset, MELDDataset
from utils.preprocessing import Pipeline, ConversationFormatter, ConversationTokenizer, TextCleaner, ToTensor, \
    ToNumpy, ToLong, KnowledgeFormatter, KnowledgeTokenizer, FilterSample, PreProcessEncoderDecoderInputDictVersion, \
    ExampleTokenizer, AudioFeatureExtractor, PostProcessResult
from settings import DEFAULT_SAVE_DIR_PREFIX, HUB_ACCESS_TOKEN, MELD_DATASET_PATH, \
    HUB_BIMODEL_MODEL_ID, HUB_BIMODEL_PRIVATE_REPO, BMEDIALOGUES_PATH, HUB_EMO_TEXT_MODEL_ID, HUB_EMO_TEXT_PRIVATE_REPO, \
    HUB_CLASSIFIER_MODEL_ID, HUB_CLASSIFIER_PRIVATE_REPO, HUB_TEXT_MODEL_ID, HUB_TEXT_PRIVATE_REPO
from utils.metrics import Metrics
from utils.trainer import Seq2SeqTrainerMultiLoss, MultiTaskTrainer

from transformers import RobertaTokenizer, AlbertTokenizer, AutoFeatureExtractor, Trainer, Seq2SeqTrainingArguments, \
    TrainingArguments, EvalPrediction, AutoTokenizer


class MultiModalResponseGeneratorMapping:

    def __init__(self, source_max_len: int = 300, target_max_len: int = 100,
                 include_knowledge: bool = True, include_example: bool = True, include_emp_losses: bool = True,
                 *args, **kwargs):
        """
        initial of class obj
        :param source_max_len:
        :param target_max_len:
        """
        self.DatasetClass = BiMEmpDialoguesDataset
        self.ModelClass = MultiModalResponseGenerator
        self.TrainerClass = Seq2SeqTrainerMultiLoss

        self.include_knowledge = include_knowledge
        self.include_example = include_example
        self.include_emp_losses = include_emp_losses

        tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
        self.CONVERSATION_TOKENIZER = ConversationTokenizer(
            source_tokenizer=RobertaTokenizer.from_pretrained("roberta-base"),
            target_tokenizer=tokenizer,
            source_max_len=source_max_len,
            label_max_len=target_max_len,
            source_new_special_tokens={'pad_token': '<pad>'},
            target_new_special_tokens={'pad_token': tokenizer.unk_token},
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
            TextCleaner(texts_key_name=['history', 'label', 'examples'], erase_punc=False),
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
            PreProcessEncoderDecoderInputDictVersion(tokenizer=self.CONVERSATION_TOKENIZER.target_tokenizer,
                                                     gen_label_key_name='labels'),
            ToLong(wanted_list=['input_ids', 'attention_mask', 'token_type_ids', 'labels', 'audio_attention_mask'] +
                               [f"{rel_name}_{suffix}" for rel_name in ['react_rel', 'social_rel', 'event_rel',
                                                                        'entity_rel']
                                for suffix in ['input_ids', 'attention_mask', 'token_type_ids']] +
                               [f"example_{i}_{suffix}" for i in range(0, 5)
                                for suffix in ['input_ids', 'attention_mask', 'token_type_ids']]),
        ])

        self.post_process = PostProcessResult(tokenizer=self.CONVERSATION_TOKENIZER.target_tokenizer,
                                              task_list=['text_generator', ])

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
        config = MultiModalResponseGeneratorConfig(
            embedding_tokens_len=len(self.CONVERSATION_TOKENIZER.source_tokenizer),
            decoder_vocab_size=len(self.CONVERSATION_TOKENIZER.target_tokenizer),
            special_token_dict={each: self.CONVERSATION_TOKENIZER.target_tokenizer(each, add_special_tokens=False)['input_ids'][0]
                                for each in self.CONVERSATION_TOKENIZER.target_tokenizer.all_special_tokens},
            bos_token_id=self.CONVERSATION_TOKENIZER.target_tokenizer.bos_token_id,
            eos_token_id=self.CONVERSATION_TOKENIZER.target_tokenizer.eos_token_id,
            pad_token_id=self.CONVERSATION_TOKENIZER.target_tokenizer.pad_token_id,
            kwn_embedding_tokens_len=len(self.KNOWLEDGE_TOKENIZER.tokenizer),
            div_loss_weight=div_loss_weight,
            main_loss_weight=main_loss_weight,
            empathy_loss_weight=empathy_loss_weight,
            hidden_size_integrator=768,
            include_knowledge=self.include_knowledge,
            include_example=self.include_example,
            include_emp_losses=self.include_emp_losses)
        return {'config': config}

    def hub_args(self):
        """
        :return:
        """
        model_id_suffix = f"_wo" if not self.include_knowledge or not self.include_example or \
                                    not self.include_emp_losses else ""

        if len(model_id_suffix) > 0:
            if not self.include_knowledge:
                model_id_suffix += '_knowledge'

            if not self.include_example:
                model_id_suffix += '_examples'

            if not self.include_emp_losses:
                model_id_suffix += '_emp_losses'

        return {
            'hub_model_id': f"{HUB_BIMODEL_MODEL_ID}{model_id_suffix}",
            'hub_private_repo': HUB_BIMODEL_PRIVATE_REPO,
            'hub_token': HUB_ACCESS_TOKEN
        }

    def default_save_dir(self):
        """
        :return:
        """
        suffix = f"_wo" if not self.include_knowledge or not self.include_example or \
                                    not self.include_emp_losses else ""

        if len(suffix) > 0:
            if not self.include_knowledge:
                suffix += '_knowledge'

            if not self.include_example:
                suffix += '_examples'

            if not self.include_emp_losses:
                suffix += '_emp_losses'

        return f"{DEFAULT_SAVE_DIR_PREFIX}/empathetic_spoken_chatbot{suffix}"

    def metric_func(self):
        """
        :return:
        """
        return Metrics(tokenizer=self.CONVERSATION_TOKENIZER.target_tokenizer, task_list=['text_generator', ]).compute

    def trainer_args_train(self, save_dir: str=None, evaluation_strategy: str = "epoch", eval_steps: int = 4,
                           save_steps: int = 4, logging_steps: int = 4, learning_rate: float = 1e-5,
                           save_strategy: str = "epoch", per_device_train_batch_size: int = 1,
                           per_device_eval_batch_size: int = 1, number_of_epochs: int = 2,
                           load_best_model_at_end: bool = True, save_total_limit: int = 2, push_to_hub: bool = True,
                           generation_config_path: str = None):

        return Seq2SeqTrainingArguments(
            generation_config=generation_config_path,
            predict_with_generate=True,
            output_dir=save_dir if save_dir is not None else self.default_save_dir(),
            overwrite_output_dir=True,
            evaluation_strategy=evaluation_strategy,
            eval_steps=eval_steps,
            save_steps=save_steps,
            logging_steps=logging_steps,
            do_train=True,
            do_eval=True,
            learning_rate=learning_rate,
            lr_scheduler_type='constant',
            save_strategy=save_strategy,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_eval_batch_size,
            num_train_epochs=number_of_epochs,
            report_to=['tensorboard'],

            # config for load and save best model
            load_best_model_at_end=load_best_model_at_end,
            save_total_limit=save_total_limit,
            metric_for_best_model='loss',
            greater_is_better=False,

            # hub configs
            push_to_hub=push_to_hub,
            hub_model_id=self.hub_args()['hub_model_id'],
            hub_private_repo=self.hub_args()['hub_private_repo'],
            hub_token=self.hub_args()['hub_token'],
            hub_strategy='end',
            # hub_strategy='checkpoint',
            # resume_from_checkpoint='last-checkpoint',
            save_safetensors=False,
        )

    def trainer_args_evaluate(self, save_dir: str = None,
                              logging_steps: int = 4,
                              per_device_eval_batch_size: int = 1,
                              push_to_hub: bool = True,
                              generation_config_path: str = None):

        return Seq2SeqTrainingArguments(
            generation_config=generation_config_path,
            predict_with_generate=True,
            output_dir=save_dir if save_dir is not None else self.default_save_dir(),
            overwrite_output_dir=True,
            logging_steps=logging_steps,
            do_eval=True,
            per_device_eval_batch_size=per_device_eval_batch_size,

            # hub configs
            push_to_hub=push_to_hub,
            hub_model_id=self.hub_args()['hub_model_id'],
            hub_private_repo=self.hub_args()['hub_private_repo'],
            hub_token=self.hub_args()['hub_token'],
            hub_strategy='end',
            # hub_strategy='checkpoint',
            # resume_from_checkpoint='last-checkpoint',
            save_safetensors=False,
        )

    def post_result(self, pred: PredictionOutput):
        result = self.post_process.compute(pred)
        result.update({'metric': pred.metrics})
        return result

    def initial_weights_from_other_models(self):
        weight_states = dict()
        mapping_keys = {EmotionalTextualResponseGeneratorMapping(): ['encoder', 'knowledge_encoder', 'example_encoders',
                                                                     'norm_layer', 'decoder'],
                        MultiModelEmotionClassifierMapping(): ['data2vec_audio', 'text_audio_integrator']}

        for mapping, key_of_layers in mapping_keys.items():

            try:
                model = mapping.ModelClass.from_pretrained(mapping.hub_args()['hub_model_id'],
                                                           token=mapping.hub_args()['hub_token'])
            except Exception as e1:
                print(f'exception from load model: {mapping.ModelClass.__name__}')
                print(e1)
                try:

                    model = mapping.ModelClass.from_pretrained(mapping.default_save_dir())
                except Exception as e2:
                    print(f'exception from load model: {mapping.ModelClass.__name__}')
                    print(e2)
                    raise Exception("can't load pretrained models")

            if isinstance(model, EmotionRoberta2DialoGPT):
                model = model.encoder_decoder

            weight_states.update({k: model.state_dict()[k]
                                  for k in model.state_dict()
                                  for key_layer in key_of_layers if k.startswith(f"{key_layer}.")})

        return weight_states


class TextualResponseGeneratorMapping:

    def __init__(self, source_max_len: int = 300, target_max_len: int = 100,
                 include_knowledge: bool = True, include_example: bool = True, include_emp_losses: bool = True,
                 *args, **kwargs):
        """

        :param source_max_len:
        :param target_max_len:
        """
        self.DatasetClass = BiMEmpDialoguesDataset
        self.ModelClass = TextualResponseGenerator
        self.TrainerClass = Seq2SeqTrainerMultiLoss

        self.include_knowledge = include_knowledge
        self.include_example = include_example
        self.include_emp_losses = include_emp_losses

        tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
        self.CONVERSATION_TOKENIZER = ConversationTokenizer(
            source_tokenizer=RobertaTokenizer.from_pretrained("roberta-base"),
            target_tokenizer=tokenizer,
            source_max_len=source_max_len,
            label_max_len=target_max_len,
            source_new_special_tokens={'pad_token': '<pad>'},
            target_new_special_tokens={'pad_token': tokenizer.unk_token},
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
            TextCleaner(texts_key_name=['history', 'label', 'examples'], erase_punc=False),
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
            PreProcessEncoderDecoderInputDictVersion(tokenizer=self.CONVERSATION_TOKENIZER.target_tokenizer,
                                                     gen_label_key_name='labels'),
            ToLong(wanted_list=['input_ids', 'attention_mask', 'token_type_ids', 'labels', ] +
                               [f"{rel_name}_{suffix}" for rel_name in ['react_rel', 'social_rel', 'event_rel',
                                                                        'entity_rel']
                                for suffix in ['input_ids', 'attention_mask', 'token_type_ids']] +
                               [f"example_{i}_{suffix}" for i in range(0, 5)
                                for suffix in ['input_ids', 'attention_mask', 'token_type_ids']]),
        ])

        self.post_process = PostProcessResult(tokenizer=self.CONVERSATION_TOKENIZER.target_tokenizer,
                                              task_list=['text_generator', ])

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
        config = TextualResponseGeneratorConfig(
            embedding_tokens_len=len(self.CONVERSATION_TOKENIZER.source_tokenizer),
            decoder_vocab_size=len(self.CONVERSATION_TOKENIZER.target_tokenizer),
            special_token_dict={each: self.CONVERSATION_TOKENIZER.target_tokenizer(each, add_special_tokens=False)['input_ids'][0]
                                for each in self.CONVERSATION_TOKENIZER.target_tokenizer.all_special_tokens},
            bos_token_id=self.CONVERSATION_TOKENIZER.target_tokenizer.bos_token_id,
            eos_token_id=self.CONVERSATION_TOKENIZER.target_tokenizer.eos_token_id,
            pad_token_id=self.CONVERSATION_TOKENIZER.target_tokenizer.pad_token_id,
            kwn_embedding_tokens_len=len(self.KNOWLEDGE_TOKENIZER.tokenizer),
            div_loss_weight=div_loss_weight,
            main_loss_weight=main_loss_weight,
            empathy_loss_weight=empathy_loss_weight,
            include_knowledge=self.include_knowledge,
            include_example=self.include_example,
            include_emp_losses=self.include_emp_losses)

        return {'config': config}

    def hub_args(self):
        """
        :return:
        """
        model_id_suffix = f"_wo" if not self.include_knowledge or not self.include_example or \
                                    not self.include_emp_losses else ""

        if len(model_id_suffix) > 0:
            if not self.include_knowledge:
                model_id_suffix += '_knowledge'

            if not self.include_example:
                model_id_suffix += '_examples'

            if not self.include_emp_losses:
                model_id_suffix += '_emp_losses'

        return {
            'hub_model_id': f"{HUB_TEXT_MODEL_ID}{model_id_suffix}",
            'hub_private_repo': HUB_TEXT_PRIVATE_REPO,
            'hub_token': HUB_ACCESS_TOKEN
        }

    def default_save_dir(self):
        """
        :return:
        """
        suffix = f"_wo" if not self.include_knowledge or not self.include_example or \
                           not self.include_emp_losses else ""

        if len(suffix) > 0:
            if not self.include_knowledge:
                suffix += '_knowledge'

            if not self.include_example:
                suffix += '_examples'

            if not self.include_emp_losses:
                suffix += '_emp_losses'

        return f"{DEFAULT_SAVE_DIR_PREFIX}/empathetic_chatbot{suffix}"

    def metric_func(self):
        """
        :return:
        """
        return Metrics(tokenizer=self.CONVERSATION_TOKENIZER.target_tokenizer, task_list=['text_generator', ]).compute

    def trainer_args_train(self, save_dir: str=None, evaluation_strategy: str = "epoch", eval_steps: int = 4,
                           save_steps: int = 4, logging_steps: int = 4, learning_rate: float = 1e-5,
                           save_strategy: str = "epoch", per_device_train_batch_size: int = 1,
                           per_device_eval_batch_size: int = 1, number_of_epochs: int = 2,
                           load_best_model_at_end: bool = True, save_total_limit: int = 2, push_to_hub: bool = True,
                           generation_config_path: str = None):

        return Seq2SeqTrainingArguments(
            generation_config=generation_config_path,
            predict_with_generate=True,
            output_dir=save_dir if save_dir is not None else self.default_save_dir(),
            overwrite_output_dir=True,
            evaluation_strategy=evaluation_strategy,
            eval_steps=eval_steps,
            save_steps=save_steps,
            logging_steps=logging_steps,
            do_train=True,
            do_eval=True,
            learning_rate=learning_rate,
            lr_scheduler_type='constant',
            save_strategy=save_strategy,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_eval_batch_size,
            num_train_epochs=number_of_epochs,
            report_to=['tensorboard'],

            # config for load and save best model
            load_best_model_at_end=load_best_model_at_end,
            save_total_limit=save_total_limit,
            metric_for_best_model='loss',
            greater_is_better=False,

            # hub configs
            push_to_hub=push_to_hub,
            hub_model_id=self.hub_args()['hub_model_id'],
            hub_private_repo=self.hub_args()['hub_private_repo'],
            hub_token=self.hub_args()['hub_token'],
            hub_strategy='end',
            # hub_strategy='checkpoint',
            # resume_from_checkpoint='last-checkpoint',
            save_safetensors=False,
        )

    def trainer_args_evaluate(self, save_dir: str = None,
                              logging_steps: int = 4,
                              per_device_eval_batch_size: int = 1,
                              push_to_hub: bool = True,
                              generation_config_path: str = None):

        return Seq2SeqTrainingArguments(
            generation_config=generation_config_path,
            predict_with_generate=True,
            output_dir=save_dir if save_dir is not None else self.default_save_dir(),
            overwrite_output_dir=True,
            logging_steps=logging_steps,
            do_eval=True,
            per_device_eval_batch_size=per_device_eval_batch_size,

            # hub configs
            push_to_hub=push_to_hub,
            hub_model_id=self.hub_args()['hub_model_id'],
            hub_private_repo=self.hub_args()['hub_private_repo'],
            hub_token=self.hub_args()['hub_token'],
            hub_strategy='end',
            # hub_strategy='checkpoint',
            # resume_from_checkpoint='last-checkpoint',
            save_safetensors=False,
        )

    def post_result(self, pred: PredictionOutput):
        result = self.post_process.compute(pred)
        result.update({'metric': pred.metrics})
        return result

    def initial_weights_from_other_models(self):
        weight_states = dict()
        mapping = EmotionalTextualResponseGeneratorMapping()
        key_of_layers = ['encoder', 'knowledge_encoder', 'example_encoders', 'norm_layer', 'decoder']


        try:
            model = mapping.ModelClass.from_pretrained(mapping.hub_args()['hub_model_id'],
                                                       token=mapping.hub_args()['hub_token'])
        except Exception as e1:
            print(f'exception from load model: {mapping.ModelClass.__name__}')
            print(e1)
            try:

                model = mapping.ModelClass.from_pretrained(mapping.default_save_dir())
            except Exception as e2:
                print(f'exception from load model: {mapping.ModelClass.__name__}')
                print(e2)
                raise Exception("can't load pretrained models")

        if model is None:
            return {}

        weight_states.update({k: model.encoder_decoder.state_dict()[k]
                              for k in model.encoder_decoder.state_dict()
                              for key_layer in key_of_layers if k.startswith(f"{key_layer}.")})

        return weight_states


class EmotionalTextualResponseGeneratorMapping:

    def __init__(self, source_max_len: int = 300, target_max_len: int = 100,
                 include_knowledge: bool = True, include_example: bool = True, include_emp_losses: bool = True,
                 *args, **kwargs):
        """

        :param source_max_len:
        :param target_max_len:
        """
        self.DatasetClass = EmpatheticDialoguesDataset
        self.ModelClass = EmotionRoberta2DialoGPT
        self.TrainerClass = MultiTaskTrainer

        self.include_knowledge = include_knowledge
        self.include_example = include_example
        self.include_emp_losses = include_emp_losses

        tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
        self.CONVERSATION_TOKENIZER = ConversationTokenizer(
            source_tokenizer=RobertaTokenizer.from_pretrained("roberta-base"),
            target_tokenizer=tokenizer,
            source_max_len=source_max_len,
            label_max_len=target_max_len,
            source_new_special_tokens={'pad_token': '<pad>'},
            target_new_special_tokens={'pad_token': tokenizer.unk_token},
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
            TextCleaner(texts_key_name=['history', 'label', 'examples'], erase_punc=False),
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
            PreProcessEncoderDecoderInputDictVersion(tokenizer=self.CONVERSATION_TOKENIZER.target_tokenizer,
                                                     gen_label_key_name='labels'),
            ToLong(wanted_list=['input_ids', 'attention_mask', 'token_type_ids', 'labels', 'emotion_labels'] +
                               [f"{rel_name}_{suffix}" for rel_name in ['react_rel', 'social_rel', 'event_rel',
                                                                        'entity_rel']
                                for suffix in ['input_ids', 'attention_mask', 'token_type_ids']] +
                               [f"example_{i}_{suffix}" for i in range(0, 5)
                                for suffix in ['input_ids', 'attention_mask', 'token_type_ids']]),
        ])
        self.post_process = PostProcessResult(tokenizer=self.CONVERSATION_TOKENIZER.target_tokenizer,
                                              task_list=['text_generator', 'classifier'])

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
        config = EmotionRoberta2DialoGPTConfig(
            embedding_tokens_len=len(self.CONVERSATION_TOKENIZER.source_tokenizer),
            decoder_vocab_size=len(self.CONVERSATION_TOKENIZER.target_tokenizer),
            special_token_dict={each: self.CONVERSATION_TOKENIZER.target_tokenizer(each, add_special_tokens=False)['input_ids'][0]
                                for each in self.CONVERSATION_TOKENIZER.target_tokenizer.all_special_tokens},
            bos_token_id=self.CONVERSATION_TOKENIZER.target_tokenizer.bos_token_id,
            eos_token_id=self.CONVERSATION_TOKENIZER.target_tokenizer.eos_token_id,
            pad_token_id=self.CONVERSATION_TOKENIZER.target_tokenizer.pad_token_id,
            num_labels=32,
            kwn_embedding_tokens_len=len(self.KNOWLEDGE_TOKENIZER.tokenizer),
            div_loss_weight=div_loss_weight,
            main_loss_weight=main_loss_weight,
            empathy_loss_weight=empathy_loss_weight,
            include_knowledge=self.include_knowledge,
            include_example=self.include_example,
            include_emp_losses=self.include_emp_losses)
        return {'config': config}

    def hub_args(self):
        """
        :return:
        """
        model_id_suffix = f"_wo" if not self.include_knowledge or not self.include_example or \
                                    not self.include_emp_losses else ""

        if len(model_id_suffix) > 0:
            if not self.include_knowledge:
                model_id_suffix += '_knowledge'

            if not self.include_example:
                model_id_suffix += '_examples'

            if not self.include_emp_losses:
                model_id_suffix += '_emp_losses'

        return {
            'hub_model_id': f"{HUB_EMO_TEXT_MODEL_ID}{model_id_suffix}",
            'hub_private_repo': HUB_EMO_TEXT_PRIVATE_REPO,
            'hub_token': HUB_ACCESS_TOKEN
        }

    def default_save_dir(self):
        """
        :return:
        """
        suffix = f"_wo" if not self.include_knowledge or not self.include_example or \
                           not self.include_emp_losses else ""

        if len(suffix) > 0:
            if not self.include_knowledge:
                suffix += '_knowledge'

            if not self.include_example:
                suffix += '_examples'

            if not self.include_emp_losses:
                suffix += '_emp_losses'

        return f"{DEFAULT_SAVE_DIR_PREFIX}/emotional_empathetic_chatbot{suffix}"

    def metric_func(self):
        """
        :return:
        """
        return Metrics(tokenizer=self.CONVERSATION_TOKENIZER.target_tokenizer,
                       task_list=['text_generator', 'classifier']).compute

    def trainer_args_train(self, save_dir: str=None, evaluation_strategy: str = "epoch", eval_steps: int = 4,
                           save_steps: int = 4, logging_steps: int = 4, learning_rate: float = 1e-5,
                           save_strategy: str = "epoch", per_device_train_batch_size: int = 1,
                           per_device_eval_batch_size: int = 1, number_of_epochs: int = 2,
                           load_best_model_at_end: bool = True, save_total_limit: int = 2, push_to_hub: bool = True,
                           generation_config_path: str = None):

        return Seq2SeqTrainingArguments(
            generation_config=generation_config_path,
            predict_with_generate=True,
            output_dir=save_dir if save_dir is not None else self.default_save_dir(),
            overwrite_output_dir=True,
            evaluation_strategy=evaluation_strategy,
            eval_steps=eval_steps,
            save_steps=save_steps,
            logging_steps=logging_steps,
            do_train=True,
            do_eval=True,
            learning_rate=learning_rate,
            lr_scheduler_type='constant',
            save_strategy=save_strategy,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_eval_batch_size,
            num_train_epochs=number_of_epochs,
            report_to=['tensorboard'],

            # config for load and save best model
            load_best_model_at_end=load_best_model_at_end,
            save_total_limit=save_total_limit,
            metric_for_best_model='loss',
            greater_is_better=False,

            # hub configs
            push_to_hub=push_to_hub,
            hub_model_id=self.hub_args()['hub_model_id'],
            hub_private_repo=self.hub_args()['hub_private_repo'],
            hub_token=self.hub_args()['hub_token'],
            hub_strategy='end',
            # hub_strategy='checkpoint',
            # resume_from_checkpoint='last-checkpoint',
            save_safetensors=False,
        )

    def trainer_args_evaluate(self, save_dir: str = None,
                              logging_steps: int = 4,
                              per_device_eval_batch_size: int = 1,
                              push_to_hub: bool = True, generation_config_path: str = None):

        return Seq2SeqTrainingArguments(
            generation_config=generation_config_path,
            predict_with_generate=True,
            output_dir=save_dir if save_dir is not None else self.default_save_dir(),
            overwrite_output_dir=True,
            logging_steps=logging_steps,
            do_eval=True,
            per_device_eval_batch_size=per_device_eval_batch_size,

            # hub configs
            push_to_hub=push_to_hub,
            hub_model_id=self.hub_args()['hub_model_id'],
            hub_private_repo=self.hub_args()['hub_private_repo'],
            hub_token=self.hub_args()['hub_token'],
            hub_strategy='end',
            # hub_strategy='checkpoint',
            # resume_from_checkpoint='last-checkpoint',
            save_safetensors=False,
        )

    def post_result(self, pred: PredictionOutput):
        result = self.post_process.compute(pred)
        result.update({'metric': pred.metrics})
        return result


class MultiModelEmotionClassifierMapping:

    def __init__(self, source_max_len: int = 300, *args, **kwargs):

        self.DatasetClass = MELDDataset
        self.ModelClass = MultiModelEmotionClassifier
        self.TrainerClass = Trainer

        self.CONVERSATION_TOKENIZER = ConversationTokenizer(
            source_tokenizer=RobertaTokenizer.from_pretrained("roberta-base"),
            target_tokenizer=None,
            source_max_len=source_max_len,
            source_new_special_tokens={'additional_special_tokens': [
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
            TextCleaner(texts_key_name=['history']),
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

        self.post_process = PostProcessResult(tokenizer=self.CONVERSATION_TOKENIZER.target_tokenizer,
                                              task_list=['classifier', ])

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
        config = MultiModelEmotionClassifierConfig(text_audio_emo_num_classes=7,
                                                   embedding_tokens_len=len(self.CONVERSATION_TOKENIZER.source_tokenizer))
        return {'config': config}

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
        return Metrics(tokenizer=None, task_list=['classifier', ]).compute

    def trainer_args_train(self, save_dir: str = None, evaluation_strategy: str = "epoch", eval_steps: int = 4,
                           save_steps: int = 4, logging_steps: int = 4, learning_rate: float = 1e-5,
                           save_strategy: str = "epoch", per_device_train_batch_size: int = 1,
                           per_device_eval_batch_size: int = 1, number_of_epochs: int = 2,
                           load_best_model_at_end: bool = True, save_total_limit: int = 2, push_to_hub: bool = True):
        return TrainingArguments(
            output_dir=save_dir if save_dir is not None else self.default_save_dir(),
            overwrite_output_dir=True,
            evaluation_strategy=evaluation_strategy,
            eval_steps=eval_steps,
            save_steps=save_steps,
            logging_steps=logging_steps,
            do_train=True,
            do_eval=True,
            learning_rate=learning_rate,
            lr_scheduler_type='constant',
            save_strategy=save_strategy,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_eval_batch_size,
            num_train_epochs=number_of_epochs,
            report_to=['tensorboard'],

            # config for load and save best model
            load_best_model_at_end=load_best_model_at_end,
            save_total_limit=save_total_limit,
            metric_for_best_model='loss',
            greater_is_better=False,

            # hub configs
            push_to_hub=push_to_hub,
            hub_model_id=self.hub_args()['hub_model_id'],
            hub_private_repo=self.hub_args()['hub_private_repo'],
            hub_token=self.hub_args()['hub_token'],
            hub_strategy='end',
            # hub_strategy='checkpoint',
            # resume_from_checkpoint='last-checkpoint',
            save_safetensors=False,
        )

    def trainer_args_evaluate(self, save_dir: str = None,
                              logging_steps: int = 4,
                              per_device_eval_batch_size: int = 1,
                              push_to_hub: bool = True):

        return TrainingArguments(
            output_dir=save_dir if save_dir is not None else self.default_save_dir(),
            overwrite_output_dir=True,
            logging_steps=logging_steps,
            do_eval=True,
            per_device_eval_batch_size=per_device_eval_batch_size,

            # hub configs
            push_to_hub=push_to_hub,
            hub_model_id=self.hub_args()['hub_model_id'],
            hub_private_repo=self.hub_args()['hub_private_repo'],
            hub_token=self.hub_args()['hub_token'],
            hub_strategy='end',
            # hub_strategy='checkpoint',
            # resume_from_checkpoint='last-checkpoint',
            save_safetensors=False,
        )

    def post_result(self, pred: PredictionOutput):
        result = self.post_process.compute(pred)
        result.update({'metric': pred.metrics})
        return result
