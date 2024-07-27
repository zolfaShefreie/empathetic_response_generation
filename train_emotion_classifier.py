from utils.interface import BaseInterface
from model_data_process.dataset import MELDDataset
from utils.preprocessing import Pipeline, ConversationFormatter, ConversationTokenizer, TextCleaner, ToTensor, \
    ToNumpy, ToLong, FilterSample, AudioFeatureExtractor
from model_data_process import models
from settings import DEFAULT_SAVE_DIR_PREFIX, HUB_CLASSIFIER_MODEL_ID, HUB_ACCESS_TOKEN, HUB_CLASSIFIER_PRIVATE_REPO, \
    MELD_DATASET_PATH
from utils.callbacks import SaveHistoryCallback
from utils.metrics import Metrics

from transformers import RobertaTokenizer, DefaultFlowCallback, TrainingArguments, Trainer, \
    EarlyStoppingCallback, trainer_utils, AutoFeatureExtractor
import argparse


class TrainInterface(BaseInterface):
    DESCRIPTION = "You can run the train process using this interface"

    ARGUMENTS = {
        'number_of_epochs': {
            'help': 'number of training epoch. it must be positive integer',
            'type': int,
            'required': True
        },
        'save_dir': {
            'help': 'diractory of model',
            'required': False,
            'default': None
        },
        'evaluation_strategy': {
            'help': ' The evaluation strategy to adopt during training.',
            'choices': ['steps', 'epoch', 'no'],
            'required': False,
            'default': 'epoch'
        },
        'eval_steps': {
            'help': 'Number of update steps between two evaluations if evaluation_strategy="steps"',
            'type': int,
            'required': False,
            'default': 4
        },

        'logging_steps': {
            'help': ' Number of update steps between two logs if logging_strategy="steps". '
                    'Should be an integer',
            'type': int,
            'required': False,
            'default': 4
        },

        'save_strategy': {
            'help': 'The checkpoint save strategy to adopt during training.',
            'choices': ['steps', 'epoch', 'no'],
            'required': False,
            'default': 'epoch'
        },

        'load_best_model_at_end': {
            'help': 'Whether or not to load the best model found during training at the end of training.',
            'type': bool,
            'required': False,
            'default': True
        },

        'save_total_limit': {
            'help': 'If a value is passed, will limit the total amount of checkpoints. Deletes the older checkpoints '
                    'in output_dir. When load_best_model_at_end is enabled, the “best” checkpoint according to '
                    'metric_for_best_model will always be retained in addition to the most recent ones. For example, '
                    'for save_total_limit=5 and load_best_model_at_end, the four last checkpoints will always be '
                    'retained alongside the best model. When save_total_limit=1 and load_best_model_at_end, it is '
                    'possible that two checkpoints are saved: the last one and the best one (if they are different).',
            'type': int,
            'required': False,
            'default': 2
        },

        'save_steps': {
            'help': 'Number of updates steps before two checkpoint saves if save_strategy="steps"',
            'type': int,
            'required': False,
            'default': 4
        },

        'per_device_train_batch_size': {
            'help': 'The batch size per GPU/XPU/TPU/MPS/NPU core/CPU for training.',
            'type': int,
            'required': False,
            'default': 1
        },

        'per_device_eval_batch_size': {
            'help': 'The batch size per GPU/XPU/TPU/MPS/NPU core/CPU for evaluation.',
            'type': int,
            'required': False,
            'default': 1
        },

        'learning_rate': {
            'help': 'The initial learning rate for AdamW optimizer.',
            'type': float,
            'required': False,
            'default': 1e-5
        },

        'push_to_hub': {
            'help': 'Whether or not to push the model to the Hub every time the model is saved. if it is true please'
                    'fill information on .env file',
            'type': bool,
            'required': False,
            'default': True
        },
    }

    CONVERSATION_TOKENIZER = ConversationTokenizer(tokenizer=RobertaTokenizer.from_pretrained("roberta-base"),
                                                   max_len=300,
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

    TRANSFORMS = Pipeline(functions=[
        TextCleaner(texts_key_name='history'),
        ConversationFormatter(history_key_name='history',
                              last_utter_key_name='last_utter'),
        ToNumpy(unwanted_keys=['audio']),
        CONVERSATION_TOKENIZER,
        AudioFeatureExtractor(feature_extractor=AutoFeatureExtractor.from_pretrained("facebook/data2vec-audio-base-960h"),
                              audio_key_name='audio',
                              result_prefix_key_name='audio'),
        FilterSample(wanted_keys=['input_ids', 'attention_mask', 'token_type_ids', 'audio_input_values',
                                  'audio_attention_mask', 'labels', ]),
        ToTensor(),
        ToLong(),
    ])

    def validate_number_of_epochs(self, value):
        if value <= 0:
            raise argparse.ArgumentTypeError(f"{value} is an invalid positive int value")
        return value

    def get_training_args(self):
        """
        set trainer based on arguments of model
        :return:
        """
        return TrainingArguments(
            output_dir=self.save_dir if self.save_dir is not None else f"{DEFAULT_SAVE_DIR_PREFIX}/emotion_recognition",
            overwrite_output_dir=True,
            evaluation_strategy=self.evaluation_strategy,
            eval_steps=self.eval_steps,
            save_steps=self.save_steps,
            logging_steps=self.logging_steps,
            do_train=True,
            do_eval=True,
            learning_rate=self.learning_rate,
            lr_scheduler_type='constant',
            save_strategy=self.save_strategy,
            per_device_train_batch_size=self.per_device_train_batch_size,
            per_device_eval_batch_size=self.per_device_eval_batch_size,
            num_train_epochs=self.number_of_epochs,

            # config for load and save best model
            load_best_model_at_end=self.load_best_model_at_end,
            save_total_limit=self.save_total_limit,
            metric_for_best_model='loss',
            greater_is_better=False,

            # hub configs
            push_to_hub=self.push_to_hub,
            hub_model_id=HUB_CLASSIFIER_MODEL_ID,
            hub_strategy='checkpoint',
            hub_private_repo=HUB_CLASSIFIER_PRIVATE_REPO,
            resume_from_checkpoint='last-checkpoint',
            hub_token=HUB_ACCESS_TOKEN,
            save_safetensors=False,
        )

    def _run_main_process(self):

        train_dataset = MELDDataset(split='train', transform=self.TRANSFORMS, dataset_path=MELD_DATASET_PATH)
        val_dataset = MELDDataset(split='validation', transform=self.TRANSFORMS, dataset_path=MELD_DATASET_PATH)

        model_class = models.MultiModelEmotionClassifier
        try:
            model = model_class.from_pretrained(HUB_CLASSIFIER_MODEL_ID, token=HUB_ACCESS_TOKEN)
        except Exception as e:
            model = model_class(num_classes=7, embedding_tokens_len=len(self.CONVERSATION_TOKENIZER.tokenizer))

        trainer = Trainer(
            model=model,
            args=self.get_training_args(),
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=Metrics(tokenizer=self.CONVERSATION_TOKENIZER.tokenizer,
                                    task_list=['classifier']).compute,
            callbacks=[SaveHistoryCallback(),
                       DefaultFlowCallback(),
                       EarlyStoppingCallback(early_stopping_patience=2)]
        )

        trainer.train(resume_from_checkpoint=None if trainer_utils.get_last_checkpoint(trainer.args.output_dir) is None else True)
        trainer.save_model()


if __name__ == "__main__":
    TrainInterface().run()
