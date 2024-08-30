from utils.interface import BaseInterface
from model_data_process.data_model_mapping import MultiModalResponseGeneratorMapping, TextualResponseGeneratorMapping,\
    EmotionalTextualResponseGeneratorMapping, MultiModelEmotionClassifierMapping
from utils.callbacks import SaveHistoryCallback

from transformers import Seq2SeqTrainingArguments, DefaultFlowCallback, EarlyStoppingCallback, trainer_utils, \
    TrainingArguments
import argparse


class TrainInterface(BaseInterface):
    DESCRIPTION = "You can run the train process using this interface"

    MAP_CONFIGS = {
        'BiModalResponseGenerator': MultiModalResponseGeneratorMapping,
        'BiModalEmotionClassifier': MultiModelEmotionClassifierMapping,
        'EmotionalTextualResponseGenerator': EmotionalTextualResponseGeneratorMapping,
        'TextualResponseGenerator': TextualResponseGeneratorMapping
    }

    ARGUMENTS = {
        'model': {
            'help': 'train which model?',
            'choices': MAP_CONFIGS.keys(),
            'required': True
        },

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

    def validate_number_of_epochs(self, value):
        if value <= 0:
            raise argparse.ArgumentTypeError(f"{value} is an invalid positive int value")
        return value

    def _run_main_process(self):
        config = self.MAP_CONFIGS[self.model]()

        train_dataset = config.DatasetClass(**config.dataset_args(split='train'))
        val_dataset = config.DatasetClass(**config.dataset_args(split='validation'))

        model_class = config.ModelClass

        try:
            model = model_class.from_pretrained(config.hub_args()['hub_model_id'], token=config.hub_args()['hub_token'])
        except Exception as e:
            model = model_class(**config.model_args())

        trainer_args = config.trainer_args_train(save_dir=self.save_dir, evaluation_strategy=self.evaluation_strategy,
                                                 eval_steps=self.eval_steps, save_steps=self.save_steps,
                                                 logging_steps=self.logging_steps, learning_rate=self.learning_rate,
                                                 save_strategy=self.save_strategy,
                                                 per_device_train_batch_size=self.per_device_train_batch_size,
                                                 per_device_eval_batch_size=self.per_device_eval_batch_size,
                                                 number_of_epochs=self.number_of_epochs,
                                                 load_best_model_at_end=self.load_best_model_at_end,
                                                 save_total_limit=self.save_total_limit, push_to_hub=self.push_to_hub)

        trainer = config.TrainerClass(
            model=model,
            args=trainer_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=config.metric_func(),
            callbacks=[SaveHistoryCallback(),
                       DefaultFlowCallback(),
                       EarlyStoppingCallback(early_stopping_patience=2)]
        )

        train_is_done = False
        while not train_is_done:
            try:
                trainer.train(resume_from_checkpoint=None if trainer_utils.get_last_checkpoint(
                    trainer.args.output_dir) is None else True)
                trainer.save_model()
                train_is_done = True
            except Exception as e:
                print(e)


if __name__ == "__main__":
    TrainInterface().run()
