from utils.interface import BaseInterface
from model_data_process.data_model_mapping import MultiModalResponseGeneratorMapping, TextualResponseGeneratorMapping,\
    EmotionalTextualResponseGeneratorMapping, MultiModelEmotionClassifierMapping
from utils.callbacks import SaveHistoryCallback

from transformers import DefaultFlowCallback, EarlyStoppingCallback
from transformers.trainer_utils import PredictionOutput
import os


class EvaluateInterface(BaseInterface):
    DESCRIPTION = "You can run the evaluation process using this interface"

    MAP_CONFIGS = {
        'BiModalResponseGenerator': MultiModalResponseGeneratorMapping,
        'BiModalEmotionClassifier': MultiModelEmotionClassifierMapping,
        'EmotionalTextualResponseGenerator': EmotionalTextualResponseGeneratorMapping,
        'TextualResponseGenerator': TextualResponseGeneratorMapping
    }

    ARGUMENTS = {

        'model': {
            'help': 'test which model?',
            'choices': MAP_CONFIGS.keys(),
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

        'load_best_model_at_end': {
            'help': 'Whether or not to load the best model found during training at the end of training.',
            'type': bool,
            'required': False,
            'default': True
        },

        'per_device_eval_batch_size': {
            'help': 'The batch size per GPU/XPU/TPU/MPS/NPU core/CPU for evaluation.',
            'type': int,
            'required': False,
            'default': 1
        },
    }

    def save_result(self, prediction_output: PredictionOutput, default_path: str):
        """
        save result of predict model
        :param prediction_output:
        :param tokenizer:
        :param default_path:
        :return:
        """
        data = {
            'predictions': prediction_output.predictions,
            'labels': prediction_output.label_ids,
            'metrics': prediction_output.metrics
        }
        output_file_path = f"{self.save_dir if self.save_dir is not None else default_path}/result"
        
        if not os.path.exists(os.path.dirname(output_file_path)):
            try:
                os.makedirs(os.path.dirname(output_file_path))
            except OSError as exc:
                print(exc)
                pass

        with open(output_file_path, mode='w', encoding='utf-8') as file:
            file.write(str(data))

    def _run_main_process(self):
        config = self.MAP_CONFIGS[self.model]()

        test_dataset = config.DatasetClass(**config.dataset_args(split='test'))

        model_class = config.ModelClass
        model = model_class.from_pretrained(config.hub_args()['hub_model_id'], token=config.hub_args()['hub_token'])

        model.eval()

        trainer_args = config.trainer_args_evaluate(save_dir=self.save_dir,
                                                    evaluation_strategy=self.evaluation_strategy,
                                                    logging_steps=self.logging_steps,
                                                    per_device_eval_batch_size=self.per_device_eval_batch_size,
                                                    load_best_model_at_end=self.load_best_model_at_end,
                                                    push_to_hub=self.push_to_hub)
        trainer = config.TrainerClass(
            model=model,
            args=trainer_args,
            compute_metrics=config.metric_func(),
            callbacks=[SaveHistoryCallback(),
                       DefaultFlowCallback(),
                       EarlyStoppingCallback(early_stopping_patience=2)]
        )
        predictions = trainer.predict(test_dataset=test_dataset)
        self.save_result(prediction_output=predictions, default_path=config.default_save_dir())


if __name__ == "__main__":
    EvaluateInterface().run()
