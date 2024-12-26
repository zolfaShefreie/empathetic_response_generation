import json

from utils.interface import BaseInterface, str2bool
from model_data_process.data_model_mapping import MultiModalResponseGeneratorMapping, TextualResponseGeneratorMapping,\
    EmotionalTextualResponseGeneratorMapping, MultiModelEmotionClassifierMapping
from utils.callbacks import SaveHistoryCallback

from transformers.trainer_utils import PredictionOutput
import os


class EvaluateInterface(BaseInterface):
    DESCRIPTION = "You can generate a text using response generator models.\n" \
                  "The example of format of conversation is saved in conversation_example.json file.\n" \
                  "Consider:\n1. the number of utterance must be even\n" \
                  "2.the last utterance must have a path for audio file for bi-modal models\n" \
                  "3. utterance must be in order of turn" \


    MAP_CONFIGS = {
        'BiModalResponseGenerator': MultiModalResponseGeneratorMapping,
        'EmotionalTextualResponseGenerator': EmotionalTextualResponseGeneratorMapping,
        'TextualResponseGenerator': TextualResponseGeneratorMapping
    }

    ARGUMENTS = {

        'model': {
            'help': 'which model do you want to chat?',
            'choices': MAP_CONFIGS.keys(),
            'required': True
        },

        'include_knowledge': {
            'help': 'is encoded context combined to encoded knowledge?',
            'type': str2bool,
            'required': False,
            'default': True
        },

        'include_example': {
            'help': 'is encoded context combined to encoded examples?',
            'type': str2bool,
            'required': False,
            'default': True
        },

        'include_emp_losses': {
            'help': 'model include empathy losses',
            'type': str2bool,
            'required': False,
            'default': True
        },

        'generation_config_path': {
            'help': 'path of generation_config (json file)',
            'required': False,
            'default': None
        },

        'conversation_path': {
            'help': 'json file that contains one conversation data',
            'required': True,
            'default': None
        }

    }

    def validate_generation_config_path(self, value):
        if value is not None and not os.path.exists(value):
            return None
        return value

    def validate_conversation_path(self, value):
        # to do check it has correct format
        if value is not None and not os.path.exists(value):
            return None
        return value

    def get_conversation(self):

        def validate_conversation(conversation: list):
            if len(conversation) % 2 != 1:
                raise Exception("invalid conversation format")
            for utter in conversation:
                if "utterance" not in utter.keys():
                    raise Exception("invalid conversation format")
                if "audio_file_path" not in utter.keys() and self.model == "BiModalResponseGenerator":
                    raise Exception("invalid conversation format")

            if self.model == "BiModalResponseGenerator":
                if conversation[-1]['audio_file_path'] is None or not os.path.exists(conversation[-1]['audio_file_path']):
                    raise Exception("invalid conversation format")

            return conversation

        def new_format_of_conversation(conversation):
            new_conv = dict()
            history = [each['utterance'] for each in conversation]
            new_conv['history'] = history
            if self.model == "BiModalResponseGenerator":
                new_conv['file_path'] = conversation[-1]['audio_file_path']
            return new_conv

        with open(self.conversation_path, mode='r', encoding='utf-8') as file:
            content = dict(json.loads(file.read()))
            if 'conversation' not in content.keys():
                raise Exception("invalid conversation format")
            return new_format_of_conversation(validate_conversation(content['conversation']))

    def get_generated_knowledge(self):
        pass

    def get_examples(self):
        pass

    def _run_main_process(self):
        config = self.MAP_CONFIGS[self.model](include_knowledge=self.include_knowledge,
                                              include_example=self.include_example,
                                              include_emp_losses=self.include_emp_losses)


if __name__ == "__main__":
    EvaluateInterface().run()
