from model_data_process.dataset import EmpatheticDialoguesDataset
from model_data_process.example_retriever import ExampleRetriever
from model_data_process.knowledge_generator import KnowledgeGenerator
from utils.audio_util import AudioModule
from utils.interface import BaseInterface, str2bool
from model_data_process.data_model_mapping import MultiModalResponseGeneratorMapping, TextualResponseGeneratorMapping,\
    EmotionalTextualResponseGeneratorMapping
from utils.preprocessing import AddBatchDimension

import argparse
import os
import json
import pandas as pd


class ResponseGeneratorInterface(BaseInterface):
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
        if value is None or not os.path.exists(value):
            raise argparse.ArgumentTypeError(f"{value} doesn't exist")
        return value

    def get_conversation(self):

        def validate_conversation(conversation: list):
            if len(conversation) % 2 != 1:
                raise argparse.ArgumentTypeError("invalid conversation format")
            for utter in conversation:
                if "utterance" not in utter.keys():
                    raise argparse.ArgumentTypeError("invalid conversation format")
                if "audio_file_path" not in utter.keys() and self.model == "BiModalResponseGenerator":
                    raise argparse.ArgumentTypeError("invalid conversation format")

            if self.model == "BiModalResponseGenerator":
                if conversation[-1]['audio_file_path'] is None or not os.path.exists(conversation[-1]['audio_file_path']):
                    raise argparse.ArgumentTypeError("invalid conversation format")

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
                raise argparse.ArgumentTypeError("invalid conversation format")
            return new_format_of_conversation(validate_conversation(content['conversation']))

    def get_generated_knowledge(self, record):
        """
        generate knowledge for conversation
        :param record:
        :return:
        """
        social, event, entity = KnowledgeGenerator.run(texts=record['history'])
        record_plus_knw = {'social_rel': social,
                           'event_rel': event,
                           'entity_rel': entity}
        record_plus_knw.update(record)
        return record_plus_knw

    def get_examples(self, record):
        """
        get nearest examples for on conversation
        :param record: a conversation
        :return:
        """
        train_df = EmpatheticDialoguesDataset.conv_preprocess(split='train', add_knowledge=True, add_examples=False)
        train_df = pd.DataFrame(train_df)
        train_df['xReact'] = train_df['social_rel'].apply(lambda x: list(x.values())[0]['xReact'])
        train_df['history_str'] = train_df['history'].apply(lambda x: ", ".join(x))
        example_retriever = ExampleRetriever(train_df=train_df, ctx_key_name='label', qs_key_name='history_str',
                                             conv_key_name='original_conv_id')

        record['history_str'] = ", ".join(record['history'])
        record['xReact'] = list(record['social_rel'].values())[0]['xReact']
        new_record = {**record, **{'original_conv_id': '-2'}}
        record = example_retriever(new_record)
        return record

    def process_audio(self, record):
        record['audio'] = AudioModule.get_audio_data(file_path=record['file_path'])
        return record

    def get_generation_config(self):
        if self.generation_config_path is None:
            return {}

        with open(self.generation_config_path, mode='r', encoding='utf-8') as file:
            generation_config = dict(json.loads(file.read()))
            return generation_config

    def _run_main_process(self):

        # conversation process
        conversation_record = self.get_conversation()
        conversation_record = self.get_examples(self.get_generated_knowledge(conversation_record))
        if self.model == "BiModalResponseGenerator":
            conversation_record = self.process_audio(conversation_record)

        # load model and its mapping
        config = self.MAP_CONFIGS[self.model](include_knowledge=self.include_knowledge,
                                              include_example=self.include_example,
                                              include_emp_losses=self.include_emp_losses)

        model_class = config.ModelClass
        try:
            model = model_class.from_pretrained(config.hub_args()['hub_model_id'], token=config.hub_args()['hub_token'])
        except Exception:
            raise Exception("there is no pretrained model on HuggingFace")
        model.eval()

        # preprocess conversation
        preprocessed_conv = config.TRANSFORMS(conversation_record)
        preprocessed_conv = AddBatchDimension()(preprocessed_conv)

        # generate response
        generation_config = self.get_generation_config()
        output = model.generate(**preprocessed_conv, **generation_config)

        # post process and print
        response = config.CONVERSATION_TOKENIZER.target_tokenizer.decode(output[0], skip_special_tokens=True)
        print(f"Generated Response: {response}")


if __name__ == "__main__":
    ResponseGeneratorInterface().run()
