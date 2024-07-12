from enum import Enum
from datasets import load_dataset
import torch
import os
import ast
import pandas as pd

from model_data_process.example_retriever import ExampleRetriever
from utils.preprocessing import NewVersionDialogues
from settings import DATASET_CACHE_PATH
from model_data_process.knowledge_generator import KnowledgeGenerator


class EmpatheticDialoguesDataset(torch.utils.data.Dataset):

    class EmotionType(Enum):
        sentimental = 0
        afraid = 1
        proud = 2
        faithful = 3
        terrified = 4
        joyful = 5
        angry = 6
        sad = 7
        jealous = 8
        grateful = 9
        prepared = 10
        embarrassed = 11
        excited = 12
        annoyed = 13
        lonely = 14
        ashamed = 15
        guilty = 16
        surprised = 17
        nostalgic = 18
        confident = 19
        furious = 20
        disappointed = 21
        caring = 22
        trusting = 23
        disgusted = 24
        anticipating = 25
        anxious = 26
        hopeful = 27
        content = 28
        impressed = 29
        apprehensive = 30
        devastated = 31

        @classmethod
        def get_emotion_name(cls, code: int):
            return {each.value: each.name for each in cls}[code]

    CACHE_PATH = DATASET_CACHE_PATH
    DATASET_NAME = "empatheticdialogues"

    SOCIAL_REL_KEY_NAME = 'social_rel'
    EVENT_REL_KEY_NAME = 'event_rel'
    ENTITY_REL_KEY_NAME = 'entity_rel'

    def __init__(self, split='train', transform=None):
        """
        initial of dataset
        :param split: train/test/validation
        :param transform:
        """
        self.data = self.conv_preprocess(split=split)
        self.transform = transform
        self.n_sample = len(self.data)

    @classmethod
    def conv_preprocess(cls, split: str, add_knowledge: bool = True, add_examples: bool = True) -> list:
        """
        change the format of dataset
        :param add_examples:
        :param split: train/test/validation
        :param add_knowledge: add knowledge to each conversation
        :return: dataset with new format
        """
        file_path = f"{cls.CACHE_PATH}/{cls.DATASET_NAME}_{split}".replace("[", "_").replace(":", "_").replace("]", "_")
        if os.path.exists(file_path):
            # load data from cache
            with open(file_path, mode='r', encoding='utf-8') as file:
                content = file.read()
                return ast.literal_eval(content)

        else:
            # reformat empathetic_dialogues dataset
            raw_dataset = load_dataset("empathetic_dialogues", split=split)
            process_manager = NewVersionDialogues(conv_id_key_name='conv_id',
                                                  turn_key_name='utterance_idx',
                                                  utter_key_name='utterance',
                                                  other_conv_features=['context', 'prompt', 'selfeval', 'tags'],
                                                  other_utter_features=['speaker_idx'],
                                                  new_conv_each_sys_responses=True,
                                                  responses_in_history=True,
                                                  context_key_name='history',
                                                  label_key_name='label')
            data = process_manager.reformat(raw_dataset=raw_dataset)

            if add_knowledge:
                data = cls._add_knowledge_to_conv(dataset=data)

            if add_examples:
                data = cls.add_examples(data=data, split=split)

            # save dataset on cache'_
            if not os.path.exists(os.path.dirname(file_path)):
                try:
                    os.makedirs(os.path.dirname(file_path))
                except OSError as exc:
                    print(exc)
                    pass
            with open(file_path, mode='w', encoding='utf-8') as file:
                file.write(str(data))

            return data

    @classmethod
    def _add_knowledge_to_conv(cls, dataset):
        """
        add knowledge to dataset
        :param dataset:
        :return:
        """
        knw_added_dataset = list()

        for record in dataset:
            social, event, entity = KnowledgeGenerator.run(texts=record['history'])
            record_plus_knw = {cls.SOCIAL_REL_KEY_NAME: social,
                               cls.EVENT_REL_KEY_NAME: event,
                               cls.ENTITY_REL_KEY_NAME: entity}
            record_plus_knw.update(record)
            knw_added_dataset.append(record_plus_knw)

        return knw_added_dataset

    @classmethod
    def add_examples(cls, data: list, split: str) -> list:
        """
        add examples to one record
        :param data:
        :param split:
        :return:
        """
        new_dataset = list()
        train_df = data if 'train' in split else EmpatheticDialoguesDataset.conv_preprocess(split='train[:13]',
                                                                                            add_knowledge=True,
                                                                                            add_examples=False)
        train_df = pd.DataFrame(train_df)
        train_df['xReact'] = train_df['social_rel'].apply(lambda x: list(x.values())[0]['xReact'])
        train_df['history_str'] = train_df['history'].apply(lambda x: ", ".join(x))
        example_retriever = ExampleRetriever(train_df=train_df, ctx_key_name='history_str', qs_key_name='label',
                                             conv_key_name='original_conv_id')

        for record in data:
            record['history_str'] = ", ".join(record['history'])
            record['xReact'] = list(record['social_rel'].values())[0]['xReact']
            record = example_retriever(record)
            new_dataset.append(record)

        return new_dataset

    def __getitem__(self, idx: int):
        """
        get item with specific index using dataset[idx]
        :param idx: index
        :return:
        """
        raw_item_data = self.data[idx]
        history, label, emotion_label = raw_item_data['history'], raw_item_data['label'], raw_item_data['context']
        emotion_label = self.EmotionType[emotion_label].value
        item_data = {'history': history, 'label': label, 'emotion_labels': emotion_label}
        if self.SOCIAL_REL_KEY_NAME in raw_item_data.keys():
            item_data.update({
                self.SOCIAL_REL_KEY_NAME: raw_item_data[self.SOCIAL_REL_KEY_NAME],
                self.EVENT_REL_KEY_NAME: raw_item_data[self.EVENT_REL_KEY_NAME],
                self.ENTITY_REL_KEY_NAME: raw_item_data[self.ENTITY_REL_KEY_NAME]
            })
        if ExampleRetriever.EXAMPLE_KEY_NAME in raw_item_data.keys():
            item_data.update({
                ExampleRetriever.EXAMPLE_KEY_NAME: raw_item_data[ExampleRetriever.EXAMPLE_KEY_NAME]
            })
        if self.transform:
            return self.transform(item_data)
        return item_data

    def __len__(self) -> int:
        """
        get length of dataset when using len(dataset)
        :return:
        """
        return self.n_sample
