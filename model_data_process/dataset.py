from enum import Enum
from datasets import load_dataset
import torch
import os
import ast

from utils.preprocessing import NewVersionDialogues
from settings import DATASET_CACHE_PATH


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

    def __init__(self, split='train', transform=None):
        """
        initial of dataset
        :param split: train/test/validation
        :param transform:
        """
        self.data = self._conv_preprocess(split=split)
        self.transform = transform
        self.n_sample = len(self.data)

    @classmethod
    def _conv_preprocess(cls, split: str) -> list:
        """
        change the format of dataset
        :param split: train/test/validation
        :return: dataset with new format
        """
        if os.path.exists(f"{cls.CACHE_PATH}/{cls.DATASET_NAME}_{split}"):
            # load data from cache
            with open(f"{cls.CACHE_PATH}/{cls.DATASET_NAME}_{split}", mode='r', encoding='utf-8') as file:
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

            # save dataset on cache'_
            file_path = f"{cls.CACHE_PATH}/{cls.DATASET_NAME}_{split}".replace("[", "_").replace(":", "_").replace("]", "_")
            if not os.path.exists(os.path.dirname(file_path)):
                try:
                    os.makedirs(os.path.dirname(file_path))
                except OSError as exc:
                    print(exc)
                    pass
            with open(file_path, mode='w', encoding='utf-8') as file:
                file.write(str(data))

            return data

    def __getitem__(self, idx: int):
        """
        get item with specific index using dataset[idx]
        :param idx: index
        :return:
        """
        history, label, emotion_label = self.data[idx]['history'], self.data[idx]['label'], self.data[idx]['context']
        emotion_label = self.EmotionType[emotion_label].value
        if self.transform:
            return self.transform((history, label, emotion_label))
        return history, label, emotion_label

    def __len__(self) -> int:
        """
        get length of dataset when using len(dataset)
        :return:
        """
        return self.n_sample
