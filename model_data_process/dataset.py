from datasets import load_dataset
import torch
import os
import ast

from utils.preprocessing import NewVersionDialogues
from settings import DATASET_CACHE_PATH


class EmpatheticDialoguesDataset(torch.utils.data.Dataset):

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
        history, label = self.data[idx]['history'], self.data[idx]['label']
        if self.transform:
            return self.transform((history, label))
        return history, label

    def __len__(self) -> int:
        """
        get length of dataset when using len(dataset)
        :return:
        """
        return self.n_sample
