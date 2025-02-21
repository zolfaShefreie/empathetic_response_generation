import json
from enum import Enum
from huggingface_hub import snapshot_download
from datasets import load_dataset
import torch
import os
import ast
import pandas as pd

from model_data_process.example_retriever import ExampleRetriever
from utils.preprocessing import NewVersionDialogues
from settings import DATASET_CACHE_PATH, HUB_ACCESS_TOKEN
from model_data_process.knowledge_generator import KnowledgeGenerator
from utils.audio_util import AudioModule
from utils.others import unzip


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
        file_path = f"{cls.CACHE_PATH}/{cls.DATASET_NAME}_{split}.json".replace("[", "_").replace(":", "_").replace("]", "_")
        if os.path.exists(file_path):
            # load data from cache
            data = list()
            with open(file_path, mode='r', encoding='utf-8') as file:
                while True:
                    record = file.readline()
                    if record == '\n' or len(record) == 0:
                        break
                    data.append(json.loads(record))
            return data
            # with open(file_path, mode='r', encoding='utf-8') as file:
            #     content = file.read()
            #     return ast.literal_eval(content)

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
            data = process_manager.two_party_reformat(raw_dataset=raw_dataset)
            print('finish new version', split)

            if add_knowledge:
                data = cls._add_knowledge_to_conv(dataset=data)
                print('finish add knowledge', split)

            if add_examples:
                data = cls.add_examples(data=data, split=split)
                print('finish add examples', split)

            # save dataset on cache'_
            if not os.path.exists(os.path.dirname(file_path)):
                try:
                    os.makedirs(os.path.dirname(file_path))
                except OSError as exc:
                    print(exc)
                    pass
            with open(file_path, mode='w', encoding='utf-8') as file:
                for record in data:
                    json_str = json.dumps(record)
                    file.write(json_str)
                    file.write("\n")
                # file.write(str(data))

            return data

    @classmethod
    def _add_knowledge_to_conv(cls, dataset):
        """
        add knowledge to dataset
        :param dataset:
        :return:
        """
        knw_added_dataset = list()
        count = 0
        length = len(dataset)
        for record in dataset:
            social, event, entity = KnowledgeGenerator.run(texts=record['history'])
            record_plus_knw = {cls.SOCIAL_REL_KEY_NAME: social,
                               cls.EVENT_REL_KEY_NAME: event,
                               cls.ENTITY_REL_KEY_NAME: entity}
            record_plus_knw.update(record)
            knw_added_dataset.append(record_plus_knw)
            print('add knowledge for one record', count, length)
            count += 1
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
        train_df = data if 'train' in split else EmpatheticDialoguesDataset.conv_preprocess(split='train',
                                                                                            add_knowledge=True,
                                                                                            add_examples=False)
        train_df = pd.DataFrame(train_df)
        train_df['xReact'] = train_df['social_rel'].apply(lambda x: list(x.values())[0]['xReact'])
        train_df['history_str'] = train_df['history'].apply(lambda x: ", ".join(x))
        example_retriever = ExampleRetriever(train_df=train_df, ctx_key_name='label', qs_key_name='history_str',
                                             conv_key_name='original_conv_id')

        count = 0
        length = len(data)
        for record in data:
            record['history_str'] = ", ".join(record['history'])
            record['xReact'] = list(record['social_rel'].values())[0]['xReact']
            record = example_retriever(record)
            new_dataset.append(record)
            print('add examples for one record', count, length)
            count += 1

        return new_dataset

    def __getitem__(self, idx: int):
        """
        get item with specific index using dataset[idx]
        :param idx: index
        :return:
        """
        raw_item_data = self.data[idx].copy()
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


class MELDDataset(torch.utils.data.Dataset):
    """
        This class is written based on data on below link
        (https://web.eecs.umich.edu/~mihalcea/downloads/MELD.Raw.tar.gz)
    """

    class EmotionType(Enum):
        surprise = 0
        anger = 1
        neutral = 2
        joy = 3
        sadness = 4
        fear = 5
        disgust = 6

    class SentimentType(Enum):
        positive = 0
        negative = 1
        neutral = 2

    SPLIT_PATHS = {'train': {'metadata': 'train_sent_emo.csv',
                             'folder': 'train/train_splits'},
                   'validation': {'metadata': 'dev_sent_emo.csv',
                                  'folder': 'dev/dev_splits_complete'},
                   'test': {'metadata': 'test_sent_emo.csv',
                            'folder': 'test/output_repeated_splits_test'}}

    FILE_PATH_KEY_NAME = 'file_path'
    AUDIO_DATA_KEY_NAME = 'audio'
    CACHE_PATH = DATASET_CACHE_PATH
    DATASET_NAME = "meld"

    def __init__(self, dataset_path: str, split='train', transform=None):
        """
        initial of dataset
        :param dataset_path: path of unzipped dataset
        :param split: train/test/validation
        :param transform:
        """
        self.data = self.conv_preprocess(split=split, dataset_path=dataset_path)
        self.data = self._audio_file_preprocessing(data=self.data, dataset_path=dataset_path, split=split)
        self.transform = transform
        self.n_sample = len(self.data)

    @classmethod
    def conv_preprocess(cls, split: str, dataset_path: str) -> list:
        """
        change the format of dataset
        :param dataset_path:
        :param split: train/test/validation
        :return: dataset with new format
        """
        file_path = f"{cls.CACHE_PATH}/{cls.DATASET_NAME}_{split}.json"
        if os.path.exists(file_path):
            # load data from cache
            data = list()
            with open(file_path, mode='r', encoding='utf-8') as file:
                while True:
                    record = file.readline()
                    if record == '\n' or len(record) == 0:
                        break
                    data.append(json.loads(record))
            return data
            # with open(file_path, mode='r', encoding='utf-8') as file:
            #     content = file.read()
            #     return ast.literal_eval(str(content))

        else:
            # reformat meld dataset
            raw_dataset = pd.read_csv(f"{dataset_path}/{cls.SPLIT_PATHS[split]['metadata']}")
            raw_dataset = raw_dataset.to_dict('records')
            process_manager = NewVersionDialogues(conv_id_key_name='Dialogue_ID',
                                                  turn_key_name='Utterance_ID',
                                                  utter_key_name='Utterance',
                                                  other_conv_features=['Season', 'Episode', ],
                                                  other_utter_features=['Sentiment', 'Emotion', 'Speaker',
                                                                        'Utterance_ID', 'StartTime', 'EndTime',
                                                                        cls.FILE_PATH_KEY_NAME],
                                                  new_conv_each_sys_responses=True,
                                                  responses_in_history=True,
                                                  context_key_name='history',
                                                  label_key_name='label')
            data = process_manager.multi_party_reformat(raw_dataset=raw_dataset)

            # save dataset on cache'_
            if not os.path.exists(os.path.dirname(file_path)):
                try:
                    os.makedirs(os.path.dirname(file_path))
                except OSError as exc:
                    print(exc)
                    pass
            with open(file_path, mode='w', encoding='utf-8') as file:
                for record in data:
                    json_str = json.dumps(record)
                    file.write(json_str)
                    file.write("\n")

            return data

    @classmethod
    def _audio_file_preprocessing(cls, data: list, dataset_path, split) -> list:
        """
        extract audio and get data of it
        :param data:
        :return:
        """

        def get_path(row):
            """
            get path of video file of meld dataset
            :param row:
            :return:
            """
            return f"{dataset_path}/{cls.SPLIT_PATHS[split]['folder']}/" \
                   f"dia{row['original_conv_id']}_utt{row['Utterance_ID_label']}.mp4"

        processed_data = list()

        for record in data:
            file_path = record[cls.FILE_PATH_KEY_NAME] = get_path(record)

            # if there are some problems with file, the record won't get considered as a record of dataset
            try:
                if not os.path.exists(file_path.replace(".mp4", ".wav")):
                    audio_file_path = AudioModule.extract_audio_from_video(video_path=file_path,
                                                                           saved_path=file_path.replace(".mp4", ".wav"))
                else:
                    audio_file_path = file_path.replace(".mp4", ".wav")

                record[f"{cls.FILE_PATH_KEY_NAME}_label"] = audio_file_path
                record[cls.AUDIO_DATA_KEY_NAME] = AudioModule.get_audio_data(file_path=audio_file_path)
                processed_data.append(record)

            except Exception as e:
                pass

        return processed_data

    def get_audio_data(self, data: list):
        """
        get data of audio
        :param data:
        :return:
        """
        processed_data = list()
        for record in data:
            try:
                record[self.AUDIO_DATA_KEY_NAME] = AudioModule.get_audio_data(file_path=record[f"{self.FILE_PATH_KEY_NAME}_label"])
                processed_data.append(data)
            except Exception as e:
                pass

        return processed_data

    def __getitem__(self, index: int) -> dict:
        """
        get item with specific index using dataset[index]
        :param index:
        :return:
        """
        item_data = self.data[index].copy()
        emotion_label = self.EmotionType[item_data['Emotion_label']].value
        sentiment_label = self.SentimentType[item_data['Sentiment_label']].value
        # print(type(item_data['history']), item_data['history'], item_data['label'])
        history = item_data['history'] + [item_data['label'], ]
        item_data.update({
            'history': history,
            'labels': emotion_label,
            'sentiment_label': sentiment_label,
        })
        if self.transform:
            return self.transform(item_data)
        return item_data

    def __len__(self):
        """
        length of dataset
        :return:
        """
        return self.n_sample


class BiMEmpDialoguesDataset(torch.utils.data.Dataset):

    REPO_ID = "Shefreie/BiMEmpDialogues_zip"
    PREFIX = "./data"
    FILE_PATH_KEY_NAME = 'file_name'
    AUDIO_DATA_KEY_NAME = 'audio'
    DATASET_NAME = "BiMEmpDialogues"
    CACHE_PATH = DATASET_CACHE_PATH

    SOCIAL_REL_KEY_NAME = 'social_rel'
    EVENT_REL_KEY_NAME = 'event_rel'
    ENTITY_REL_KEY_NAME = 'entity_rel'

    def __init__(self, dataset_dir: str = None, split='train', transform=None, include_audio: bool = True,
                 chunk_length: int = 2000):
        if dataset_dir is None:
            dataset_dir = self.get_from_huggingface()
        self.chunk_length = chunk_length
        self.data = self.conv_process_chunk_management(split=split, dataset_dir=dataset_dir)
        self.include_audio = include_audio
        # if include_audio:
        #     self.data = self._audio_file_preprocessing(data=self.data, dataset_path=dataset_dir, split=split)
        self.transform = transform
        self.n_sample = len(self.data)
        self.split = split
        self.dataset_dir = dataset_dir

    def conv_process_chunk_management(self, dataset_dir: str,  split: str, add_knowledge: bool = True,
                                      add_examples: bool = True):

        file_path = f"{self.CACHE_PATH}/{self.DATASET_NAME}_{split}.json"
        if os.path.exists(file_path):
            # load data from cache
            data = list()
            with open(file_path, mode='r', encoding='utf-8') as file:
                while True:
                    record = file.readline()
                    if record == '\n' or len(record) == 0:
                        break
                    data.append(json.loads(record))
            return data
                # content = file.read()
                # print('after read content')
                # return ast.literal_eval(content)

        else:
            conv_num = int(len(pd.read_csv(f"{dataset_dir}/{split}/metadata.csv"))/2)

            for chunk_index in range(int(conv_num / self.chunk_length) + 1):
                self.conv_preprocess(split=split, add_knowledge=add_knowledge, add_examples=add_examples,
                                     chunk_index=chunk_index, dataset_dir=dataset_dir)
                print(f"chunk: {chunk_index} finished")

            data = self.merge_sub_data(split=split, number_chunks=int(conv_num / self.chunk_length) + 1)

            if not os.path.exists(os.path.dirname(file_path)):
                try:
                    os.makedirs(os.path.dirname(file_path))
                except OSError as exc:
                    print(exc)
                    pass
            with open(file_path, mode='w', encoding='utf-8') as file:
                for record in data:
                    json_str = json.dumps(record)
                    file.write(json_str)
                    file.write("\n")
                # file.write(str(data))

            return data

    def merge_sub_data(self, split: str, number_chunks: int) -> list:
        """
        merge all files for chunks
        :param split:
        :param number_chunks:
        :return:
        """
        merged_data = list()
        for i in range(number_chunks):
            file_path = f"{self.CACHE_PATH}/{self.DATASET_NAME}_{split}_{i}"
            if os.path.exists(file_path):
                # load data from cache
                with open(file_path, mode='r', encoding='utf-8') as file:
                    content = file.read()
                    merged_data += ast.literal_eval(content)
        return merged_data

    def conv_preprocess(self, dataset_dir: str, split: str, chunk_index: int, add_knowledge: bool = True,
                        add_examples: bool = True) -> list:
        """
        change the format of dataset
        :param chunk_index:
        :param dataset_dir:
        :param add_examples:
        :param split: train/test/validation
        :param add_knowledge: add knowledge to each conversation
        :return: dataset with new format
        """
        file_path = f"{self.CACHE_PATH}/{self.DATASET_NAME}_{split}_{chunk_index}"
        if os.path.exists(file_path):
            # load data from cache
            with open(file_path, mode='r', encoding='utf-8') as file:
                content = file.read()
                return ast.literal_eval(content)

        else:
            # reformat empathetic_dialogues dataset
            raw_dataset = pd.read_csv(f"{dataset_dir}/{split}/metadata.csv").fillna(-1)
            raw_dataset = raw_dataset.to_dict('records')
            process_manager = NewVersionDialogues(conv_id_key_name='conv_id',
                                                  turn_key_name='utter_id',
                                                  utter_key_name='utterance',
                                                  other_conv_features=[],
                                                  other_utter_features=['speaker', 'file_name', 'Emotion',
                                                                        'Sentiment', 'emotion', 'act', 'scam',
                                                                        'client_talk_type', 'main_therapist_behaviour'],
                                                  new_conv_each_sys_responses=True,
                                                  responses_in_history=True,
                                                  context_key_name='history',
                                                  label_key_name='label')
            data = process_manager.two_party_reformat(raw_dataset=raw_dataset)[chunk_index * self.chunk_length:
                                                                               (chunk_index + 1) * self.chunk_length]
            print('finish new version', split)

            if add_knowledge:
                data = self._add_knowledge_to_conv(dataset=data)
                print('finish add knowledge', split, chunk_index)

            if add_examples:
                data = self.add_examples(data=data, split=split)
                print('finish add examples', split, chunk_index)

            # save dataset on cache'_
            if not os.path.exists(os.path.dirname(file_path)):
                try:
                    os.makedirs(os.path.dirname(file_path))
                except OSError as exc:
                    print(exc)
                    pass
            with open(file_path, mode='w', encoding='utf-8') as file:
                file.write(str(data))

            # torch.cuda.empty_cache()

            return data

    @classmethod
    def _add_knowledge_to_conv(cls, dataset):
        """
        add knowledge to dataset
        :param dataset:
        :return:
        """
        knw_added_dataset = list()

        count = 0
        length = len(dataset)
        for record in dataset:
            social, event, entity = KnowledgeGenerator.run(texts=record['history'])
            record_plus_knw = {cls.SOCIAL_REL_KEY_NAME: social,
                               cls.EVENT_REL_KEY_NAME: event,
                               cls.ENTITY_REL_KEY_NAME: entity}
            record_plus_knw.update(record)
            knw_added_dataset.append(record_plus_knw)
            print('add knowledge for one record', count, length)
            count += 1

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
        train_df = EmpatheticDialoguesDataset.conv_preprocess(split='train', add_knowledge=True, add_examples=False)
        train_df = pd.DataFrame(train_df)
        train_df['xReact'] = train_df['social_rel'].apply(lambda x: list(x.values())[0]['xReact'])
        train_df['history_str'] = train_df['history'].apply(lambda x: ", ".join(x))
        example_retriever = ExampleRetriever(train_df=train_df, ctx_key_name='label', qs_key_name='history_str',
                                             conv_key_name='original_conv_id')

        count = 0
        length = len(data)
        for record in data:
            record['history_str'] = ", ".join(record['history'])
            record['xReact'] = list(record['social_rel'].values())[0]['xReact']
            record = example_retriever(record)
            new_dataset.append(record)
            print('add examples for one record', count, length)
            count += 1

        return new_dataset

    @classmethod
    def _single_audio_file_preprocessing(cls, record: dict, dataset_path: str, split: str) -> dict:
        """
        extract audio and get data of it
        :param record:
        :return:
        """

        def get_path(row):
            """
            get path of audio file
            :param row:
            :return:
            """
            file_path_last_utter = row[cls.FILE_PATH_KEY_NAME][-1]
            return f"{dataset_path}/{split}/{file_path_last_utter}"

        audio_file_path = get_path(record)

        record[cls.AUDIO_DATA_KEY_NAME] = AudioModule.get_audio_data(file_path=audio_file_path)

        return record

    @classmethod
    def _audio_file_preprocessing(cls, data: list, dataset_path: str, split: str) -> list:
        """
        extract audio and get data of it
        :param data:
        :return:
        """

        def get_path(row):
            """
            get path of audio file
            :param row:
            :return:
            """
            file_path_last_utter = row[cls.FILE_PATH_KEY_NAME][-1]
            return f"{dataset_path}/{split}/{file_path_last_utter}"

        processed_data = list()

        for record in data:
            audio_file_path = get_path(record)

            # if there are some problems with file, the record won't get considered as a record of dataset
            try:
                record[cls.AUDIO_DATA_KEY_NAME] = AudioModule.get_audio_data(file_path=audio_file_path)
                processed_data.append(record)

            except Exception as e:
                pass

        return processed_data

    @classmethod
    def get_from_huggingface(cls):
        if not os.path.exists(f"{cls.PREFIX}/BiMEmpDialogues"):
            zip_path = snapshot_download(repo_id=cls.REPO_ID,
                                         repo_type="dataset",
                                         cache_dir=cls.PREFIX,
                                         token=HUB_ACCESS_TOKEN)

            unzip(zip_path=f"{zip_path}/BiMEmpDialouges.zip", des_path=cls.PREFIX)
        return f"{cls.PREFIX}/BiMEmpDialogues"

    def __getitem__(self, idx: int):
        """
        get item with specific index using dataset[idx]
        :param idx: index
        :return:
        """
        raw_item_data = self.data[idx].copy()
        history, label = raw_item_data['history'], raw_item_data['label']
        item_data = {'history': history, 'label': label}
        if self.include_audio:
            item_data['audio'] = self._single_audio_file_preprocessing(raw_item_data, split=self.split,
                                                                       dataset_path=self.dataset_dir)['audio']

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

    def __len__(self):
        """
        length of dataset
        :return:
        """
        return self.n_sample
