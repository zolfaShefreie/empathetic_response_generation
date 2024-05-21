import numpy as np
import torch


class NewVersionDialogues:
    """
    process that convert dataset to version of context (history) and response

    WARNING:
        this class support list-dict format of two-party conversations
        and dialogue must look like  (user_utter, system_utter)* sequences.
    """
    ORIGINAL_CONV_ID_KEY_NAME = 'original_conv_id'

    def __init__(self,
                 conv_id_key_name: str,
                 turn_key_name: str,
                 utter_key_name: str,
                 other_conv_features: list = None,
                 other_utter_features: list = None,
                 new_conv_each_sys_responses=True,
                 responses_in_history=True,
                 context_key_name='context',
                 label_key_name='label'):
        """
        config of processes
        :param conv_id_key_name:
        :param turn_key_name:
        :param utter_key_name:
        :param other_conv_features: list of other key names for each dialogues
        :param other_utter_features: list of other key names for each utterances
        :param new_conv_each_sys_responses: a boolean that shows
                                            True: make new conversation record for each system response
                                            False: consider last response for made conversation record
        :param responses_in_history: add system responses to history or not
        :param context_key_name:
        :param label_key_name:
        """

        # related to raw dataset
        self.conv_id_key_name = conv_id_key_name
        self.turn_key_name = turn_key_name
        self.utter_key_name = utter_key_name
        self.other_conv_features = other_conv_features
        self.other_utter_features = other_utter_features

        # related to result and settings
        self.new_conv_each_sys_responses = new_conv_each_sys_responses
        self.responses_in_history = responses_in_history

        # related to result
        self.context_key_name = context_key_name
        self.label_key_name = label_key_name

    def _merge_conv_record(self, record: dict, current_conv: dict, is_label=False) -> dict:
        """
        add new utterance record to conversation record as label or history
        :param record:
        :param current_conv:
        :param is_label:
        :return: conversation with new record information
        """
        merged_result = dict()
        merged_result.update(current_conv)

        merged_result[self.ORIGINAL_CONV_ID_KEY_NAME] = record[self.conv_id_key_name]
        for key_name in self.other_conv_features:
            merged_result[key_name] = record.get(key_name, None)

        if not is_label:
            for key_name in self.other_utter_features:
                merged_result[key_name] = current_conv.get(key_name, list()) + [record.get(key_name, None)]

            merged_result[self.context_key_name] = current_conv.get(self.context_key_name, list()) + \
                                                  [record.get(self.utter_key_name, None)]

        else:
            for key_name in self.other_utter_features:
                merged_result[f"{key_name}_{self.label_key_name}"] = record.get(key_name, None)
            merged_result[self.label_key_name] = record.get(self.utter_key_name, str())

        return merged_result

    def _empty_conv_dict(self, conv_id) -> dict:
        """
        :param conv_id: id of this conversation
        :return: new empty conversation struct
        """
        return {
            self.ORIGINAL_CONV_ID_KEY_NAME: None,
            self.context_key_name: list(),
            self.label_key_name: str(),
            self.conv_id_key_name: conv_id
        }

    def reformat(self, raw_dataset) -> list:
        """
        :param raw_dataset:
        :return: a list of conversations
        """

        conversations = list()
        conv_id = 0
        is_user_turn = True
        current_conversation = self._empty_conv_dict(conv_id=conv_id)
        previous_record = dict()

        for record in raw_dataset:

            # if it is the user turn
            if is_user_turn:
                # if it is the record of new conversation and we save last system response (it didn't save before)
                # and it is not first record of dataset
                if current_conversation[self.ORIGINAL_CONV_ID_KEY_NAME] != record[self.conv_id_key_name] and \
                        previous_record:
                    # merge conversation with previous record as label
                    if not self.new_conv_each_sys_responses:
                        conversations.append(self._merge_conv_record(record=previous_record,
                                                                     current_conv=current_conversation,
                                                                     is_label=True))
                    # reset whole current conversation
                    conv_id += 1
                    current_conversation = self._empty_conv_dict(conv_id=conv_id)

                # if we still in same conversation and we must save system utter as history
                # and it is not first record of dataset
                elif self.responses_in_history and previous_record:
                    # merge prev_record (sys_utter) to current_conversation
                    current_conversation = self._merge_conv_record(record=previous_record,
                                                                   current_conv=current_conversation,
                                                                   is_label=False)

                # merge conversation with this record (user utter)
                current_conversation = self._merge_conv_record(current_conv=current_conversation,
                                                               record=record,
                                                               is_label=False)

            # if it is system turn and we make new conversation for each system response
            elif self.new_conv_each_sys_responses:
                # add and merge this record to current_conversation as label
                conversations.append(self._merge_conv_record(record=record,
                                                             current_conv=current_conversation,
                                                             is_label=True))
                # reset by id
                conv_id += 1
                current_conversation[self.conv_id_key_name] = conv_id

            # setup for end of the loop
            previous_record = record
            is_user_turn = not is_user_turn

        if is_user_turn and previous_record:
            conversations.append(self._merge_conv_record(record=previous_record,
                                                         current_conv=current_conversation,
                                                         is_label=True))

        return conversations


class Pipeline:

    def __init__(self, functions: list):
        self.functions = functions

    def __call__(self, data):
        for func in self.functions:
            data = func(data)
        return data


class TextCleaner:
    PUNC = '''!()-[]{.};:'"\,<>/?@#$%^&*_~`|’“”…—–'''

    def __init__(self, have_label=True):
        self.have_label = have_label

    @classmethod
    def _clean_single_text(cls, text: str) -> str:
        """
        return cleaned text
        :param text:
        :return:
        """
        text = text.lower()
        for each in cls.PUNC:
            text = text.replace(each, ' ')

        return text

    @classmethod
    def _clean_list_of_texts(cls, texts: list) -> list:
        """
        clean list of texts
        :param texts:
        :return:
        """
        return [cls._clean_single_text(text) for text in texts]

    def __call__(self, sample):
        """

        :param sample:
        :return:
        """
        texts = sample[0][0] if self.have_label else sample[0]

        # value of each (row, col) can be list type or str type
        cleaned_result = self._clean_list_of_texts(texts) if isinstance(texts, list) or isinstance(texts, np.ndarray) \
            else self._clean_single_text(texts)

        if self.have_label:
            return np.array([cleaned_result]), np.array(sample[-1])
        else:
            return np.array([cleaned_result])


class ConversationFormatter:
    """
    join utterance with special tokens
    """

    SPECIAL_TOKEN_SPLIT_UTTERANCE = "<USEP>"

    def __init__(self, train_split=True):
        self.train_split = train_split

    def __call__(self, sample):
        """
        get the sample and create context based on history and get the last utterance as query
        :param sample:
        :return: context, last utterance, response
        """
        texts = sample[0] if self.train_split else sample

        last_utter = texts[-1]
        conversation = f"{self.SPECIAL_TOKEN_SPLIT_UTTERANCE}".join(texts[:-1])

        if self.train_split:
            return np.array([conversation]), np.array([last_utter]), np.array([sample[-1]])
        else:
            return np.array([conversation]), np.array([last_utter])


class ToTensor:
    """
    Convert ndarrays to Tensors
    """

    def __call__(self, sample):
        return tuple(torch.from_numpy(np.array(each)) for each in sample)


class ConversationTokenizer:

    def __init__(self, tokenizer, train_split=True, max_len=128, new_special_tokens=None):
        """
        :param tokenizer:
        :param train_split:
        :param max_len:
        :param new_special_tokens:
        """
        self.tokenizer = tokenizer

        if new_special_tokens:
            tokenizer.add_special_tokens(new_special_tokens)

        self.train_split = train_split
        self.MAX_LEN = max_len

    def __call__(self, sample):
        """
        :param sample: get context, last_utter and response (response is optional)
        :return:
        """

        inputs = self.tokenizer.encode_plus((sample[0][0], sample[1][0]),
                                            add_special_tokens=True,
                                            max_length=self.MAX_LEN,
                                            padding='max_length',
                                            return_attention_mask=True,
                                            return_token_type_ids=True,
                                            truncation=True)

        if self.train_split:
            label = self.tokenizer.encode_plus(sample[-1][0],
                                               add_special_tokens=True,
                                               max_length=self.MAX_LEN,
                                               padding='max_length',
                                               return_attention_mask=True,
                                               return_token_type_ids=True,
                                               truncation=True)
            return inputs['input_ids'], inputs['attention_mask'], inputs['token_type_ids'], \
                   label['input_ids'], label['attention_mask'], label['token_type_ids']
        else:
            return inputs['input_ids'], inputs['attention_mask'], inputs['token_type_ids']


class ConvertInputToDict:

    def __init__(self, dict_meta_data: dict):
        """

        :param dict_meta_data: meta data about how the output must be look like
         (key, index in sample)
        """
        self.dict_meta_data = dict_meta_data

    def __call__(self, sample):
        """

        :param sample:
        :return:
        """
        return {key: sample[index] for key, index in self.dict_meta_data.items()}
    

class PreProcessEncoderDecoderInput:

    # it is coded based on output of ConversationTokenizer class
    OUTPUT_KEYS = {'input_ids': 0, 'attention_mask': 1, 'token_type_ids': 2,
                   'decoder_input_ids': 3, 'decoder_attention_mask': 4, 'labels': 3}

    def __init__(self, tokenizer, dict_meta_data: dict = None):
        """
        :param tokenizer:
        :param dict_meta_data: meta data about how the output must be look like
         (key, index in sample)
        """
        self.dict_meta_data = self.OUTPUT_KEYS if dict_meta_data is None else dict_meta_data
        self.tokenizer = tokenizer

    def __call__(self, sample) -> dict:
        """
        convert it to dictionary format
        :param sample:
        :return:
        """

        data = {key: np.array(sample[index], copy=True).tolist()
                for key, index in self.dict_meta_data.items()
                if index < len(sample)}
        if 'labels' in data:
            data["labels"] = [
                -100 if token == self.tokenizer.pad_token_id else token for token in data["labels"]
            ]

        data = {key: torch.from_numpy(np.array(value)) for key, value in data.items()}
        return data
