import numpy as np
import torch
from transformers import EvalPrediction
from transformers.trainer_utils import PredictionOutput


class NewVersionDialogues:
    """
    process that convert dataset to version of context (history) and response

    WARNING:
        this class support list-dict format of conversations
        and dialogue must look like  (user_utter, system_utter)* sequences for two_party dialogues
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

    def two_party_reformat(self, raw_dataset) -> list:
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

    def multi_party_reformat(self, raw_dataset) -> list:
        """
        each utterance is a response
        :param raw_dataset:
        :return: a list of conversations
        """

        conversations = list()
        conv_id = 0
        current_conversation = self._empty_conv_dict(conv_id=conv_id)
        previous_record = dict()

        for record in raw_dataset:

            # if it is the record of new conversation
            if current_conversation[self.ORIGINAL_CONV_ID_KEY_NAME] != record[self.conv_id_key_name]:
                # reset current_conversation
                conv_id += 1
                current_conversation = self._empty_conv_dict(conv_id=conv_id)

            # add a conversation with new response
            conversations.append(self._merge_conv_record(record=record,
                                                         current_conv=current_conversation,
                                                         is_label=True))

            # merge record with new record for next record history
            current_conversation = self._merge_conv_record(current_conv=current_conversation,
                                                           record=record,
                                                           is_label=False)

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

    def __init__(self, texts_key_name: str = 'history'):
        self.texts_key_name = texts_key_name

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
        # texts = sample[0][0] if self.have_label else sample[0]
        texts = sample[self.texts_key_name]

        # value of each (row, col) can be list type or str type
        cleaned_result = self._clean_list_of_texts(texts) if isinstance(texts, list) or isinstance(texts, np.ndarray) \
            else self._clean_single_text(texts)

        sample[self.texts_key_name] = cleaned_result
        return sample


class ConversationFormatter:
    """
    join utterance with special tokens
    works with dictionaries as input and output
    """

    SPECIAL_TOKEN_SPLIT_UTTERANCE = "<USEP>"

    def __init__(self, history_key_name: str = 'history', gen_label_key_name: str = 'label',
                 last_utter_key_name: str = 'last_utter', utter_sep: str = None):
        self.history_key_name = history_key_name
        self.gen_label_key_name = gen_label_key_name
        self.last_utter_key_name = last_utter_key_name
        self.utter_sep = utter_sep if utter_sep else self.SPECIAL_TOKEN_SPLIT_UTTERANCE

    def __call__(self, sample):
        """
        get the sample and create context based on history and get the last utterance as query
        :param sample:
        :return: context, last utterance, response
        """
        texts = sample[self.history_key_name]

        last_utter = texts[-1]
        conversation = f"{self.utter_sep}".join(texts[:-1])
        sample[self.history_key_name] = conversation
        sample[self.last_utter_key_name] = last_utter

        return sample


class ToNumpy:

    def __init__(self, unwanted_keys: list = None):
        self.unwanted_keys = unwanted_keys if unwanted_keys is not None else list()

    def __call__(self, sample):
        if isinstance(sample, dict):
            result = {k: np.array(v) if isinstance(v, list) or isinstance(v, int) or isinstance(v, np.ndarray)
                      else np.array([v]) for k, v in sample.items() if k not in self.unwanted_keys}
            result.update({k: v for k, v in sample.items() if k in self.unwanted_keys})
            return result

        return tuple([np.array(v) if isinstance(v, list) or isinstance(v, int) or isinstance(v, np.ndarray)
                      else np.array([v]) for v in sample])


class KnowledgeFormatter:
    """works with dictionaries as input and output"""

    # same as knowledge generator
    SOCIAL_INTERACTION_REL = {'xIntent': 'because X wanted',
                              'xReact': 'as a result, X feels',
                              'xNeed': 'but before, X needed',
                              'xWant': 'as a result, X wants',
                              'xEffect': 'as a result, X wil', }
    EVENT_CENTERED_REL = {'Causes': 'causes',
                          'HinderedBy': 'can be hindered by',
                          'xReason': 'because',
                          'isAfter': 'happens after',
                          'isBefore': 'happens before',
                          'HasSubEvent': 'includes the event/action',
                          'isFilledBy': 'blank can be filled by', }
    PHYSICAL_ENTITIES = {'ObjectUse': 'is used for',
                         'CapableOf': 'is/are capable of',
                         'HasProperty': 'can be characterized by being/having',
                         'Desires': 'desires', }

    def __init__(self, social_rel_key_name: str = 'social_rel', event_rel_key_name: str = 'event_rel',
                 entity_rel_key_name: str = 'entity_rel',
                 react_rel_key_name: str = 'react_rel',
                 use_special_tokens: bool = True):
        self.social_rel_key_name = social_rel_key_name
        self.event_rel_key_name = event_rel_key_name
        self.entity_rel_key_name = entity_rel_key_name
        self.react_rel_key_name = react_rel_key_name
        self.use_special_tokens = use_special_tokens

    @staticmethod
    def _join_all_nodes_for_same_rel(nodes: list) -> str:
        """
        join all result of generate nodes for a text and rel
        :param nodes:
        :return:
        """
        if len(nodes) == 0:
            return None
        if len(nodes) == 1:
            return nodes[0]
        return ", ".join(nodes[:-1]) + f" and {nodes[-1]}"

    def _convert_nodes_with_rel(self, nodes: list, rel: str, root_node: str = None) -> str:
        """
        make a text with root_node, rel, nodes
        :param nodes: nodes shows generated nodes
        :param rel: name of relation that is in SOCIAL_INTERACTION_REL or EVENT_CENTERED_REL or PHYSICAL_ENTITIES keys
        :param root_node:
        :return: text
        """
        if len(nodes) == 0:
            return str()

        readable_rel = None
        for categories in [self.SOCIAL_INTERACTION_REL, self.PHYSICAL_ENTITIES, self.EVENT_CENTERED_REL]:
            if rel in categories.keys():
                if self.use_special_tokens:
                    readable_rel = f"[{rel}]"
                else:
                    readable_rel = categories[rel]
        return f"{root_node}{' ' if root_node is not None else ''}{readable_rel} {self._join_all_nodes_for_same_rel(nodes)}"

    def _formatting_social_interaction_rel_results(self, results: dict) -> tuple:
        """
        return relations with text format
        :param results:
        :return: xreact results, other rel results
        """
        social_knw_result = list(results.values())[0]
        other_rel_reformat = ".\n".join([self._convert_nodes_with_rel(nodes=rel_nodes,
                                                                     rel=rel_name,
                                                                     root_node=None)
                                         for rel_name, rel_nodes in social_knw_result.items()
                                         if rel_name != 'xReact'])
        return ", ".join(social_knw_result['xReact']), other_rel_reformat

    def _formatting_event_entity_rel_results(self, results: dict) -> str:
        """
        return relations with text format
        :param results:
        :return: text version of these data
        """
        return ".\n".join([self._convert_nodes_with_rel(nodes=nodes,
                                                       rel=rel_name,
                                                       root_node=text)
                           for text, rel_nodes in results.items()
                           for rel_name, nodes in rel_nodes.items()])

    def __call__(self, sample) -> dict:
        """
        apply reformatting for knowledge
        :param sample:
        :return:
        """
        # apply social reformatting
        sample[self.react_rel_key_name], sample[self.social_rel_key_name] = self._formatting_social_interaction_rel_results(sample[self.social_rel_key_name])

        # apply other reformatting
        other_map_func = {rel_key_name: self._formatting_event_entity_rel_results
                          for rel_key_name in [self.event_rel_key_name, self.entity_rel_key_name]
                          if rel_key_name in sample.keys()}

        return {k: v if k not in other_map_func.keys() else other_map_func[k](v) for k, v in sample.items()}


class KnowledgeTokenizer:
    """works with dictionaries as input and output"""

    def __init__(self,
                 tokenizer,
                 max_len=128,
                 new_special_tokens=None,
                 react_key_name: str = 'react_rel',
                 social_rel_key_name: str = 'social_rel',
                 event_rel_key_name: str = 'event_rel',
                 entity_rel_key_name: str = 'entity_rel',
                 use_special_tokens: bool = True):
        """
        WARNING: key_names is used for prefix of result
        :param tokenizer:
        :param max_len:
        :param new_special_tokens:
        :param react_key_name:
        :param social_rel_key_name:
        :param event_rel_key_name:
        :param entity_rel_key_name:
        :param use_special_tokens:
        """
        self.tokenizer = tokenizer
        self.tokenizer.truncation_side = 'left'

        if use_special_tokens:
            new_tokens = {
                'additional_special_tokens': [f"[{rel}]" for categories in [KnowledgeFormatter.SOCIAL_INTERACTION_REL,
                                                                            KnowledgeFormatter.EVENT_CENTERED_REL,
                                                                            KnowledgeFormatter.PHYSICAL_ENTITIES]
                                              for rel in categories.keys()]
            }
            if new_special_tokens:
                new_special_tokens['additional_special_tokens'] = new_tokens['additional_special_tokens'] + \
                                                                  new_special_tokens.get('additional_special_tokens',
                                                                                         list())
                self.tokenizer.add_special_tokens(new_special_tokens)
            else:
                self.tokenizer.add_special_tokens(new_tokens)

        if new_special_tokens and not use_special_tokens:
            self.tokenizer.add_special_tokens(new_special_tokens)

        self.MAX_LEN = max_len
        # key_name configs
        self.react_key_name = react_key_name
        self.social_rel_key_name = social_rel_key_name
        self.event_rel_key_name = event_rel_key_name
        self.entity_rel_key_name = entity_rel_key_name

    def __call__(self, sample):
        """
        warning make sure to apply ToNumpy before using this function
        :param sample:
        :return:
        """
        data = dict()
        for key_name in [self.react_key_name, self.social_rel_key_name,
                         self.event_rel_key_name, self.entity_rel_key_name]:
            if key_name in sample.keys():
                inputs = self.tokenizer.encode_plus(sample[key_name][0],
                                                    add_special_tokens=True,
                                                    max_length=self.MAX_LEN,
                                                    padding='max_length',
                                                    return_attention_mask=True,
                                                    return_token_type_ids=True,
                                                    truncation=True)
                data[f"{key_name}_input_ids"] = inputs['input_ids']
                data[f"{key_name}_attention_mask"] = inputs['attention_mask']
                data[f"{key_name}_token_type_ids"] = inputs['token_type_ids']

        data.update(sample)
        return data


class ExampleTokenizer:

    def __init__(self, tokenizer, example_key_name: str = 'examples', number_of_examples: int = 5, max_len: int = 300):
        self.tokenizer = tokenizer
        self.example_key_name = example_key_name
        self.number_of_examples = number_of_examples
        self.MAX_LEN = max_len

    def __call__(self, sample):
        """
        warning make sure to apply ToNumpy before using this function
        :param sample:
        :return:
        """
        data = dict()
        for i in range(min(len(sample[self.example_key_name]), self.number_of_examples)):

            inputs = self.tokenizer.encode_plus(sample[self.example_key_name][i],
                                                add_special_tokens=True,
                                                max_length=self.MAX_LEN,
                                                padding='max_length',
                                                return_attention_mask=True,
                                                return_token_type_ids=True,
                                                truncation=True)

            data[f"example_{i}_input_ids"] = inputs['input_ids']
            data[f"example_{i}_attention_mask"] = inputs['attention_mask']
            data[f"example_{i}_token_type_ids"] = inputs['token_type_ids']

        data.update(sample)
        return data


class ToTensor:
    """
    Convert ndarrays to Tensors
    """

    def __call__(self, sample):
        if isinstance(sample, dict):
            return {k: torch.from_numpy(np.array(v)) for k, v in sample.items()}
        return tuple(torch.from_numpy(np.array(each)) for each in sample)


class ToLong:

    def __init__(self, wanted_list: list = None):
        self.wanted_list = wanted_list

    def __call__(self, sample):
        if isinstance(sample, dict):
            return {k: torch.Tensor(v).type(torch.long) if k in self.wanted_list else torch.Tensor(v)
                    for k, v in sample.items()}
        return tuple(torch.Tensor(each).type(torch.long) for each in sample)


class ConversationTokenizer:
    """works with dictionaries as input and output"""

    def __init__(self,
                 tokenizer,
                 source_max_len=128,
                 label_max_len=100,
                 new_special_tokens=None,
                 last_utter_key_name: str = 'last_utter',
                 history_key_name: str = 'history',
                 gen_label_key_name: str = 'label',
                 context_ids_key_name: str = 'input_ids',
                 context_mask_key_name: str = 'attention_mask',
                 context_token_type_key_name: str = 'token_type_ids',
                 gen_label_ids_key_name: str = 'labels',
                 gen_label_mask_key_name: str = 'gen_label_mask',
                 gen_label_token_type_key_name: str = 'gen_label_token_type', ):
        """

        :param tokenizer:
        :param source_max_len:
        :param label_max_len:
        :param new_special_tokens:
        :param last_utter_key_name:
        :param history_key_name:
        :param gen_label_key_name:
        :param context_ids_key_name:
        :param context_mask_key_name:
        :param context_token_type_key_name:
        :param gen_label_ids_key_name:
        :param gen_label_mask_key_name:
        :param gen_label_token_type_key_name:
        """
        self.tokenizer = tokenizer
        self.tokenizer.truncation_side = 'left'

        if new_special_tokens:
            self.tokenizer.add_special_tokens(new_special_tokens)

        self.source_max_len = source_max_len
        self.label_max_len = label_max_len
        # key_name configs
        self.history_key_name = history_key_name
        self.gen_label_key_name = gen_label_key_name
        self.last_utter_key_name = last_utter_key_name
        self.context_ids_key_name = context_ids_key_name
        self.context_mask_key_name = context_mask_key_name
        self.context_token_type_key_name = context_token_type_key_name
        self.gen_label_ids_key_name = gen_label_ids_key_name
        self.gen_label_mask_key_name = gen_label_mask_key_name
        self.gen_label_token_type_key_name = gen_label_token_type_key_name

    def __call__(self, sample):
        """
        warning make sure to apply ToNumpy before using this function
        :param sample: get context, last_utter and response (response is optional)
        :return:
        """
        inputs = self.tokenizer.encode_plus(sample[self.history_key_name][0], sample[self.last_utter_key_name][0],
                                            add_special_tokens=True,
                                            max_length=self.source_max_len,
                                            padding='max_length',
                                            return_attention_mask=True,
                                            return_token_type_ids=True,
                                            truncation=True)

        sample[self.context_ids_key_name] = inputs['input_ids']
        sample[self.context_mask_key_name] = inputs['attention_mask']
        sample[self.context_token_type_key_name] = inputs['token_type_ids']

        if self.gen_label_key_name in sample.keys():
            label = self.tokenizer.encode_plus(sample[self.gen_label_key_name][0],
                                               add_special_tokens=True,
                                               max_length=self.label_max_len,
                                               padding='max_length',
                                               return_attention_mask=True,
                                               return_token_type_ids=True,
                                               truncation=True)
            sample[self.gen_label_ids_key_name] = label['input_ids']
            sample[self.gen_label_mask_key_name] = label['attention_mask']
            sample[self.gen_label_token_type_key_name] = label['token_type_ids']

        return sample


class FilterSample:

    def __init__(self, wanted_keys: list):
        """

        :param wanted_keys:
        """
        self.wanted_keys = wanted_keys

    def __call__(self, sample):
        """

        :param sample:
        :return:
        """
        return {k: v for k, v in sample.items() if k in self.wanted_keys}


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


class PreProcessEncoderDecoderInputTupleVersion:
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


class PreProcessEncoderDecoderInputDictVersion:

    def __init__(self, tokenizer, gen_label_key_name: str):
        """
        :param tokenizer:
        :param gen_label_key_name:
        """
        self.gen_label_key_name = gen_label_key_name
        self.tokenizer = tokenizer

    def __call__(self, sample) -> dict:
        """
        convert it to dictionary format
        :param sample:
        :return:
        """

        if self.gen_label_key_name in sample:
            sample[self.gen_label_key_name] = [
                -100 if token == self.tokenizer.pad_token_id else token for token in sample[self.gen_label_key_name]
            ]

        return sample


class AudioFeatureExtractor:

    def __init__(self, feature_extractor, audio_key_name='audio', result_prefix_key_name='audio'):
        """
        :param feature_extractor: like AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base")
        :param audio_key_name:
        :param result_prefix_key_name:
        """
        self.feature_extractor = feature_extractor
        self.audio_key_name = audio_key_name
        self.result_prefix_key_name = result_prefix_key_name

    def __call__(self, sample: dict) -> dict:
        """
        apply feature extractor on audio
        :param sample:
        :return:
        """
        result = self.feature_extractor(sample[self.audio_key_name]['array'],
                                        sampling_rate=self.feature_extractor.sampling_rate,
                                        max_length=16000,
                                        truncation=True)

        sample.update({f"{self.result_prefix_key_name}_{k}": v[0] for k, v in result.items()})
        return sample


class PostProcessResult:

    def __init__(self, task_list: list, tokenizer=None):
        self.tokenizer = tokenizer
        self._validate_task_list(task_list=task_list)
        self.task_list = task_list

    def _validate_task_list(self, task_list: list):
        for task in task_list:
            if getattr(self, f'{task}_result', None) is None:
                raise Exception(f"{task} doesn't exist")

    @classmethod
    def classifier_result(cls, pred, labels):
        result = 1 / (1 + np.exp(-pred))
        result = np.argmax(result, axis=-1).tolist()
        return {'pred': result, 'labels': labels if isinstance(labels, list) else labels.tolist()}

    def text_generator_result(self, pred, labels):
        pred_str = self.tokenizer.batch_decode(pred, skip_special_tokens=True)
        labels[labels == -100] = self.tokenizer.pad_token_id
        label_str = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        return {'pred': pred_str, 'labels': label_str}

    def compute(self, pred: PredictionOutput) -> dict:
        result = dict()

        if len(self.task_list) == 1:
            func_task = getattr(self, f"{self.task_list[0]}_result", None)
            if func_task is not None:
                result.update({f'{self.task_list[0]}_result': func_task(pred=pred.predictions, labels=pred.label_ids)})

        else:
            for i, task_name in enumerate(self.task_list):
                pred_task, labels_task = pred.predictions[i], pred.label_ids[i]
                func_task = getattr(self, f"{task_name}_result", None)
                if func_task is not None:
                    result.update({f"{task_name}_result": func_task(pred=pred_task, labels=labels_task)})

        return result
