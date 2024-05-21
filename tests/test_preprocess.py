import pytest
from datasets import load_dataset

from utils import preprocessing


class TestPreprocessReformat:

    @pytest.fixture
    def raw_dataset(self):
        """
        :return:
        """
        return load_dataset("empathetic_dialogues", split="train[:13]")

    @pytest.fixture
    def config_reformat_class_all_history_all_new_conv(self):
        return {
            'conv_id_key_name': 'conv_id',
            'turn_key_name': 'utterance_idx',
            'utter_key_name': 'utterance',
            'other_conv_features': ['context', 'prompt', 'selfeval', 'tags'],
            'other_utter_features': ['speaker_idx'],
            'new_conv_each_sys_responses': True,
            'responses_in_history': True,
            'context_key_name': 'history',
            'label_key_name': 'label'
        }

    @pytest.fixture
    def config_reformat_class_res_history_all_new_conv(self):
        return {
            'conv_id_key_name': 'conv_id',
            'turn_key_name': 'utterance_idx',
            'utter_key_name': 'utterance',
            'other_conv_features': ['context', 'prompt', 'selfeval', 'tags'],
            'other_utter_features': ['speaker_idx'],
            'new_conv_each_sys_responses': True,
            'responses_in_history': False,
            'context_key_name': 'history',
            'label_key_name': 'label'
        }

    @pytest.fixture
    def config_reformat_class_all_history_no_new_conv(self):
        return {
            'conv_id_key_name': 'conv_id',
            'turn_key_name': 'utterance_idx',
            'utter_key_name': 'utterance',
            'other_conv_features': ['context', 'prompt', 'selfeval', 'tags'],
            'other_utter_features': ['speaker_idx'],
            'new_conv_each_sys_responses': False,
            'responses_in_history': True,
            'context_key_name': 'history',
            'label_key_name': 'label'
        }

    def test_can_make_dialogues(self, raw_dataset, config_reformat_class_all_history_no_new_conv):
        process_manager = preprocessing.NewVersionDialogues(**config_reformat_class_all_history_no_new_conv)

        new_version_dataset = process_manager.reformat(raw_dataset=raw_dataset)
        assert len(new_version_dataset) == 2
        assert new_version_dataset[0]['label'] == 'Oh was this something that happened because of an argument?'

    def test_can_include_history_correct(self, raw_dataset, config_reformat_class_res_history_all_new_conv):
        process_manager = preprocessing.NewVersionDialogues(**config_reformat_class_res_history_all_new_conv)

        new_version_dataset = process_manager.reformat(raw_dataset=raw_dataset)
        assert len(new_version_dataset[2]['history']) == 3
        assert len(new_version_dataset[1]['history']) == 2
        assert len(new_version_dataset[0]['history']) == 1


class TestConversationTransforms:

    @pytest.fixture
    def conversation(self):
        return [
            'hi how are you?',
            "fine thanks, and you?",
            "great, nice to see you"
        ], "nice to meet you too"

    def test_conversation_formatter_without_res(self, conversation):
        history, _ = conversation
        context, last_utter = preprocessing.ConversationFormatter(train_split=False)(history)
        assert last_utter[0] == history[-1]
        assert context[0] == history[0] + preprocessing.ConversationFormatter.SPECIAL_TOKEN_SPLIT_UTTERANCE + history[1]

    def test_conversation_formatter_with_res(self, conversation):
        history, response = conversation
        context, last_utter, res = preprocessing.ConversationFormatter(train_split=True)((history, response))
        assert last_utter[0] == history[-1]
        assert context[0] == history[0] + preprocessing.ConversationFormatter.SPECIAL_TOKEN_SPLIT_UTTERANCE + history[1]
        assert response == res[0]

