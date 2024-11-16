import json
import os
import ast
import re

from model_data_process.dataset import EmpatheticDialoguesDataset, BiMEmpDialoguesDataset
from settings import BMEDIALOGUES_PATH
from utils.interface import BaseInterface
from utils.metrics import ExtraMetricsManagement, Metrics


class EvaluateTextInterface(BaseInterface):
    DESCRIPTION = "You can run the evaluation generated responses with fed, empathy and dynaeval metrics using" \
                  " this interface. consider this script is write based on format of result file in this project."

    ARGUMENTS = {
        'model': {
            'help': 'rsearch_model. own is our research',
            'choices': ['cem', 'cab', 'knowdt', 'iamm', 'care', 'own'],
            'required': True
        },

        'result_path': {
            'help': 'path of result',
            'required': True,
            'default': None
        },

        'dataset_name': {
            'help': 'The dataset that need to prepared',
            'choices': ['EmpatheticDialogues', 'BiMEmpDialogues'],
            'required': False,
            'default': 'BiMEmpDialogues'
        },

        'batch_size': {
            'help': 'The batch size for evaluation.',
            'type': int,
            'required': False,
            'default': 16
        },

    }

    def validate_result_path(self, value):
        if value is not None and not os.path.exists(value):
            raise Exception('you have to show to path of result file')
        return value

    def load_test_data(self):
        """
        load test_data based on dataset_name
        :return:
        """
        if self.dataset_name.lower() == 'EmpatheticDialogues'.lower():
            return EmpatheticDialoguesDataset(split='test')

        elif self.dataset_name.lower() == 'BiMEmpDialogues'.lower():
            return BiMEmpDialoguesDataset(split='test', dataset_dir=BMEDIALOGUES_PATH, include_audio=False)

        else:
            raise Exception('wrong dataset_name')

    def load_result(self):
        """
        load result
        :return:
        """
        with open(self.result_path, mode='r', encoding='utf-8') as file:
            content = file.read()
            return ast.literal_eval(content)

    def combine_result_test_data_own(self, test_data):
        """
        add generated response to each record
        :param test_data: test dataset
        :return: combined data, metrics
        """

        def load_result(result_path):
            """
            load result
            :return:
            """
            with open(result_path, mode='r', encoding='utf-8') as file:
                content = file.read()
                return ast.literal_eval(content)

        result = load_result(self.result_path)
        combined_data = list()
        for i in range(len(test_data)):
            record_dict = {'generated_response': result['text_generator_result']['pred'][i]}
            record_dict.update(test_data[i])
            combined_data.append(record_dict)
        return combined_data, result['metric']

    def save(self, data_plus, metrics):
        file_name, _ = os.path.splitext(os.path.basename(self.result_path))
        file_path = f"{os.path.dirname(self.result_path)}/extra_{file_name}.json"
        with open(file_path, mode='w', encoding='utf-8') as file:
            json_str = json.dumps(metrics)
            file.write(json_str)
            file.write("\n")

            for record in data_plus:
                json_str = json.dumps(record)
                file.write(json_str)
                file.write("\n")

    def combine_result_test_data_care(self):
        """
        load result and test_data based on care research
        :return: result and none as metrics
        """
        name_map = {'history': 'dial', 'labels': 'ref', 'generated_response': 'hyp_b'}
        result = list()
        with open(self.result_path, mode='r', encoding='utf-8') as file:
            for content in file:
                if content:
                    record = ast.literal_eval(content)
                    new_record = {new_key_name: record[old_key_name] for new_key_name, old_key_name in name_map.items()}
                    more_info_record = {key: value for key, value in record.items() if key not in name_map.values()}
                    result.append({**new_record, **more_info_record})

        return result

    def combine_result_test_data_cab(self):
        """

        :return:
        """
        regex_pattern = r"((Context):(\[.*\]) \n?)((Situation):(\[.*\]) \n?)((Knowledge):(\[.*\]) \n?)((Emotion_s):" \
                        r"(\[.*\]) \n?)((Emotion_l):(\[.*\]) \n?)((Emotion):(\[.*\]) \n?)((Act):(.*) \n?)" \
                        r"((Greedy):(.*) \n?)((Ref):(.*) \n?)"

        key_names = ['Context', 'Situation', 'Knowledge', 'Emotion_s', 'Emotion_l', 'Emotion', 'Act', 'Greedy', 'Ref']
        key_names_list_type = ['Context', 'Situation', 'Knowledge', 'Emotion_s', 'Emotion_l', 'Emotion', ]
        name_map = {'history': 'Context', 'labels': 'Ref', 'generated_response': 'Greedy'}

        content = open(self.result_path, mode='r', encoding='utf-8').read()
        records = self.extract_records(regex_pattern=regex_pattern, text=content, key_names=key_names,
                                       key_names_list_type=key_names_list_type)

        result = list()
        for each in records:
            new_record = {new_key_name: each[old_key_name] for new_key_name, old_key_name in name_map.items()}
            more_info_record = {key: value for key, value in each.items() if key not in name_map.values()}
            result.append({**new_record, **more_info_record})

        return result

    def combine_result_test_data_knowdt(self):
        """

        :return:
        """
        regex_pattern = r"((Emotion): (.*)\n?)((Pred Emotions): (.*)\n?)?((x_intent):(\[.*\])\n?)?((x_need):" \
                        r"(\[.*\])\n?)?((x_want):(\[.*\])\n?)?((x_effect):(\[.*\])\n?)?((x_react):(\[.*\])\n?)?" \
                        r"((Context):(\[.*\])\n?)((Beam):(.*)\n?)?((Greedy):(.*)\n?)((Ref):(.*)\n?)"

        key_names = ['Emotion', 'Pred Emotions', 'x_need', 'x_intent', 'x_want', 'x_effect', 'x_react', 'Context',
                     'Beam', 'Greedy', 'Ref']
        key_names_list_type = ['x_need', 'x_intent', 'x_want', 'x_effect', 'x_react', 'Context', ]
        name_map = {'history': 'Context', 'labels': 'Ref', 'generated_response': 'Greedy'}

        content = open(self.result_path, mode='r', encoding='utf-8').read()
        records = self.extract_records(regex_pattern=regex_pattern, text=content, key_names=key_names,
                                       key_names_list_type=key_names_list_type)

        result = list()
        for each in records:
            new_record = {new_key_name: each[old_key_name] for new_key_name, old_key_name in name_map.items()}
            more_info_record = {key: value for key, value in each.items() if key not in name_map.values()}
            result.append({**new_record, **more_info_record})

        return result

    def combine_result_test_data_cem(self):
        """

        :return:
        """
        regex_pattern = r"((Emotion): (.*)\n?)((Pred Emotions): (.*)\n?)?((x_intent):(\[.*\])\n?)?((x_need):" \
                        r"(\[.*\])\n?)?((x_want):(\[.*\])\n?)?((x_effect):(\[.*\])\n?)?((x_react):(\[.*\])\n?)?" \
                        r"((Context):(\[.*\])\n?)((Beam):(.*)\n?)?((Greedy):(.*)\n?)((Ref):(.*)\n?)"
        key_names = ['Emotion', 'Pred Emotions', 'x_need', 'x_intent', 'x_want', 'x_effect', 'x_react', 'Context',
                     'Beam', 'Greedy', 'Ref']
        key_names_list_type = ['x_need', 'x_intent', 'x_want', 'x_effect', 'x_react', 'Context', ]
        name_map = {'history': 'Context', 'labels': 'Ref', 'generated_response': 'Greedy'}

        content = open(self.result_path, mode='r', encoding='utf-8').read()
        records = self.extract_records(regex_pattern=regex_pattern, text=content, key_names=key_names,
                                       key_names_list_type=key_names_list_type)

        result = list()
        for each in records:
            new_record = {new_key_name: each[old_key_name] for new_key_name, old_key_name in name_map.items()}
            more_info_record = {key: value for key, value in each.items() if key not in name_map.values()}
            result.append({**new_record, **more_info_record})

        return result

    def combine_result_test_data_iamm(self):
        """

        :return:
        """
        regex_pattern = r"((Emotion): (.*)\n?)((Pred Emotions): (.*)\n?)?((x_intent):(\[.*\])\n?)?((x_need):" \
                        r"(\[.*\])\n?)?((x_want):(\[.*\])\n?)?((x_effect):(\[.*\])\n?)?((x_react):(\[.*\])\n?)?" \
                        r"((Context):(\[.*\])\n?)((Beam):(.*)\n?)?((Greedy):(.*)\n?)((Ref):(.*)\n?)"
        key_names = ['Emotion', 'Pred Emotions', 'x_need', 'x_intent', 'x_want', 'x_effect', 'x_react', 'Context',
                     'Beam', 'Greedy', 'Ref']
        key_names_list_type = ['x_need', 'x_intent', 'x_want', 'x_effect', 'x_react', 'Context', ]
        name_map = {'history': 'Context', 'labels': 'Ref', 'generated_response': 'Greedy'}

        content = open(self.result_path, mode='r', encoding='utf-8').read()
        records = self.extract_records(regex_pattern=regex_pattern, text=content, key_names=key_names,
                                       key_names_list_type=key_names_list_type)

        result = list()
        for each in records:
            new_record = {new_key_name: each[old_key_name] for new_key_name, old_key_name in name_map.items()}
            more_info_record = {key: value for key, value in each.items() if key not in name_map.values()}
            result.append({**new_record, **more_info_record})

        return result

    def extract_records(self, regex_pattern, text, key_names: list, key_names_list_type: list = None):
        """
        extract records text and return as list of dictionaries
        :param regex_pattern:
        :param text:
        :param key_names:
        :param key_names_list_type:
        :return:
        """
        records = list()
        match_result = re.findall(regex_pattern, text)
        for each_match in match_result:
            record = dict()
            for index_group in range(len(each_match)):
                if each_match[index_group] in key_names:
                    # based on all regex in text key, value would be in this format (key):(value)
                    value = each_match[index_group + 1]
                    if key_names_list_type is not None and each_match[index_group] in key_names_list_type:
                        value = ast.literal_eval(value)
                    record[each_match[index_group]] = value
            records.append(record)

        return records

    def compute_base_metrics(self, test_data, label_key_name='labels', generated_res_key_name='generated_response'):
        """

        :param test_data:
        :param label_key_name:
        :param generated_res_key_name:
        :return:
        """
        labels = [each[label_key_name] for each in test_data]
        pred = [each[generated_res_key_name] for each in test_data]
        metric_obj = Metrics(tokenizer=None, task_list=list())
        return metric_obj.compute_text_generator_metric(labels=labels, pred=pred, need_preprocessing=False)

    def _run_main_process(self):

        if self.model == 'own':
            test_data, metrics_p = self.combine_result_test_data_own(test_data=self.load_test_data())
        else:
            test_data = getattr(self, f'combine_result_test_data_{self.model}')()
            metrics_p = self.compute_base_metrics(test_data)

        test_data_plus, metrics_extra = ExtraMetricsManagement.compute(test_data=test_data,
                                                                       history_key_name='history',
                                                                       label_key_name='labels',
                                                                       generated_res_key_name='generated_response',
                                                                       batch_size=self.batch_size)
        self.save(data_plus=test_data_plus, metrics={**metrics_p, **metrics_extra})


if __name__ == "__main__":
    EvaluateTextInterface().run()
