import json
import os
import ast

from model_data_process.dataset import EmpatheticDialoguesDataset, BiMEmpDialoguesDataset
from settings import BMEDIALOGUES_PATH
from utils.interface import BaseInterface

from utils.metrics import ExtraMetricsManagement


class EvaluateTextInterface(BaseInterface):
    DESCRIPTION = "You can run the evaluation generated responses with fed, empathy and dynaeval metrics using" \
                  " this interface. consider this script is write based on format of result file in this project."

    ARGUMENTS = {

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
        }

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

    def combine_result_test_data(self, test_data, result: dict):
        """
        add generated response to each record
        :param test_data: test dataset
        :param result:
        :return: combined data, metrics
        """
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

    def _run_main_process(self):

        test_data, metrics_p = self.combine_result_test_data(test_data=self.load_test_data(),
                                                             result=self.load_result())

        test_data_plus, metrics_extra = ExtraMetricsManagement.compute(test_data=test_data,
                                                                       history_key_name='history',
                                                                       label_key_name='labels',
                                                                       generated_res_key_name='generated_response')
        self.save(data_plus=test_data_plus, metrics={**metrics_p, **metrics_extra})


if __name__ == "__main__":
    EvaluateTextInterface().run()
