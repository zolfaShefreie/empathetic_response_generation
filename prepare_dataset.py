from utils.interface import BaseInterface
from model_data_process.dataset import EmpatheticDialoguesDataset


class PrepareDatasetInterface(BaseInterface):
    DESCRIPTION = "You can prepare dataset and save it"

    def _run_main_process(self):
        """
        EmpatheticDialoguesDataset load model and apply some changes and get knowledge and examples and save them
        when you call it again this class just data from files
        :return:
        """
        EmpatheticDialoguesDataset(split='train')
        EmpatheticDialoguesDataset(split='test')
        EmpatheticDialoguesDataset(split='validation')


if __name__ == "__main__":
    PrepareDatasetInterface().run()
    