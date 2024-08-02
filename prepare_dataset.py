from utils.interface import BaseInterface
from model_data_process.dataset import EmpatheticDialoguesDataset, MELDDataset
from settings import MELD_DATASET_PATH


class PrepareDatasetInterface(BaseInterface):
    DESCRIPTION = "You can prepare dataset and save it"

    ARGUMENTS = {
        'dataset_name': {
            'help': 'The dataset that need to prepared',
            'choices': ['MELD', 'EmpatheticDialogues', ],
            'required': False,
            'default': 'EmpatheticDialogues'
        }
    }

    def _run_main_process(self):
        """
        EmpatheticDialoguesDataset load model and apply some changes and get knowledge and examples and save them
        when you call it again this class just data from files
        :return:
        """
        if self.dataset_name == 'EmpatheticDialogues':
            EmpatheticDialoguesDataset(split='train')
            EmpatheticDialoguesDataset(split='test')
            EmpatheticDialoguesDataset(split='validation')

        elif self.dataset_name == 'MELD':
            MELDDataset(split='train', dataset_path=MELD_DATASET_PATH)
            MELDDataset(split='test', dataset_path=MELD_DATASET_PATH)
            MELDDataset(split='validation', dataset_path=MELD_DATASET_PATH)


if __name__ == "__main__":
    PrepareDatasetInterface().run()
    