from utils.interface import BaseInterface
from model_data_process.dataset import EmpatheticDialoguesDataset, MELDDataset, BiMEmpDialoguesDataset
from settings import MELD_DATASET_PATH, BMEDIALOGUES_PATH


class PrepareDatasetInterface(BaseInterface):
    DESCRIPTION = "You can prepare dataset and save it"

    ARGUMENTS = {
        'dataset_name': {
            'help': 'The dataset that need to prepared',
            'choices': ['MELD', 'EmpatheticDialogues', 'BiMEmpDialogues'],
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
        if self.dataset_name.lower() == 'EmpatheticDialogues'.lower():
            EmpatheticDialoguesDataset(split='train')
            EmpatheticDialoguesDataset(split='test')
            EmpatheticDialoguesDataset(split='validation')

        elif self.dataset_name.lower() == 'MELD'.lower():
            MELDDataset(split='train', dataset_path=MELD_DATASET_PATH)
            MELDDataset(split='test', dataset_path=MELD_DATASET_PATH)
            MELDDataset(split='validation', dataset_path=MELD_DATASET_PATH)

        elif self.dataset_name.lower() == 'BiMEmpDialogues'.lower():
            BiMEmpDialoguesDataset(split='train', dataset_dir=BMEDIALOGUES_PATH)
            BiMEmpDialoguesDataset(split='test', dataset_dir=BMEDIALOGUES_PATH)
            BiMEmpDialoguesDataset(split='validation', dataset_dir=BMEDIALOGUES_PATH)


if __name__ == "__main__":
    PrepareDatasetInterface().run()
    