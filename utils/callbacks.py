from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl
import os


class SaveHistoryCallback(TrainerCallback):

    FILE_NAME = 'state_histories'

    def _save(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        save state history after each epoch
        :param args:
        :param state:
        :param control:
        :param kwargs:
        :return:
        """
        log_path = f"{args.logging_dir}/{self.FILE_NAME}"
        if not os.path.exists(os.path.dirname(log_path)):
            try:
                os.makedirs(os.path.dirname(log_path))
            except OSError as exc:
                print(exc)
                pass
        with open(log_path, mode="w+") as file:
            file.write(str(state.log_history))

    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        save state history after each epoch
        :param args:
        :param state:
        :param control:
        :param kwargs:
        :return:
        """
        self._save(args=args, state=state, control=control, **kwargs)

    def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        save Event called at the end of training.
        """
        self._save(args=args, state=state, control=control, **kwargs)
        