import os
import json
import shutil
from datetime import datetime

from transformers import TrainerCallback


class FileWriterCallback(TrainerCallback):
    """
    A bare [`TrainerCallback`] that just writes the logs to a file
    https://discuss.huggingface.co/t/logs-of-training-and-validation-loss/1974/4
    """

    def __init__(self, log_file_path=None, write_interval=0, log_file_final_path=None):
        self.log_history = list()
        self.log_file_path = log_file_path.rstrip("/")
        self.log_file_final_path = log_file_final_path.rstrip("/")
        self.log_file_name = f"run_log_{datetime.now().strftime('%Y%m%d_%H_%M_%S')}.jsonl"
        self.write_interval = write_interval

        self._setup_output_path(self.log_file_path)
        self._setup_output_path(self.log_file_final_path)

    def on_log(self, args, state, control, logs=None, **kwargs):
        _ = logs.pop("total_flos", None)
        if state.is_local_process_zero:
            with open(f'{self.log_file_path}/{self.log_file_name}', "a") as log_file:
                log_file.write(json.dumps(logs) + "\n")

    def on_evaluate(self, args, state, control, **kwargs):
        shutil.copy(f'{self.log_file_path}/{self.log_file_name}', f'{self.log_file_final_path}/{self.log_file_name}')

    def on_train_end(self, args, state, control, **kwargs):
        shutil.copy(f'{self.log_file_path}/{self.log_file_name}', f'{self.log_file_final_path}/{self.log_file_name}')

    def _setup_output_path(self, path):
        if not os.path.isdir(path):
            os.makedirs(path + "/")
        with open(f'{path}/{self.log_file_name}', "w"):
            pass