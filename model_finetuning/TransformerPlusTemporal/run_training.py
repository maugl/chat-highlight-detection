import os
import shutil

import data_prep

# this script may be reused for multiple training setups

# define all hyper parameters here
# import as environment variables which can be set in the setup file?
# alternatively paremeterize
DATA_PATH = "./data/"
CHAT_DIR = "final_data/"
HL_DIR = "gt/"
TRAIN_REGEX = "nalcs_w*_g[13]"
VAL_REGEX = "nalcs_w[1-4]*_g2"
TEST_REGEX = "nalcs_w[5-9]*_g2"

TRANSFORMER_MODEL = "./model/TwitchLeagueBERT"
TOKENIZER_NAME = "./model/TwitchLeagueBERT" # path to transformer model directory or
CHUNK_LENGTH = 32
WINDOW_SIZE = 2
CONTEXT_SIZE = (1, 1)
STEP_SIZE = 3
ADDITIONAL_FEATURES = ["message_density"]
ADDITIONAL_FEATURES_ARGS = {
    "message_density":{
        "window_size": 1000,
        "step_size": 1,
        "mvg_avg_N": 1000
    }
}
CHUNKING_COLUMNS = ['highlights', 'input_ids', 'attention_mask'] + [af + "_scaled"for af in ADDITIONAL_FEATURES]

# copy data to node once, use there
# make sure the correct files get copied back to storage from node
#   training logs from trainer
#   best saved model
# make sure that the datasets cache is in an appropriate directory on the node

def copy_data_to_node():
    # implementation specific to SLURM cluster setup
    os.mkdir("./model/")
    shutil.copytree("/netscratch/gutsche/data/TwitchLeagueBert/", "./model/")
    os.mkdir("./data/")
    shutil.copytree("/netscratch/gutsche/data/final_data", "./data/")

def copy_data_to_storage():


if __name__ == "__main__":
    copy_data_to_node()

    dataset = data_prep.prepare_data(
        chat_directory=DATA_PATH + CHAT_DIR,
        higlight_directory=DATA_PATH + HL_DIR,
        additional_features=ADDITIONAL_FEATURES,
        additional_features_args=ADDITIONAL_FEATURES_ARGS,
        chunking_columns=CHUNKING_COLUMNS,
        tokenizer_name=TOKENIZER_NAME
    )

    dataset.save_to_disk("./data/ds_transformer_plus_temporal")

    copy_data_to_storage()