import os
import shutil
import uuid

import datasets

import data_prep
from sacred import Experiment
from train_model import train_model

from hub_token import HUB_TOKEN

ex = Experiment("TransformerLikeFu")

# this script may be reused for multiple training setups

# define all hyper parameters here
# import as environment variables which can be set in the setup file?
# alternatively paremeterize


@ex.config
def setup_config():
    data_path = "/tmp/data/"
    chat_dir = "final_data/"
    hl_dir = "gt/"
    train_regex = "nalcs_w*_g[13]"  # train: nalcs_w*_g[13]
    val_regex = "nalcs_w[1-4]*_g2"  # val: nalcs_w[1-4]*_g2
    test_regex = "nalcs_w[5-9]*_g2"  # test: nalcs_w[5-9]*_g2

    model_output_dir = None  # if set, the last model will be saved

    transformer_model = "roberta-base"  # "/tmp/model/TwitchLeagueBert" # path to transformer model directory or name of model from huggingface
    tokenizer_name = "roberta-base"  # "/tmp/model/TwitchLeagueBert" # path to transformer model directory or name of model from huggingface
    window_size = 7 * 30
    step_size = 1 * 30

    run_id = uuid.uuid4()
    run_name = "TransformerLikeFu"
    precomputed_dataset_path = None
    ds_name_hub = None
    train = True


# copy data to node once, use there
# make sure the correct files get copied back to storage from node
#   training logs from trainer
#   best saved model
# make sure that the datasets cache is in an appropriate directory on the node
def copy_data_to_node():
    print("copying data...")
    # implementation specific to SLURM cluster setup
    try:
        os.mkdir("/tmp/model/")
    except FileExistsError:
        pass
    shutil.copytree("/netscratch/gutsche/data/TwitchLeagueBert/", "/tmp/model/TwitchLeagueBert/")

    try:
        os.mkdir("/tmp/data/")
    except FileExistsError:
        pass
    shutil.copytree("/netscratch/gutsche/data/final_data/", "/tmp/data/final_data/")
    shutil.copytree("/netscratch/gutsche/data/gt/", "/tmp/data/gt/")

    try:
        os.mkdir("/tmp/run/")
    except FileExistsError:
        pass


def load_precomputed_dataset(data_path):
    """
    copies a dataset from the storage drive to the node and loads it to be used
    :param data_path:
    :return:
    """
    print("loading precomputed dataset")
    try:
        os.mkdir("/tmp/data/")
    except FileExistsError:
        pass

    shutil.copytree(data_path, f"/tmp/data/{data_path.strip('/').split('/')[-1]}")

    ds = datasets.load_from_disk(f"/tmp/data/{data_path.strip('/').split('/')[-1]}")
    return ds


@ex.capture
def copy_training_output_to_storage(run_id, run_name, results_dir):
    # we use ./my_experiment.py -F run/output/sacred as logging directory
    # implementation specific to SLURM cluster setup
    try:
        os.makedirs(results_dir)
    except FileExistsError:
        pass
    try:
        shutil.copytree("/tmp/run/output/", f"{results_dir}/output/")
    except Exception as e:
        print("cannot copy '/tmp/run/output/' to storage")
        print(e)
    try:
        shutil.copytree("/tmp/run/training/", f"{results_dir}/training/")
    except Exception as e:
        print("cannot copy '/tmp/run/training/' to storage")
        print(e)


@ex.capture
def copy_data_to_storage(run_id, run_name, results_dir):
    # we use ./my_experiment.py -F run/output/sacred as logging directory
    # implementation specific to SLURM cluster setup

    try:
        os.makedirs(results_dir + "/output/")
        pass
    except FileExistsError:
        pass
    shutil.copytree(f"/tmp/data/ds_{run_id}_{run_name}", f"{results_dir}/data/ds_{run_id}_{run_name}")


@ex.automain
def run_training(data_path,
                 chat_dir,
                 hl_dir,
                 tokenizer_name,
                 transformer_model,
                 run_id,
                 train_regex,
                 val_regex,
                 test_regex,
                 run_name,
                 window_size,
                 step_size,
                 train,
                 _seed,
                 precomputed_dataset_path,
                 ds_name_hub
                 ):
    print(ds_name_hub)
    copy_data_to_node()

    out_dir = f"/tmp/run/training/{run_id}_{run_name}"

    if ds_name_hub:
        dataset = datasets.load_dataset(ds_name_hub, use_auth_token=HUB_TOKEN)
    elif precomputed_dataset_path:
        dataset = load_precomputed_dataset(precomputed_dataset_path)
    else:
        dataset = data_prep.prepare_data(
            chat_directory=data_path + chat_dir,
            highlight_directory=data_path + hl_dir,
            tokenizer_name=tokenizer_name,
            window_len=window_size,
            step=step_size,
            train_identifier=train_regex,
            val_identifier=val_regex,
            test_identifier=test_regex,
            seed=_seed
        )
        dataset.save_to_disk(f"{data_path.rstrip('/')}/ds_{run_id}_{run_name}")
        # copy data after preparation in case training fails, then we can start off from previous run
        copy_data_to_storage(run_id, run_name, results_dir=f"/netscratch/gutsche/data/training/{run_id}_{run_name}")

    print("data preparation complete")

    if train:
        os.makedirs(out_dir)
        try:
            train_model(dataset,
                        tokenizer_path=tokenizer_name,
                        lm_path=transformer_model,
                        output_dir=out_dir,
                        run_id=f"{run_id}_{run_name}")
            print("model training complete")
        except Exception as e:
            print("Error in train_model")
            print(e)
        copy_training_output_to_storage(results_dir=f"/netscratch/gutsche/data/training/{run_id}_{run_name}")
