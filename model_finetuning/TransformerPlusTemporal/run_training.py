import os
import shutil
import uuid

import datasets

import data_prep
from sacred import Experiment
from train_model import train_model
from hub_token import HUB_TOKEN

ex = Experiment("TransformerPlusTemporal")

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
    chunk_length = 32
    window_size = 2
    context_size = (1, 1)
    step_size = 3
    additional_features = ["message_density"]
    additional_features_args = {
        "message_density": {
            "window_size": 1000,
            "step_size": 1,
            "mvg_avg_N": 1000
        }
    }
    chunking_columns = ['highlights', 'input_ids', 'attention_mask'] + [af + "_scaled" for af in additional_features] + ["highlights_raw"]
    run_id = uuid.uuid4()
    run_name = "TransformerPlusTemporal"
    precomputed_dataset_path = None
    ds_name_hub = None
    do_training = True
    ds_intermediate = None


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
def copy_data_to_storage(run_id, run_name):
    # we use ./my_experiment.py -F run/output/sacred as logging directory
    # implementation specific to SLURM cluster setup
    results_dir = f"/netscratch/gutsche/data/training/results/{run_id}_{run_name}"
    try:
        os.makedirs(results_dir)
    except FileExistsError:
        pass
    shutil.copytree(f"/tmp/data/ds_{run_id}_{run_name}", f"{results_dir}/output/ds_{run_id}_{run_name}")


@ex.capture
def copy_training_output_to_storage(run_id, run_name):
    # we use ./my_experiment.py -F run/output/sacred as logging directory
    # implementation specific to SLURM cluster setup
    results_dir = f"/netscratch/gutsche/data/training/results/{run_id}_{run_name}"
    try:
        os.makedirs(results_dir)
    except FileExistsError:
        pass
    try:
        shutil.copytree("/tmp/run/output/", f"{results_dir}/output/{run_id}_{run_name}")
    except Exception as e:
        print("cannot copy '/tmp/run/output/' to storage")
        print(e)
    try:
        shutil.copytree("/tmp/run/training/", f"{results_dir}/training/{run_id}_{run_name}")
    except Exception as e:
        print("cannot copy '/tmp/run/training/' to storage")
        print(e)


@ex.automain
def run_training(data_path,
                 chat_dir,
                 hl_dir,
                 transformer_model,
                 additional_features,
                 additional_features_args,
                 chunking_columns,
                 tokenizer_name,
                 run_id,
                 chunk_length,
                 train_regex,
                 val_regex,
                 test_regex,
                 run_name,
                 window_size,
                 context_size,
                 _seed,
                 precomputed_dataset_path,
                 ds_name_hub,
                 do_training,
                 ds_intermediate
                 ):

    print(transformer_model)
    print(tokenizer_name)
    print(do_training)

    copy_data_to_node()

    sequence_length = sum(context_size) + window_size

    if ds_name_hub:
        dataset = datasets.load_dataset(ds_name_hub, use_auth_token=HUB_TOKEN)
    elif precomputed_dataset_path:
        dataset = load_precomputed_dataset(precomputed_dataset_path)
    else:
        dataset = data_prep.prepare_data(
            chat_directory=data_path + chat_dir,
            highlight_directory=data_path + hl_dir,
            additional_features=additional_features,
            additional_features_args=additional_features_args,
            chunking_columns=chunking_columns,
            tokenizer_name=tokenizer_name,
            max_input_len=sequence_length * chunk_length,
            train_identifier=train_regex,
            val_identifier=val_regex,
            test_identifier=test_regex,
            seed=_seed,
            ds_intermediate=ds_intermediate
        )

        # copy data after preparation in case training fails, then we can start off from previous run
        dataset.save_to_disk(f"{data_path.rstrip('/')}/ds_{run_id}_{run_name}")
        try:
            copy_data_to_storage(run_id, run_name)
        except Exception as e:
            print("cannot copy dataset to storage")
            print(e)
        try:
            if "/" in transformer_model:
                mn = transformer_model.split("/")[-1]
            else:
                mn = transformer_model
            dataset.push_to_hub(f"Epidot/private_fuetal2017_highlights_temporal_preprocessed_{mn}_oversampled", private=True, token=HUB_TOKEN)
        except Exception as e:
            print("cannot push dataset to hub")
            print(e)

    print("data preparation complete")

    if do_training:
        try:
            train_model(dataset,
                        lm_path=transformer_model,
                        output_dir=f"/tmp/run/training/{run_id}_{run_name}",
                        additional_features_size=len(additional_features) * sequence_length,  # might have to be adjusted for future features
                        window_size=window_size,
                        sequence_length=sequence_length,
                        num_dist_categories=None,  # only needed for additional objective
                        num_dist_steps=None,  # only needed for additional objective
                        main_loss_ratio=1,
                        run_id=f"{run_id}_{run_name}")
            print("model training complete")
        except Exception as e:
            print("Error in train_model")
            print(e)

        copy_training_output_to_storage()
