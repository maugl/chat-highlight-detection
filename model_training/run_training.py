import os
import shutil
import uuid
from sacred import Experiment
import train_lm

ex = Experiment("LOLTwitchBertTraining")

# this script may be reused for multiple training setups

# define all hyper parameters here
# import as environment variables which can be set in the setup file?
# alternatively paremeterize


@ex.config
def setup_config():
    model_output_dir = None  # if set, the last model will be saved

    run_id = uuid.uuid4()
    run_name = "LOLTwitchBertTraining"


# copy data to node once, use there
# make sure the correct files get copied back to storage from node
#   training logs from trainer
#   best saved model
# make sure that the datasets cache is in an appropriate directory on the node
def copy_data_to_node():
    print("copying data...")
    try:
        os.mkdir("/tmp/model/")
    except FileExistsError:
        pass
    shutil.copytree("/netscratch/gutsche/data/TwitchLeagueBert/", "/tmp/model/TwitchLeagueBert/")
    try:
        os.mkdir("/tmp/data/")
    except FileExistsError:
        pass
    shutil.copy("/netscratch/gutsche/data/twitch_lol_combined.txt", "/tmp/data/")
    shutil.copytree("/netscratch/gutsche/data/corpus_grouped_dataset", "/tmp/data/corpus_grouped_dataset")

    try:
        os.mkdir("/tmp/run/")
    except FileExistsError:
        pass


"""
@ex.capture
def copy_data_to_storage(run_id, run_name):
    # we use ./my_experiment.py -F run/output/sacred as logging directory
    # implementation specific to SLURM cluster setup
    results_dir = f"/netscratch/gutsche/data/mlm_training/{run_id}"
    try:
        # os.makedirs(results_dir)
        pass
    except FileExistsError:
        pass
    try:
        # os.makedirs(f"{results_dir}/output/")
        shutil.copytree("/tmp/run/output/", f"{results_dir}/output/")
    except Exception as e:
        print("cannot copy run output")
        print(e)
    try:
        # os.makedirs(f"{results_dir}/training/")
        shutil.copytree("/tmp/run/training/", f"{results_dir}/training/")
    except Exception as e:
        print("cannot copy run training")
        print(e)
    try:
        # os.makedirs(f"{results_dir}/model/TwitchLeagueBert/")
        shutil.copytree("/tmp/run_mlm/TwitchLeagueBert/", f"{results_dir}/model/TwitchLeagueBert")
    except Exception as e:
        print("cannot copy model")
        print(e)
"""

@ex.automain
def run_training(run_id,
                 run_name,
                 _seed
                 ):
    copy_data_to_node()

    out_dir = f"/netscratch/gutsche/data/{run_name}_{run_id}"

    try:
        # model training
        print("preparing data / training model")
        train_lm.main(train_model=True,
                      seed=_seed,
                      model_path="/tmp/model/TwitchLeagueBert",
                      data_path="/tmp/data/",
                      output_path=out_dir,
                      load_data_from_hub="Epidot/twitch_lol_corpus_for_mlm_training")
        print("model training complete")
    except Exception as e:
        print("Error in train_model")
        print(e)