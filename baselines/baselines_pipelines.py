import sys
sys.path.insert(0, "../")

from argparse import ArgumentParser

from data_loading import ChatHighlightData
import numpy as np
from datetime import datetime
import json

from utils import moving_avg

from sklearn.pipeline import Pipeline
from sklearn.model_selection import ParameterGrid, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import MinMaxScaler
from ScipyPeaks import ScipyPeaks
from RealTimePeakPredictor import RealTimePeakPredictor


def load_data(mode, ch_dir, hl_dir, em_dir, param_grid):
    file_identifier = ""
    if mode == "small":
        file_identifier = "nalcs_w1*_g1"
    if mode == "train":
        file_identifier = "nalcs_w[134579]*_g[13]"
    if mode == "test":
        file_identifier = "nalcs_w[268]*_g[13]"

    chd = ChatHighlightData(chat_dir=ch_dir, highlight_dir=hl_dir, emote_dir=em_dir, frame_rate=30)
    chd.load_data(file_identifier=file_identifier)
    chd.load_emotes()

    pg = list(ParameterGrid(param_grid))

    for i, params in enumerate(pg):
        chd.set_window_step(window=params["window"], step=params["step"])

        x_data = chd.get_chat_measure(params["measure"])
        y_data = chd.get_highlight_data()

        x = np.empty(0)
        y = np.empty(0)
        for m in x_data.keys():
            x = np.concatenate([x, x_data[m]])
            y = np.concatenate([y, y_data[m]])
        yield i, x, y, params


def param_search(eval_params, model, data_loader):
    best_scores_params = list()

    print(f"{datetime.now().strftime('%Y%m%d_%H_%M_%S')}")

    for i, x, y, p in data_loader:
        pl = Pipeline([("avg", FunctionTransformer(moving_avg)),
                       ("scaler", MinMaxScaler()),
                       ("clf", model())
                       ])

        gs = GridSearchCV(pl, eval_params, cv=5, n_jobs=4, scoring=["f1"], refit="f1", verbose=1)
        gs.fit(x.reshape((-1, 1)), y)

        best_scores_params.append({
            "best_params": gs.best_params_,
            "best_score": gs.best_score_,
            "prep_params": p
        })
        if i % 30 == 0:
            print(f"{datetime.now().strftime('%Y%m%d_%H_%M_%S')}: evaluated {i} configurations")
            with open(
                    f"../data/analysis/baselines/grid_search/GridSearchCV_{type(model()).__name__}_{datetime.now().strftime('%Y%m%d_%H_%M_%S')}_PART_{i}.json",
                    "w") as out_file:
                json.dump(best_scores_params, out_file)

    print(f"{datetime.now().strftime('%Y%m%d_%H_%M_%S')}")
    with open(
            f"../data/analysis/baselines/grid_search/GridSearchCV_{type(model()).__name__}_{datetime.now().strftime('%Y%m%d_%H_%M_%S')}.json",
            "w") as out_file:
        json.dump(best_scores_params, out_file)

# example run
# python3 baselines_pipelines.py -m train -b spp -p "/home/mgut1/data" -o

if __name__ == "__main__":
    parser = ArgumentParser(description="tune, run, evaluate baseline classfiers for highlight prediction on chat"
                                        "messages")
    parser.add_argument("-m", "--mode", choices=["small", "train", "test"])
    parser.add_argument("-b", "--baseline", choices=["rtpp", "spp"], help="which baseline to run.\n\trtpp:\treal time peak predictor")
    parser.add_argument("-p", "--data_path", help="path to the data directory")
    parser.add_argument("-o", "--out_path", help="directory to store the results in")

    args = parser.parse_args()

    prep_param_grid = {
        "measure": ["message_density", "emote_density", "copypasta_density"],
        "window": list(range(50, 201, 25)),
        "step": list(range(20, 101, 20))
    }

    eval_params = dict()
    model = None

    if args.baseline == "spp":
        eval_params = {
            "avg__kw_args": [
                {"N": 5},
                {"N": 50},
                {"N": 500},
            ],
            "clf__prominence": [0.5, 0.55, 0.6, 0.65, 0.7],
            "clf__width": [[5, 2000]],
            "clf__rel_height": [0.4, 0.5, 0.6],
            "clf__shift": [0.25, 0.3, 0.35]
        }
        model = ScipyPeaks
    if args.baseline == "rtpp":
        eval_params = eval_params_RTPP = {
            "avg__kw_args": [
                {"N": 5},
                {"N": 50},
                {"N": 500},
            ],
            "clf__lag": [25, 30, 35],
            "clf__threshold": [1, 2],
            "clf__influence": [0.7],
        }
        model = RealTimePeakPredictor

    dat = iter(load_data(mode=args.mode, ch_dir=f"{args.data_path}/final_data/", hl_dir=f"{args.data_path}/gt/",
                         em_dir=f"{args.data_path}/emotes/", param_grid=prep_param_grid))
    param_search(eval_params=eval_params, model=model, data_loader=dat)
