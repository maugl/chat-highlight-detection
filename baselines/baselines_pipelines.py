import sys
sys.path.insert(0, "/..")

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


def load_data(mode):
    if mode == "train":
        file_identifier = "nalcs_w[134579]*_g[13]"
    if mode == "test":
        file_identifier = "nalcs_w[268]*_g[13]"

    chd = ChatHighlightData(chat_dir="../data/final_data", highlight_dir="../data/gt", emote_dir="../data/emotes", frame_rate=30)
    chd.load_data(file_identifier=file_identifier)
    chd.load_emotes()

    param_grid = {
        "measure": ["message_density", "emote_density", "copypasta_density"],
        "window": list(range(50, 201, 25)),
        "step": list(range(20, 101, 20))
    }

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


def param_search_SPP():
    eval_params_SPP = {
        "avg__kw_args": [
            {"N": 5},
            {"N": 50},
            {"N": 500},
        ],
        "SPP__prominence": [0.5, 0.55, 0.6, 0.65, 0.7],
        "SPP__width": [[5, 2000]],
        "SPP__rel_height": [0.4, 0.5, 0.6],
        "SPP__shift": [0.25, 0.3, 0.35]
    }

    best_scores_params = list()

    print(f"{datetime.now().strftime('%Y%m%d_%H_%M_%S')}")

    for i, x, y, p in iter(load_data("train")):
        pl = Pipeline([("avg", FunctionTransformer(moving_avg)),
                       ("scaler", MinMaxScaler()),
                       ("SPP", ScipyPeaks())
                       ])

        gs = GridSearchCV(pl, eval_params_SPP, cv=5, n_jobs=4, scoring=["f1"], refit="f1", verbose=1)
        gs.fit(x.reshape((-1, 1)), y)

        best_scores_params.append({
            "best_params": gs.best_params_,
            "best_score": gs.best_score_,
            "prep_params": p
        })

        if i % 30 == 0:
            print(f"{datetime.now().strftime('%Y%m%d_%H_%M_%S')}: evaluated {i} configurations")
            with open(
                    f"data/analysis/baselines/grid_search/GridSearchCV_SPP_{datetime.now().strftime('%Y%m%d_%H_%M_%S')}_PART_{i}.json",
                    "w") as out_file:
                json.dump(best_scores_params, out_file)

    print(f"{datetime.now().strftime('%Y%m%d_%H_%M_%S')}")
    with open(f"data/analysis/baselines/grid_search/GridSearchCV_SPP_{datetime.now().strftime('%Y%m%d_%H_%M_%S')}.json",
              "w") as out_file:
        json.dump(best_scores_params, out_file)


if __name__ == "__main__":
    parser = ArgumentParser(description="tune, run, evaluate baseline classfiers for highlight prediction on chat"
                                        "messages")
    parser.add_argument("-b", "--baseline", choices=["rtpp", "spp"], help="which baseline to run.\n\trtpp:\treal time"
                                                                          "peak predictor\n\tspp: scipy's find peaks")

    args = parser.parse_args()

    if args.baseline == "spp":
        print(f"{datetime.now().strftime('%Y%m%d_%H_%M_%S')}")
        param_search_SPP()
        print(f"{datetime.now().strftime('%Y%m%d_%H_%M_%S')}")
    if args.baseline == "rtpp":
        pass
