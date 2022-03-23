import datetime
import glob
import json
from argparse import ArgumentParser
from copy import copy

from sklearn.model_selection import ParameterGrid
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

from sklearn.preprocessing import MinMaxScaler

import analysis
from analysis import highlight_span, moving_avg
from baselines.RealTimePeakPredictor import RealTimePeakPredictor
from baselines.ScipyPeaks import ScipyPeaks
from chat_measures import message_density
from data_loading import load_chat, load_highlights, remove_missing_matches, cut_same_length

import warnings
warnings.filterwarnings('ignore')


#TODO replace with parameterized method from data_loading
def load_experiments_data(file_regex, load_random, random_state, data_path="data"):
    chat = load_chat(f"{data_path}/final_data", file_identifier=file_regex, load_random=load_random, random_state=random_state)
    highlights = load_highlights(f"{data_path}/gt", file_identifier=file_regex)  # nalcs_w1d3_TL_FLY_g2
    remove_missing_matches(chat, highlights)
    matches_meta = {}
    data_totals = {
        "video_count": 0,  # number of videos in dataset
        "video_length_secs": 0,  # total length of all videos combined
        "highlight_count": 0,  # number of total highlights
        "highlight_length_secs": 0,  # total length of all highlights combined
    }
    # calculate measures to operate on
    for match in chat.keys():
        ch_match, hl_match = cut_same_length(chat[match], highlights[match])
        chat[match] = ch_match
        highlights[match] = hl_match

        cd_message_density = message_density(ch_match, window_size=300)

        hl_spans = highlight_span(hl_match)
        hl_lens = [e - s + 1 for s, e in hl_spans]
        hl_count = len(hl_lens)

        matches_meta[match] = {
            "highlight_spans": hl_spans,
            "highlight_lengths": hl_lens,
            "highlight_count": hl_count,
            "highlights": hl_match,
            "chat_message_density": cd_message_density,
            "cd_message_density_smoothed": moving_avg(MinMaxScaler().fit_transform(cd_message_density.reshape(-1, 1)).ravel(),
                                                      N=1500)
        }

        data_totals["video_count"] += 1
        data_totals["video_length_secs"] += len(ch_match) / 30  # total video length in seconds (30fps)
        data_totals["highlight_count"] += hl_count

    return matches_meta


def save_results(directory, matches_data, file_name):
    with open(f"{directory}/{file_name}.json", "w") as out_file:
        json.dump(matches_data, out_file)


# TODO delete and replace with sklearn pipeline
def evaluate_config(config_file, match_data):
    scores = dict()
    with open(config_file, "r") as in_file:
        config = json.load(in_file)
        params = config["config"]
        gold_total = list()
        pred_total = list()

        for match, prediction in config.items():
            # TODO: change that along with the output
            if match == "config":
                continue
            if "scale" in params:
                gold_data = match_data[match]["highlights"][::params["scale"]]
            else:
                gold_data = match_data[match]["highlights"]
            scores[match] = dict()
            scores[match]["scores"] = eval_scores(gold_data, prediction)
            gold_total.extend(gold_data)
            pred_total.extend(prediction)

    return scores, eval_scores(gold_total, pred_total), params


# TODO replace with sklearn pipeline
def eval_scores(gold, pred):
    p, r, f, _ = precision_recall_fscore_support(gold, pred, average="binary")
    acc = accuracy_score(gold, pred)

    return {"precision": list(p),
            "recall": list(r),
            "f-score": list(f),
            "accuracy": acc
            }


if __name__ == "__main__":
    parser = ArgumentParser(description="tune, run, evaluate baseline classfiers for highlight prediction on chat"
                                        "messages")
    parser.add_argument("-p", "--data_path", help="path to the data directory")
    parser.add_argument("-b", "--baseline", choices=["rtpp", "spp"], help="which baseline to run.\n\trtpp:\treal time"
                                                                          "peak predictor\n\tspp: scipy's find peaks")
    parser.add_argument("-c", "--config_file", help="path to the config file for the baseline parameter values")
    parser.add_argument("-o", "--out_path", help="directory to store the results in")
    parser.add_argument("-a", "--action", choices=["tr", "te", "r", "e", "test"], help="tr: run tuning on multiple parameter"
                                                                               "combinations defined in config_file\nte"
                                                                               ": evaluate all tested parameters in tr"
                                                                               "with gold data in data_path and tuning"
                                                                               "results in results_path. Store"
                                                                               "evaluation results in out_path"
                                                                               "\nr: run baseline on test data in"
                                                                               "data_path and with parameters defined in"
                                                                               "config_file\ne: evaluate results from "
                                                                               "running r", required=True)
    parser.add_argument("-rp", "--results_path", help="Where to get tr results from.")

    args = parser.parse_args()

    # before anything else, perform test if requested
    if args.action == "test":
        with open(args.config_file, "r") as in_file:
            baseline_params = json.load(in_file)
            params = baseline_params[args.baseline]
            matches = load_experiments_data("nalcs_*", load_random=3, random_state=42, data_path=args.data_path)
        if args.baseline == "spp":
            spp = ScipyPeaks(**params)
            for match, data in matches.items():
                pred = spp.predict(data["cd_message_density_smoothed"])
                data["pred_spp"] = pred
                print(eval_scores(data["highlights"], pred))

        if args.baseline == "rtpp":
            lag = params["lag"]
            scale = params["scale"]
            del params["scale"]
            for name, data in matches.items():
                msg_density_scaled = data["cd_message_density_smoothed"][::scale]
                data["pred_chat_message_density"] = data["cd_message_density_smoothed"][::scale]
                del data["chat_message_density"]
                rtpp = RealTimePeakPredictor(array=msg_density_scaled[:lag], **params)
                rtpp.fit(msg_density_scaled[lag:])
                pred = [abs(d) for d in rtpp.predict()]
                data["pred_rtpp"] = pred
                # scale down gold data as well, maybe losing some highlights?
                data["highlights"] = data["highlights"][::scale]
                print(eval_scores(data["highlights"], pred))

        analysis.plot_matches(matches)

    # data loading

    # load training data
    if args.action in ["tr", "te"]:
        # load train data
        files = "nalcs*g[13]"  # "nalcs_w1d3_TL_FLY_g*" # "nalcs_w*d3_*g1"
        matches = load_experiments_data(files, load_random=None, random_state=None, data_path=args.data_path)
    if args.action in ["r", "e"]:
        # load test data
        files = "nalcs*g2"  # "nalcs_w1d3_TL_FLY_g*" # "nalcs_w*d3_*g1"
        matches = load_experiments_data(files, load_random=None, random_state=None, data_path=args.data_path)

    if args.action == "tr":
        with open(args.config_file, "r") as in_file:
            baseline_params = json.load(in_file)

        if args.baseline == "rtpp":
            peaks_params = baseline_params["rtpp"]
            param_grid = ParameterGrid(peaks_params)
            print(args.baseline)
            print(f"calculating {len(param_grid)} parameter combinations")
            for i, config in enumerate(param_grid):
                lag = config["lag"]
                preds_configs = dict()
                preds_configs["config"] = copy(config)
                print(f"{i:03d}", config)
                scale = config["scale"]
                del config["scale"]
                for name, m in matches.items():
                    msg_density_scaled = m["cd_message_density_smoothed"][::scale]
                    rtpp = RealTimePeakPredictor(array=msg_density_scaled[:lag], **config)
                    rtpp.fit(msg_density_scaled[lag:])
                    preds_configs[name] = rtpp.predict()
                save_results(args.out_path, preds_configs, f"rtpp_config_{i:03d}")

        if args.baseline == "spp":
            peaks_params = baseline_params["spp"]
            param_grid_scipy = ParameterGrid(peaks_params["scipy_params"])
            param_grid = list()
            for shift in peaks_params["shift"]:
                for scipy_params in param_grid_scipy:
                    param_grid.append({
                        "shift": shift,
                        "scipy_params": scipy_params
                    })

            print(f"calculating {len(param_grid)} parameter combinations")
            for i, config in enumerate(param_grid):
                preds_configs = dict()
                preds_configs["config"] = config
                print(config)
                for name, m in matches.items():
                    spp = ScipyPeaks(**config)
                    # TODO: should be preds_configs["matches"][name]
                    # sticking to this pattern for current testing
                    preds_configs[name] = spp.predict(m["cd_message_density_smoothed"]).tolist()
                save_results(args.out_path, preds_configs, f"spp_config_{i:03d}")

    # evaluate multiple parameter combinations
    if args.action == "te":
        tuning_predictions = glob.glob(f"{args.results_path}/*_config_{'[0-9]' * 3}.json")
        print(tuning_predictions)
        config_scores = dict()
        for config_file in tuning_predictions:
            baseline_name = config_file.split("/")[-1].split("_")[0]
            if baseline_name not in config_scores:
                config_scores[baseline_name] = list()

            scores, total_scores, params = evaluate_config(config_file, matches)
            config_scores[baseline_name].append({"params": params,
                                                 "match_scores": scores,
                                                 "total_scores": total_scores
                                                 })

        with open(f"{args.out_path}/eval_configs_{datetime.datetime.now().strftime('%Y%m%d_%H_%M_%S')}.json", "w") as out_file:
            json.dump(config_scores, out_file, indent=4)


    test_data_file_identifier = "test_data_predictions"
    if args.action == "r":
        # run baseline on data in data_path and with parameters defined in config_file
        with open(args.config_file, "r") as in_file:
            test_params = json.load(in_file)
            config = test_params[args.baseline]
        preds = list()
        for name, m in matches.items():
            spp = ScipyPeaks(scipy_params=config)
            preds.append({
                "match": name,
                "pred": spp.predict(m["cd_message_density_smoothed"]).tolist()
            })
        save_results(f"{args.out_path}", preds, f"{args.baseline}_{test_data_file_identifier}")

    if args.action == "e":
        # e: evaluate results from running r
        with open(f"{args.baseline}_{test_data_file_identifier}", "r") as in_file:
            test_preds = json.load(in_file)
            total_pred = list()
            total_gold = list()
            scores = list()
            for m in test_preds:
                name = m["name"]
                pred = m["pred"]
                total_pred.extend(pred)
                total_gold.extend(matches[name]["highlights"])
                scores.append(eval_scores(matches[name]["highlights"], pred))


