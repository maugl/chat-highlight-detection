import parser
from argparse import ArgumentParser
from pprint import pprint

import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import find_peaks
from sklearn.preprocessing import MinMaxScaler

from analysis import load_chat, load_highlights, remove_missing_matches, cut_same_length, message_density, \
    highlight_span, plot_matches, moving_avg


class RealTimePeakPredictor():
    """
    theory:
    https://stackoverflow.com/a/22640362

    implementation
    https://stackoverflow.com/a/56451135
    Spike detection algorithm
    """
    def __init__(self, array, lag, threshold, influence, scaler=MinMaxScaler, smoothing=moving_avg,
                 smoothing_strength=1500):
        """

        :param array: prime algorithm with some data points
        :param lag: minimum number of datapoints with which to prime alorithm, determines how much your data will be
        smoothed and how adaptive the algorithm is to changes in the long-term average of the data
        :param threshold: number of standard deviations from the moving mean above which the algorithm will classify a
        new datapoint as being a signal
        :param influence: determines the influence of signals on the algorithm's detection threshold
        """
        self.y = list(array)
        self.length = len(self.y)
        self.lag = lag
        self.threshold = threshold
        self.influence = influence
        self.signals = [0] * len(self.y)
        self.filteredY = np.array(self.y).tolist()
        self.avgFilter = [0] * len(self.y)
        self.stdFilter = [0] * len(self.y)
        self.avgFilter[self.lag - 1] = np.mean(self.y[0:self.lag])
        self.stdFilter[self.lag - 1] = np.std(self.y[0:self.lag])

        self.fitted = False

    def thresholding_algo(self, new_value):
        self.y.append(new_value)
        i = len(self.y) - 1
        self.length = len(self.y)
        if i < self.lag:
            return 0
        elif i == self.lag:
            self.signals = [0] * len(self.y)
            self.filteredY = np.array(self.y).tolist()
            self.avgFilter = [0] * len(self.y)
            self.stdFilter = [0] * len(self.y)
            self.avgFilter[self.lag] = np.mean(self.y[0:self.lag])
            self.stdFilter[self.lag] = np.std(self.y[0:self.lag])
            return 0

        self.signals += [0]
        self.filteredY += [0]
        self.avgFilter += [0]
        self.stdFilter += [0]
        if abs(self.y[i] - self.avgFilter[i - 1]) > self.threshold * self.stdFilter[i - 1]:
            if self.y[i] > self.avgFilter[i - 1]:
                self.signals[i] = 1
            else:
                self.signals[i] = -1

            self.filteredY[i] = self.influence * self.y[i] + (1 - self.influence) * self.filteredY[i - 1]
            self.avgFilter[i] = np.mean(self.filteredY[(i - self.lag):i])
            self.stdFilter[i] = np.std(self.filteredY[(i - self.lag):i])
        else:
            self.signals[i] = 0
            self.filteredY[i] = self.y[i]
            self.avgFilter[i] = np.mean(self.filteredY[(i - self.lag):i])
            self.stdFilter[i] = np.std(self.filteredY[(i - self.lag):i])

        return self.signals[i]

    def fit(self, x):
        if x is np.ndarray:
            x = x.tolist()
        for data_point in x:
            self.thresholding_algo(data_point)
        self.fitted = True

    def predict(self):
        if not self.fitted:
            return None  # or raise error
        return self.signals


class ScipyPeaks:
    def __init__(self, shift=False, scipy_params=None):
        self.peaks = None
        self.props = None
        self.shift = shift # TODO implement shift for peaks
        self.params = scipy_params

    def predict(self, x):
        self.peaks, self.props = find_peaks(x, **self.params)
        # TODO width can be made more elaborate by adding more information about the peaks to the calculation
        width_inds = np.asarray([i for p, w in zip(self.peaks, self.props["widths"]) for i in
                                 range(np.int(p - w / 2), np.int(p + w / 2))]).ravel()
        speaks = np.zeros(len(x))
        speaks[width_inds] = 1
        return speaks


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
            "cd_message_density_smoothed": moving_avg(MinMaxScaler().fit_transform(cd_message_density.reshape(-1, 1)),
                                                      N=1500)
        }

        data_totals["video_count"] += 1
        data_totals["video_length_secs"] += len(ch_match) / 30  # total video length in seconds (30fps)
        data_totals["highlight_count"] += hl_count

    return matches_meta


if __name__ == "__main__":
    parser = ArgumentParser(description="tune, run, evaluate baseline classfiers for highlight prediction on chat"
                                        "messages")
    parser.add_argument("-p", "--data_path", help="path to the data directory")
    parser.add_argument("-b", "--baseline", choices=["rtpp", "spp"], help="which baseline to run.\n\trtpp:\treal time"
                                                                          "peak predictor\n\tspp: scipy's find peaks")
    parser.add_argument("-c", "--config_file", help="path to the config file for the baseline parameter values")

    args = parser.parse_args()

    # load data
    files = "nalcs*"  # "nalcs_w1d3_TL_FLY_g*" # "nalcs_w*d3_*g1"
    matches = load_experiments_data(files, load_random=3, random_state=42)

    if args["baseline"] == "rtpp"
        lag = 750
        for name, m in matches.items():
            rtpp = RealTimePeakPredictor(array=m["cd_message_density_smoothed"][:lag], lag=lag, threshold=3,
                                         influence=0.9)
            rtpp.fit(m["cd_message_density_smoothed"][lag:])
            m["pred_md_spikes"] = rtpp.predict()

    if args["baseline"] == "spp":
        peaks_params = {
            "height": None,
            "threshold": None,
            "distance": None,
            "prominence": [0.1],
            "width": (1000, 5000),
            "wlen": None,
            "rel_height": 0.5,
            "plateau_size": None
        }

        spp = ScipyPeaks(scipy_params=peaks_params)
        m["pred_scipy_peaks"] = spp.predict(m["cd_message_density_smoothed"])