import numpy as np

import analysis
from scipy.interpolate import krogh_interpolate
import matplotlib.pyplot as plt


def interpolate(series):
    # assuming 1D series
    steps = range(len(series))
    f = krogh_interpolate(steps, series, steps[::5], der=2)
    return f


if __name__ == "__main__":
    file_regex = "nalcs*"  # "nalcs_w1d3_TL_FLY_g*" # "nalcs_w*d3_*g1"
    chat = analysis.load_chat("data/final_data", file_identifier=file_regex, load_random=1, random_state=42)
    highlights = analysis.load_highlights("data/gt", file_identifier=file_regex)  # nalcs_w1d3_TL_FLY_g2
    analysis.remove_missing_matches(chat, highlights)

    cut = 30 * 10  # 5, 10 sec intervals in 30 fps video, why? just because!

    density_data = list()
    matches = list()
    interpolations = list()
    hl_spans = list()

    for match in chat.keys():
        ch_match, hl_match = analysis.cut_same_length(chat[match], highlights[match])
        chat[match] = ch_match
        highlights[match] = hl_match
        matches.append(match)

        # examine highlight regions only
        highlight_spans = analysis.highlight_span(hl_match)
        print(highlight_spans)
        # examine 20% of highlight length before and after
        highlight_spans_p20 = [(int(b-(e-b)*0.2), int(e+(e-b)*0.2)) for b, e in highlight_spans]
        print(highlight_spans_p20)
        hl_spans.append(highlight_spans_p20)

        cd_message_density_smooth = analysis.moving_avg(analysis.message_density(ch_match, interval=cut))
        density_data.append(cd_message_density_smooth)
        # interpolations.append(interpolate(cd_message_density_smooth))
    # flatten
    hls = np.ravel(hl_spans)

    factor = len(density_data[0])/len(highlights[matches[0]])

    time_steps = range(len(density_data[0]))
    print(factor)
    plt.plot(time_steps, density_data[0])
    plt.plot(highlights[matches[0]][::actual_cut])
    plt.scatter(hls*factor, np.ones(hls.shape[0]))
    plt.show()
