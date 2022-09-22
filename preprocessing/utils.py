import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler


"""HIGHLIGHT STATISTICS"""
def highlight_count(hl):
    prev = -1
    hlc = 0
    for frame in hl:
        if frame == 1 and prev < 1:
            hlc += 1
        prev = frame
    return hlc


def highlight_length(hl):
    hll = list()

    prev = -1
    for frame in hl:
        if frame == 1:
            if prev < 1:
                hll.append(1)
            else:
                hll[-1] += 1
        prev = frame
    return hll


def highlight_span(hl):
    """
    extracts beginning and end indices for each highlight in hl
    :param hl: iterable of 0 for non-hihglight frame and 1 for highlight frame
    :return: tuples of indices over all frames in hl where highlights are found
    """
    hls = list()

    prev = -1
    for i, frame in enumerate(hl):
        if frame == 1 and prev < 1:
            hls.append((i, -1))
        if frame == 0 and prev == 1:
            hls[-1] = (hls[-1][0], i-1)
        prev = frame

    if len(hls) > 0 and hls[-1][1] < 0:
        # highlight goes until the end
        hls[-1] = (hls[-1][0], hl.shape[0]-1)
    return hls


def msgs_hl_non_hl(cd, hl):
    cd = np.asarray(cd)

    msgs_hl = cd[np.where(hl == 1)]
    msgs_non_hl = cd[np.where(hl == 0)]

    return unpack_messages(msgs_hl), unpack_messages(msgs_non_hl)


def moving_avg(mylist, N=5):
    try:
        shape = mylist.shape
        mylist = np.ravel(mylist)
    except AttributeError:
        shape = (1, -1)

    cumsum, moving_aves = [0], []

    for i, x in enumerate(mylist, 1):
        cumsum.append(cumsum[i-1] + x)
        if i>=N:
            moving_ave = (cumsum[i] - cumsum[i-N])/N
            #can do stuff with moving_ave here
            moving_aves.append(moving_ave)
    diff = len(mylist) - len(moving_aves)
    if diff > 0:
        tmp = [0 for i in range(diff)]
        tmp.extend(moving_aves)
        moving_aves = tmp

    return np.asarray(moving_aves).reshape(shape)


def unpack_messages(cd):
    """
    :param cd: iterable of strings with chat messages, individual messages separated by '\n'
    :return: unpacked messages in iterable, one string per message, no empty strings returned
    """
    unpacked = []
    for m in cd:
        ms = m.split("\n")
        unpacked.extend([m1 for m1 in ms if len(m1) > 0])
    return unpacked


def plot_matches(matches):
    fig, axs = plt.subplots(len(matches.keys()), sharex="all")
    for i, k1 in enumerate(matches.keys()): # fails if only one match is selected
        ax = axs[i]
        ax.title.set_text(k1)
        for k2 in matches[k1].keys():
            if k2.startswith("chat"):
                dat = matches[k1][k2]
                # flip moving average and MinMaxScaler
                ax.plot(np.arange(len(dat)), moving_avg(MinMaxScaler().fit_transform(dat.reshape(-1, 1)), N=1500), linewidth=.5, label=k2)
                # ax.plot(np.arange(len(dat)), MinMaxScaler().fit_transform(dat.reshape(-1, 1)), linewidth=.5, label=f"{k2} no smoothing")
            if k2 == "highlights" or k2.startswith("pred"):
                dat = matches[k1][k2]
                ax.plot(np.arange(len(dat)), dat, linewidth=.5, label=k2)

    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper left')
    plt.show()