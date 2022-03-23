import numpy as np
from scipy.signal import find_peaks


class ScipyPeaks:
    """
        Possible parameters
                {
                "height": None,
                "threshold": None,
                "distance": None,
                "prominence": [0.1],
                "width": (1000, 5000),
                "wlen": None,
                "rel_height": 0.5,
                "plateau_size": None
            }
    """
    def __init__(self, shift=None, width_scale=0.5, scipy_params=None):
        self.peaks = None
        self.props = None
        self.shift = shift
        self.width_scale = width_scale
        self.params = scipy_params

    def predict(self, x):
        self.peaks, self.props = find_peaks(x, **self.params)
        """
        width_inds = np.asarray([i for p, w in zip(self.peaks, self.props["widths"]) for i in
                                 range(max(0, int(p - w / 2)), min(len(x), int(p + w / 2)))]).ravel()
        """

        width_inds = list()
        for p, w in zip(self.peaks, self.props["widths"]):
            shift_amount = 0
            if self.shift:
                shift_amount = int(w * self.shift)
            hl_start = max(0, int(p - w/2 * self.width_scale - shift_amount)) # should be changed to + shift amount for more intuitive values
            hl_end = min(len(x), p + w/2 * self.width_scale - shift_amount)

            width_inds.extend(list(range(int(hl_start), int(hl_end))))

        speaks = np.zeros(len(x))
        # check that a prediction exists
        if len(width_inds) > 0:
            speaks[width_inds] = 1
        return speaks
