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
    def __init__(self, shift=None, width_scale=0.5, scipy_params=None, prominence=None, width=None, rel_height=None):
        self.peaks = None
        self.props = None
        self.width_scale = width_scale
        self.params = None
        self.shift = None
        self.scipy_params = None
        self.set_params(shift=shift,
                        scipy_params=scipy_params,
                        prominence=prominence,
                        width=width)

    def set_params(self, shift=None, scipy_params=None, prominence=None, width=None, rel_height=None):
        if scipy_params:
            self.params = scipy_params
        else:
            self.params = {
                "prominence": prominence,
                "width": width,
                "rel_height": rel_height
            }
        self.shift = shift

    def fit(self, x, y, **kwargs):
        return self

    def predict(self, x):
        try:
            shape = x.shape
            x = np.ravel(x)
        except AttributeError:
            shape = (len(x),)
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

        print(speaks.shape, shape)
        if shape == speaks.shape:
            return speaks
        else:
            return speaks.reshape(shape)
