import numpy as np


class RealTimePeakPredictor:
    """
    theory:
    https://stackoverflow.com/a/22640362

    implementation
    https://stackoverflow.com/a/56451135
    Spike detection algorithm
    """
    def __init__(self, array=None, lag=0, threshold=1, influence=0):
        """

        :param array: prime algorithm with some data points
        :param lag: minimum number of datapoints with which to prime alorithm, determines how much your data will be
        smoothed and how adaptive the algorithm is to changes in the long-term average of the data
        :param threshold: number of standard deviations from the moving mean above which the algorithm will classify a
        new datapoint as being a signal
        :param influence: determines the influence of signals on the algorithm's detection threshold
        """

        self.stdFilter = None
        self.avgFilter = None
        self.filteredY = None
        self.signals = None
        self.length = None
        self.y = None

        self.lag = None
        self.threshold = None
        self.influence = None

        self.set_params(lag, threshold, influence)

        if array:
            self.prime_algorithm(array)

    def prime_algorithm(self, array):
        self.y = list(array)
        self.length = len(self.y)
        self.signals = [0] * len(self.y)
        self.filteredY = np.array(self.y).tolist()
        self.avgFilter = [0] * len(self.y)
        self.stdFilter = [0] * len(self.y)
        self.avgFilter[self.lag - 1] = np.mean(self.y[0:self.lag])
        self.stdFilter[self.lag - 1] = np.std(self.y[0:self.lag])


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

    def fit(self, x, y, **kwargs):
        return self

    def predict(self, x):
        shape = None
        if type(x) is np.ndarray:
            shape = x.shape
            x = x.ravel().tolist()

        primer = x[:self.lag]
        x = x[self.lag:]
        self.prime_algorithm(primer)

        for data_point in x:
            self.thresholding_algo(data_point)

        ret = [abs(d) for d in self.signals]
        if shape:
            ret = np.asarray(ret).reshape(shape)
        return ret

    def set_params(self, lag, threshold, influence):
        self.lag = lag
        self.threshold = threshold
        self.influence = influence
