from numbers import Number


class Average:
    """Moving average."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0.0
        self.count = 0

    @property
    def value(self) -> Number:
        return self.sum / self.count

    def add(self, value: Number, n: int = 1):
        self.sum += n * value
        self.count += n


class ExponentialMovingAverage:
    """Exponential weighted moving average."""

    def __init__(self, beta: float = 0.98):
        self.beta = beta
        self.reset()

    def reset(self):
        self.count = 0
        self.biased_avg = 0.0

    @property
    def value(self) -> Number:
        # bias correction
        return self.biased_avg / (1 - self.beta ** self.count)
    
    def add(self, value: Number):
        self.count += 1
        self.biased_avg = self.beta * self.biased_avg + (1 - self.beta) * value
