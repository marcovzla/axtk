class Average:
    """
    A class for calculating a moving average.

    Attributes:
        sum (float): The sum of added values.
        count (int): The number of added values.
    """

    def __init__(self):
        """Initialize a new moving average."""
        self.reset()

    def reset(self):
        """Reset the moving average to its initial state."""
        self.total = 0.0
        self.count = 0

    @property
    def value(self) -> float:
        """
        Calculate the current moving average value.

        Returns:
            float: The moving average value.

        Raises:
            ZeroDivisionError: If there are no values added (count is 0).
        """
        if self.count == 0:
            raise ZeroDivisionError('Cannot calculate the average of an empty sequence.')
        return self.total / self.count

    def add(self, value: float, n: int = 1):
        """
        Add one or more values to the moving average.

        Args:
            value (float): The value(s) to add to the moving average.
            n (int, optional): The number of times to add the value(s).
        """
        self.total += n * value
        self.count += n


class ExponentialMovingAverage:
    """
    A class for calculating an Exponential Moving Average (EMA).

    Attributes:
        beta (float): The smoothing factor, typically between 0 and 1.
        count (int): The number of values added.
        biased_avg (float): The current EMA value.
    """

    def __init__(self, beta: float = 0.98):
        """
        Initialize a new Exponential Moving Average.

        Args:
            beta (float, optional): The smoothing factor (default is 0.98).
        """
        self.beta = beta
        self.reset()

    def reset(self):
        """Reset the Exponential Moving Average to its initial state."""
        self.count = 0
        self.biased_avg = 0.0

    @property
    def value(self) -> float:
        """
        Calculate the current Exponential Moving Average value with bias correction.

        Returns:
            float: The Exponential Moving Average value.

        Raises:
            ZeroDivisionError: If there are no values added (count is 0).
        """
        if self.count == 0:
            raise ZeroDivisionError("Cannot calculate the EMA of an empty sequence.")
        return self.biased_avg / (1 - self.beta ** self.count)

    def add(self, value: float):
        """
        Add a value to the Exponential Moving Average.

        Args:
            value (float): The value to add to the EMA.
        """
        self.count += 1
        self.biased_avg = self.beta * self.biased_avg + (1 - self.beta) * value
