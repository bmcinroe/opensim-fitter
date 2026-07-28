import numpy as np

class Bounds:
    lower_bound: float = -np.inf
    upper_bound: float = np.inf

    def __init__(self, lower_bound: float, upper_bound: float):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    @classmethod
    def as_default(cls):
        return cls(lower_bound=-np.inf, upper_bound=np.inf)

    @classmethod
    def as_equality(cls, value: float):
        return cls(lower_bound=value, upper_bound=value)
