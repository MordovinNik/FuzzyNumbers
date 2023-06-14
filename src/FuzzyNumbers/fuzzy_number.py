import numpy as np
from scipy.interpolate import interp1d

ARRAY_TYPE = (list, np.ndarray)
INTERPOLATION_KINDS = ("linear", "nearest", "quadratic", "cubic", "spline")
BASE_TYPE = (int, float, np.int_, np.float_)


class FuzzyNuber:
    def __init__(self, peak=None, left=None, right=None, offset: float = 1., levels_count: int = 2):
        assert levels_count >= 2, f"levels_count ({levels_count}) must be larger that or equal to 2"

        self._levels_count = levels_count

        if isinstance(left, ARRAY_TYPE):
            self._levels_count = max(self._levels_count, len(left))
        if isinstance(right, ARRAY_TYPE):
            self._levels_count = max(self._levels_count, len(right))

        self._left = np.full(self._levels_count, np.nan)
        self._right = np.full(self._levels_count, np.nan)
        self._func_left = None
        self._func_right = None
        self._interpolation_kind = "linear"

        self._alphas = np.linspace(0, 1, self._levels_count)

        if peak is None and (left is None or isinstance(left, BASE_TYPE)) and (
                right is None or isinstance(right, BASE_TYPE)):
            raise ValueError("no peak value provided")

        if isinstance(peak, ARRAY_TYPE) and len(peak) == 2:
            self._left[-1] = peak[0]
            self._right[-1] = peak[1]
        elif isinstance(peak, BASE_TYPE):
            self._left[-1] = self._right[-1] = peak

        if callable(left):
            for i in range(self._levels_count):
                self._left[i] = left(self._alphas[i])
        elif isinstance(left, ARRAY_TYPE) and len(left) == self._levels_count:
            self._left = np.array(left)

        if callable(right):
            for i in range(self._levels_count):
                self._right[i] = right(self._alphas[i])
        elif isinstance(right, ARRAY_TYPE) and len(right) == self._levels_count:
            self._right = np.array(right)

        if np.isnan(self._left[-1]):
            if np.isnan(self._right[-1]):
                raise ValueError("no peak value provided")
            else:
                self._left[-1] = self._right[-1]
        elif np.isnan(self._right[-1]):
            self._right[-1] = self._left[-1]

        if isinstance(left, BASE_TYPE):
            self._left = left + self._alphas * (self._left[-1] - left)
        elif left is None:
            self._left = self._left[-1] - offset + offset * self._alphas

        if isinstance(right, BASE_TYPE):
            self._right = right + self._alphas * (self._right[-1] - right)
        if right is None:
            self._right = self._right[-1] + offset - offset * self._alphas

        if np.isnan(self._left).any() or np.isnan(self._right).any() or not np.all(
                np.diff(self._left) >= 0) or not np.all(np.diff(self._right) <= 0) or np.any(
            self._left > self._right):
            raise ValueError("incorrect data")

    def _build_interpolation_functions(self):
        if self._func_left is None or self._func_right is None:
            alphas = np.linspace(0, 1, self._levels_count, True)
            self._func_left = interp1d(alphas, self._left, kind=self._interpolation_kind)
            self._func_right = interp1d(alphas, self._right, kind=self._interpolation_kind)

    def alpha_level(self, alpha: float):
        assert np.all(0 <= alpha) and np.all(alpha <= 1), "alpha must be in [0,1]"
        self._build_interpolation_functions()

        if isinstance(alpha, np.ndarray):
            return np.array(list(zip(self._func_left(alpha), self._func_right(alpha))))
        else:
            return np.array([self._func_left(alpha), self._func_right(alpha)])

    def graph_data(self, n=None):
        if n is None:
            x_out = np.concatenate((self._left, np.flip(self._right)))
            y_out = np.concatenate((self._alphas, np.flip(self._alphas)))
        else:
            self._build_interpolation_functions()
            x = np.linspace(0, 1, n, True)
            x_out = np.concatenate((self._func_left(x), self._func_right(np.flip(x))))
            y_out = np.concatenate((x, np.flip(x)))

        return x_out, y_out

    @property
    def left(self):
        return self._left

    @property
    def right(self):
        return self._right

    @property
    def inf(self):
        return self._left[0]

    @property
    def sup(self):
        return self._right[0]

    @property
    def support(self):
        return self.alpha_level(0)

    @property
    def kernel(self):
        return self.alpha_level(1)

    @property
    def copy(self):
        return type(self)(left=self._left, right=self._right)

    @property
    def wid(self):
        return self._right[0] - self._left[0]

    @property
    def mid(self):
        return 0.5 * (self._left[0] + self._right[0])

    @property
    def interpolation_kind(self):
        return self._interpolation_kind

    @interpolation_kind.setter
    def interpolation_kind(self, value: str):
        if value in INTERPOLATION_KINDS:
            self._interpolation_kind = value
            self._func_left = self._func_right = None
        else:
            raise ValueError(f"interpolation_kind ({value}) must be in {INTERPOLATION_KINDS}")

    @property
    def levels_count(self):
        return self._levels_count

    @levels_count.setter
    def levels_count(self, levels_count: int):
        if levels_count >= 2:
            self._left, self._right = self.get_borders(levels_count)
            self._func_left = self._func_right = None
        else:
            raise ValueError(f"levels_count ({levels_count}) must be larger that or equal to 2")

    def get_borders(self, levels_count):
        if levels_count == self._levels_count:
            return self._left, self._right
        else:
            self._build_interpolation_functions()
            alphas = np.linspace(0, 1, levels_count, True)
            return self._func_left(alphas), self._func_right(alphas)

    def __repr__(self):
        if self._left[-1] == self._right[-1]:
            return f"({self._left[-1]}, {self._left[-1] - self.inf} , {self.sup - self._left[-1]})"
        return f"({self.kernel}, {self.kernel - self.inf} , {self.sup - self.kernel})"

    def __add__(self, other):
        if isinstance(other, BASE_TYPE):
            return FuzzyNuber(left=self._left + other, right=self._right + other)
        max_levels = max(self.levels_count, other.levels_count)
        self_left, self_right = self.get_borders(max_levels)
        other_left, other_right = other.get_borders(max_levels)
        return FuzzyNuber(left=self_left + other_left, right=self_right + other_right)

    def __sub__(self, other):
        if isinstance(other, BASE_TYPE):
            return FuzzyNuber(left=self._left - other, right=self._right - other)
        max_levels = max(self.levels_count, other.levels_count)
        self_left, self_right = self.get_borders(max_levels)
        other_left, other_right = other.get_borders(max_levels)
        return FuzzyNuber(left=self_left - other_right, right=self_right - other_left)

    def __mul__(self, other):
        if isinstance(other, BASE_TYPE):
            mult = [self._left * other,
                    self._right * other]
        else:
            max_levels = max(self.levels_count, other.levels_count)
            self_left, self_right = self.get_borders(max_levels)
            other_left, other_right = other.get_borders(max_levels)
            mult = [self_left * other_left,
                    self_left * other_right,
                    self_right * other_left,
                    self_right * other_right]
        return FuzzyNuber(left=np.minimum.reduce(mult), right=np.maximum.reduce(mult))

    def __truediv__(self, other):
        if isinstance(other, BASE_TYPE):
            assert other != 0, "dividing by zero"
            div = [self._left / other,
                   self._right / other]
        else:
            assert other._left[0] > 0 or other._right[0] < 0, "dividing by zero-containing fuzzy number"
            max_levels = max(self.levels_count, other.levels_count)
            self_left, self_right = self.get_borders(max_levels)
            other_left, other_right = other.get_borders(max_levels)
            div = [self_left / other_left,
                   self_left / other_right,
                   self_right / other_left,
                   self_right / other_right]
        return FuzzyNuber(left=np.minimum.reduce(div), right=np.maximum.reduce(div))

    def __radd__(self, other):
        if isinstance(other, BASE_TYPE):
            return FuzzyNuber(left=self._left + other, right=self._right + other)
        max_levels = max(self.levels_count, other.levels_count)
        self_left, self_right = self.get_borders(max_levels)
        other_left, other_right = other.get_borders(max_levels)
        return FuzzyNuber(left=self_left + other_left, right=self_right + other_right)

    def __rsub__(self, other):
        if isinstance(other, BASE_TYPE):
            return FuzzyNuber(left=other - self._right, right=other - self._left)
        max_levels = max(self.levels_count, other.levels_count)
        self_left, self_right = self.get_borders(max_levels)
        other_left, other_right = other.get_borders(max_levels)
        return FuzzyNuber(left=other_left - self_right, right=other_right - self_left)

    def __rmul__(self, other):
        if isinstance(other, BASE_TYPE):
            mult = [self._left * other,
                    self._right * other]
        else:
            max_levels = max(self.levels_count, other.levels_count)
            self_left, self_right = self.get_borders(max_levels)
            other_left, other_right = other.get_borders(max_levels)
            mult = [self_left * other_left,
                    self_left * other_right,
                    self_right * other_left,
                    self_right * other_right]
        return FuzzyNuber(left=np.minimum.reduce(mult), right=np.maximum.reduce(mult))

    def __rtruediv__(self, other):
        assert self._left[0] > 0 or self._right[0] < 0, "dividing by zero-containing fuzzy number"
        if isinstance(other, BASE_TYPE):
            div = [other / self._left,
                   other / self._right]
        else:
            max_levels = max(self.levels_count, other.levels_count)
            self_left, self_right = self.get_borders(max_levels)
            other_left, other_right = other.get_borders(max_levels)
            div = [other_left / self_left,
                   other_right / self_left,
                   other_left / self_right,
                   other_right / self_right]
        return FuzzyNuber(left=np.minimum.reduce(div), right=np.maximum.reduce(div))
