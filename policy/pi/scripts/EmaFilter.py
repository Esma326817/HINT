"""
Exponential Moving Average (EMA) Filter.

Used to smooth trajectories from input controllers (VR, keyboard, etc.)
to reduce jitter and small-range noise before sending commands to robot.

EMA formula:  output = alpha * input + (1 - alpha) * last_output
  - alpha close to 1.0 → less smoothing, more responsive
  - alpha close to 0.0 → more smoothing, more latency
"""

import numpy as np


class EmaFilter:
    """
    General-purpose EMA filter for scalar or vector inputs.

    Parameters
    ----------
    alpha : float
        Smoothing factor in (0, 1]. Higher = less smoothing.
    """

    def __init__(self, alpha: float):
        if not 0 < alpha <= 1:
            raise ValueError(f"alpha must be in (0, 1], got {alpha}")
        self._alpha = alpha
        self._last_output: np.ndarray | None = None

    def run(self, input_val) -> np.ndarray:
        """
        Apply one step of EMA filtering.

        Parameters
        ----------
        input_val : float, list, or np.ndarray
            Current input value (scalar or vector).

        Returns
        -------
        np.ndarray
            Filtered output.
        """
        x = np.asarray(input_val, dtype=np.float64)
        if self._last_output is None:
            self._last_output = x.copy()
        else:
            self._last_output = self._alpha * x + (1.0 - self._alpha) * self._last_output
        return self._last_output.copy()

    def reset(self, value=None):
        """
        Reset filter state.

        Parameters
        ----------
        value : optional
            If provided, set the filter state to this value instead of None.
        """
        if value is not None:
            self._last_output = np.asarray(value, dtype=np.float64)
        else:
            self._last_output = None

    @property
    def alpha(self) -> float:
        return self._alpha

    @alpha.setter
    def alpha(self, value: float):
        if not 0 < value <= 1:
            raise ValueError(f"alpha must be in (0, 1], got {value}")
        self._alpha = value


class QuaternionEmaFilter:
    """
    EMA filter specialized for quaternions.

    Handles quaternion double-cover (q ≡ -q) by flipping the sign
    when the dot product with the previous output is negative, then
    normalizes the result to keep it on the unit sphere.

    Parameters
    ----------
    alpha : float
        Smoothing factor in (0, 1]. Higher = less smoothing.
    """

    def __init__(self, alpha: float):
        if not 0 < alpha <= 1:
            raise ValueError(f"alpha must be in (0, 1], got {alpha}")
        self._alpha = alpha
        self._last_output: np.ndarray | None = None

    def run(self, quat) -> np.ndarray:
        """
        Apply one step of quaternion EMA filtering.

        Parameters
        ----------
        quat : array-like, shape (4,)
            Input quaternion (any convention, e.g. WXYZ or XYZW).

        Returns
        -------
        np.ndarray, shape (4,)
            Filtered and normalized quaternion.
        """
        q = np.asarray(quat, dtype=np.float64)
        if self._last_output is None:
            self._last_output = q / np.linalg.norm(q)
        else:
            # Handle quaternion double-cover: q and -q represent the same rotation.
            # Choose the sign that is closer to the previous output.
            if np.dot(self._last_output, q) < 0:
                q = -q
            self._last_output = self._alpha * q + (1.0 - self._alpha) * self._last_output
            # Re-normalize to unit quaternion
            self._last_output = self._last_output / np.linalg.norm(self._last_output)
        return self._last_output.copy()

    def reset(self, value=None):
        """Reset filter state."""
        if value is not None:
            q = np.asarray(value, dtype=np.float64)
            self._last_output = q / np.linalg.norm(q)
        else:
            self._last_output = None

    @property
    def alpha(self) -> float:
        return self._alpha

    @alpha.setter
    def alpha(self, value: float):
        if not 0 < value <= 1:
            raise ValueError(f"alpha must be in (0, 1], got {value}")
        self._alpha = value