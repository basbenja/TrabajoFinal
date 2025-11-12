import numpy as np

def gen_next_time_step(
    base_mean: float,
    phi: float,
    previous: float,
    std_error: float
) -> float:
    """
    Generate the next value in a time series based on an autoregressive model.

    Parameters:
        - base_mean: The base mean value for the time series.
        - phi: The autoregressive coefficient.
        - previous: The previous value in the time series.
        - std_error: The standard deviation of the error term.

    Returns:
        - The next value in the time series.
    """
    return (
        (1 - phi) * base_mean + phi * previous + np.random.normal(0, 1) * std_error
    )


def gen_time_series_with_trend(
    steps: int,
    n_per_dep: int,
    treatment_start: int,
    ups_max_count: int,
    phi: float,
    mean_fixed_effects: float,
    mean_time_effects: float,
    fixed_effect_i: float,
    std_error: float
) -> np.ndarray:
    """
    Generate a time series with a trend component.
    """
    y = np.zeros(steps)

    base_mean = mean_fixed_effects + fixed_effect_i + mean_time_effects
    y[0] = base_mean + np.random.normal(0, 1) * std_error

    trend_start = treatment_start - n_per_dep
    trend_end   = treatment_start - 1

    ups_count = 0
    for t in range(1, steps):
        next_value = gen_next_time_step(base_mean, phi, y[t - 1], std_error)
        if trend_start <= t <= trend_end:
            if next_value > y[t-1] and ups_count < ups_max_count:
                ups_count += 1
            else:
                while next_value >= y[t-1]:
                    next_value = gen_next_time_step(base_mean, phi, y[t - 1], std_error)
        y[t] = next_value

    return y
