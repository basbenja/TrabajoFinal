import numpy as np

def gen_next_time_step(base_mean, phi, previous, std_error):
    return (
        (1 - phi) * base_mean + phi * previous + np.random.normal(0, 1) * std_error
    )

def gen_time_series_with_trend(
    steps,
    n_per_dep,
    mean_fixed_effects,
    std_error,
    fixed_effect_i,
    mean_temp_effects,
    phi,
    treatment_start,
    ups_max_count
):
    y = np.zeros(steps)

    base_mean = mean_fixed_effects + fixed_effect_i + mean_temp_effects
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
