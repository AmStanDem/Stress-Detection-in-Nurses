"""Module for loading, upsampling, and aligning physiological signals from Empatica E4.

This module provides functions to parse the E4 CSV format (extracting start time
and sampling frequency), upsample signals to a target frequency (e.g. 4Hz),
temporally align multiple signals using their start times and frequencies,
and construct lag features for machine learning.
"""

from pathlib import Path
import numpy as np
import pandas as pd


def load_e4_csv(file_path: Path) -> tuple[float, float, np.ndarray]:
    """Load physiological signal from an Empatica E4 CSV file.

    The Empatica E4 CSV format consists of:
    - Line 1: Start timestamp (POSIX UTC float)
    - Line 2: Sampling frequency (Hz float)
    - Line 3 onwards: Raw signal values

    Parameters
    ----------
    file_path : Path
        Path to the E4 CSV file.

    Returns
    -------
    tuple[float, float, np.ndarray]
        A tuple containing (start_time, frequency, signal_values).
    """
    with open(file_path, "r", encoding="utf-8") as f:
        start_time = float(f.readline().strip())
        frequency = float(f.readline().strip())

    signal_values = np.loadtxt(file_path, skiprows=2, delimiter=",")
    return start_time, frequency, signal_values


def align_signals(
    eda_data: tuple[float, float, np.ndarray],
    temp_data: tuple[float, float, np.ndarray],
    hr_data: tuple[float, float, np.ndarray],
    target_freq: float = 4.0,
) -> tuple[float, pd.DataFrame]:
    """Upsample and temporally align EDA, TEMP, and HR signals to a target frequency.

    The signals are upsampled by sample repetition to match the target frequency,
    then aligned to the maximum of the start times and cropped to the minimum
    of the end times.

    Parameters
    ----------
    eda_data : tuple[float, float, np.ndarray]
        (start_time, frequency, values) for Electrodermal Activity.
    temp_data : tuple[float, float, np.ndarray]
        (start_time, frequency, values) for Skin Temperature.
    hr_data : tuple[float, float, np.ndarray]
        (start_time, frequency, values) for Heart Rate.
    target_freq : float, default 4.0
        The frequency (Hz) to align the signals to.

    Returns
    -------
    tuple[float, pd.DataFrame]
        A tuple containing:
        - reference_time (float): The timestamp of the first sample in the dataframe.
        - df_aligned (pd.DataFrame): Dataframe with columns ['EDA', 'HR', 'temp'].
    """
    eda_start, eda_freq, eda_vals = eda_data
    temp_start, temp_freq, temp_vals = temp_data
    hr_start, hr_freq, hr_vals = hr_data

    repeat_eda = int(round(target_freq / eda_freq))
    eda_upsampled = np.repeat(eda_vals, repeat_eda)

    repeat_temp = int(round(target_freq / temp_freq))
    temp_upsampled = np.repeat(temp_vals, repeat_temp)

    repeat_hr = int(round(target_freq / hr_freq))
    hr_upsampled = np.repeat(hr_vals, repeat_hr)

    start_time = max(eda_start, temp_start, hr_start)

    eda_end = eda_start + (len(eda_upsampled) - 1) / target_freq
    temp_end = temp_start + (len(temp_upsampled) - 1) / target_freq
    hr_end = hr_start + (len(hr_upsampled) - 1) / target_freq
    end_time = min(eda_end, temp_end, hr_end)

    eda_idx_start = int(round((start_time - eda_start) * target_freq))
    temp_idx_start = int(round((start_time - temp_start) * target_freq))
    hr_idx_start = int(round((start_time - hr_start) * target_freq))

    num_samples = int(round((end_time - start_time) * target_freq)) + 1

    eda_aligned = eda_upsampled[eda_idx_start : eda_idx_start + num_samples]
    temp_aligned = temp_upsampled[temp_idx_start : temp_idx_start + num_samples]
    hr_aligned = hr_upsampled[hr_idx_start : hr_idx_start + num_samples]

    df_aligned = pd.DataFrame(
        {
            "EDA": eda_aligned,
            "HR": hr_aligned,
            "temp": temp_aligned,
        }
    )

    return start_time, df_aligned


def create_lag_features(
    df_features: pd.DataFrame, lag_steps: int = 10
) -> pd.DataFrame:
    """Construct lag features for HR_Mean, temp_Mean, and EDA_Mean.

    Parameters
    ----------
    df_features : pd.DataFrame
        Dataframe containing the computed window features.
    lag_steps : int, default 10
        Number of steps to shift back.

    Returns
    -------
    pd.DataFrame
        Dataframe containing the 30 lag columns (named '30' down to '1')
        aligned with df_features (first lag_steps rows will have NaNs).
    """
    cols = [str(x) for x in range(30, 0, -1)]
    lag_cols = []

    for step in range(lag_steps, 0, -1):
        lag_cols.append(df_features["HR_Mean"].shift(step))
    for step in range(lag_steps, 0, -1):
        lag_cols.append(df_features["temp_Mean"].shift(step))
    for step in range(lag_steps, 0, -1):
        lag_cols.append(df_features["EDA_Mean"].shift(step))

    df_lags = pd.concat(lag_cols, axis=1)
    df_lags.columns = cols
    return df_lags
