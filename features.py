"""Module for extracting physiological features from aligned E4 signals.

This module computes statistical, shape, RMSSD, and peak detection features over
rolling time windows for Electrodermal Activity (EDA), Skin Temperature (TEMP),
and Heart Rate (HR).
"""

import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.stats import kurtosis, skew


def statistical_features(arr: np.ndarray) -> tuple[float, float, float, float]:
    """Compute descriptive statistics for a 1D numeric array.

    Parameters
    ----------
    arr : np.ndarray
        One-dimensional array of numeric values.

    Returns
    -------
    tuple[float, float, float, float]
        Minimum, maximum, mean, and standard deviation of the array.
    """
    return float(np.amin(arr)), float(np.amax(arr)), float(np.mean(arr)), float(np.std(arr))


def shape_features(arr: np.ndarray) -> tuple[float, float]:
    """Compute skewness and kurtosis for a 1D numeric array.

    Parameters
    ----------
    arr : np.ndarray
        One-dimensional array of numeric values.

    Returns
    -------
    tuple[float, float]
        Skewness and kurtosis of the array.
    """
    return float(skew(arr)), float(kurtosis(arr))


def compute_rmssd(arr: np.ndarray) -> float:
    """Compute Root Mean Square of Successive Differences (RMSSD) for a 1D array.

    Parameters
    ----------
    arr : np.ndarray
        One-dimensional array of numeric values.

    Returns
    -------
    float
        RMSSD of the array.
    """
    diffs = np.ediff1d(arr)
    if len(diffs) == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(diffs))))


def compute_eda_peaks(arr: np.ndarray, min_width: float = 5.0) -> tuple[int, float, float]:
    """Detect peaks in Electrodermal Activity (EDA) signal.

    Finds peaks using scipy's find_peaks and computes the number of peaks,
    the sum of peak prominences, and the sum of peak widths.

    Parameters
    ----------
    arr : np.ndarray
        One-dimensional array of EDA values.
    min_width : float, default 5.0
        Minimum width threshold for peak detection.

    Returns
    -------
    tuple[int, float, float]
        A tuple containing (num_peaks, sum_prominences, sum_widths).
    """
    peaks, properties = find_peaks(arr, width=min_width)
    num_peaks = len(peaks)
    prominences = np.array(properties.get("prominences", [0.0]))
    widths = np.array(properties.get("widths", [0.0]))
    amplitude = float(np.sum(prominences))
    duration = float(np.sum(widths))
    return num_peaks, amplitude, duration


def extract_window_features(
    df_aligned: pd.DataFrame, window_size: int = 40, step: int = 20
) -> pd.DataFrame:
    """Extract physiological features over rolling windows.

    Parameters
    ----------
    df_aligned : pd.DataFrame
        Dataframe with columns ['EDA', 'HR', 'temp'].
    window_size : int, default 40
        Size of the rolling window in samples (e.g. 40 samples at 4Hz = 10s).
    step : int, default 20
        Step size between windows in samples (e.g. 20 samples = 50% overlap).

    Returns
    -------
    pd.DataFrame
        Dataframe containing 18 extracted features for each window.
    """
    cols = [
        "EDA_Mean", "EDA_Min", "EDA_Max", "EDA_Std", "EDA_Kurtosis", "EDA_Skew",
        "EDA_Num_Peaks", "EDA_Amphitude", "EDA_Duration",
        "HR_Mean", "HR_Min", "HR_Max", "HR_Std", "HR_RMS",
        "temp_Mean", "temp_Min", "temp_Max", "temp_Std"
    ]

    features_list = []
    eda_vals = df_aligned["EDA"].values
    hr_vals = df_aligned["HR"].values
    temp_vals = df_aligned["temp"].values

    for i in range(0, len(df_aligned), step):
        if i + window_size > len(df_aligned):
            break

        eda_win = eda_vals[i : i + window_size]
        hr_win = hr_vals[i : i + window_size]
        temp_win = temp_vals[i : i + window_size]

        eda_min, eda_max, eda_mean, eda_std = statistical_features(eda_win)
        hr_min, hr_max, hr_mean, hr_std = statistical_features(hr_win)
        temp_min, temp_max, temp_mean, temp_std = statistical_features(temp_win)

        eda_skew, eda_kurt = shape_features(eda_win)
        hr_rms = compute_rmssd(hr_win)
        num_peaks, amplitude, duration = compute_eda_peaks(eda_win)

        features_list.append([
            eda_mean, eda_min, eda_max, eda_std, eda_kurt, eda_skew,
            num_peaks, amplitude, duration,
            hr_mean, hr_min, hr_max, hr_std, hr_rms,
            temp_mean, temp_min, temp_max, temp_std
        ])

    return pd.DataFrame(features_list, columns=cols)
