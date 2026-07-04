"""Module for change point detection and stress event segmentation.

This module uses the Pelt search algorithm from the ruptures library to identify
state change points in the predicted stress time series. It classifies each segment
using segment-wide average stress levels and formats the start and end of these
intervals as human-readable timestamps.
"""

from datetime import datetime
import pandas as pd
import ruptures as rpt


def detect_change_points(predictions: pd.Series, penalty: float = 10.0) -> list[int]:
    """Run Pelt change point detection on predicted stress labels.

    Parameters
    ----------
    predictions : pd.Series
        Series of predicted stress labels.
    penalty : float, default 10.0
        The penalty parameter for Pelt search.

    Returns
    -------
    list[int]
        List of change point indices, starting with 0.
    """
    signal = predictions.values.reshape(-1)
    algo = rpt.Pelt(model="l2").fit(signal)
    result = algo.predict(pen=penalty)
    return [0] + result


def classify_and_merge_segments(
    df_total: pd.DataFrame,
    change_points: list[int],
    reference_time: float,
    window_step_seconds: float = 5.0,
) -> pd.DataFrame:
    """Classify and merge segments of stress labels using segment-wide averages.

    Parameters
    ----------
    df_total : pd.DataFrame
        Dataframe containing aligned features and the 'pred' column.
    change_points : list[int]
        List of change point indices.
    reference_time : float
        POSIX timestamp corresponding to the start of the aligned time series.
    window_step_seconds : float, default 5.0
        Seconds per step (e.g. 20 samples at 4Hz = 5s).

    Returns
    -------
    pd.DataFrame
        Dataframe with merged stress intervals: ['start_time', 'end_time', 'duration', 'stress_level'].
    """
    stress_labels = []

    for i in range(len(change_points) - 1):
        segment = df_total.iloc[change_points[i] : (change_points[i + 1] - 1)]
        if not segment.empty:
            mean_stress = segment["pred"].mean()
        else:
            mean_stress = 0.0

        if mean_stress > 1.3:
            level = 2
        elif mean_stress >= 0.65:
            level = 1
        else:
            level = 0

        stress_labels.append(level)

    temp_intervals = []
    for i in range(len(change_points) - 1):
        temp_intervals.append(
            {
                "start": change_points[i],
                "end": change_points[i + 1],
                "stress": stress_labels[i],
            }
        )

    merged_intervals = []
    if not temp_intervals:
        return pd.DataFrame(columns=["start_time", "end_time", "duration", "stress_level"])

    stress_start = temp_intervals[0]["start"]
    stress_end = temp_intervals[0]["end"]
    previous_stress = temp_intervals[0]["stress"]

    for item in temp_intervals[1:]:
        if item["stress"] == previous_stress:
            stress_end = item["end"]
        else:
            start_dt = datetime.fromtimestamp(reference_time + (stress_start * window_step_seconds))
            end_dt = datetime.fromtimestamp(reference_time + (stress_end * window_step_seconds))
            duration = end_dt - start_dt

            merged_intervals.append(
                {
                    "start_time": start_dt.strftime("%H:%M:%S"),
                    "end_time": end_dt.strftime("%H:%M:%S"),
                    "duration": str(duration),
                    "stress_level": float(previous_stress),
                }
            )
            stress_start = item["start"]
            stress_end = item["end"]
            previous_stress = item["stress"]

    start_dt = datetime.fromtimestamp(reference_time + (stress_start * window_step_seconds))
    end_dt = datetime.fromtimestamp(reference_time + (stress_end * window_step_seconds))
    duration = end_dt - start_dt

    merged_intervals.append(
        {
            "start_time": start_dt.strftime("%H:%M:%S"),
            "end_time": end_dt.strftime("%H:%M:%S"),
            "duration": str(duration),
            "stress_level": float(previous_stress),
        }
    )

    return pd.DataFrame(merged_intervals)
