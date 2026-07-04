"""Main entrypoint for the stress detection model pipeline.

This script runs the entire pipeline:
1. Loads the combined lag training dataset.
2. Trains RandomForest, KNN, and SVM models using chronological split validation.
3. Loads raw Electrodermal Activity (EDA), Skin Temperature (TEMP), and Heart Rate (HR)
   data for subject 'DF'.
4. Aligns the data temporally and upsamples to 4Hz.
5. Computes windowed features and generates lag columns.
6. Scales features and predicts stress levels.
7. Saves the outputs to predDF.csv and predSVMDF.csv.
8. Runs change point detection and logs the stress events chronologically.
"""

from pathlib import Path
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from src.data_processing import align_signals, create_lag_features, load_e4_csv
from src.features import extract_window_features
from src.models import train_and_evaluate_all
from src.change_point import classify_and_merge_segments, detect_change_points


def main() -> None:
    """Run the physiological signal preprocessing and stress prediction pipeline.

    Loads the datasets, trains the models, extracts features for subject 'DF',
    runs prediction, and outputs stress segmentation intervals.
    """
    base_dir = Path(".")
    train_data_path = base_dir / "combined_lagEDA.csv"

    print("Loading combined lag features training dataset...")
    df_lag = pd.read_csv(train_data_path)
    X_train_full = df_lag.iloc[:, 0:48]
    y_train_full = df_lag.iloc[:, 48:49]

    print("Training models...")
    trained_models = train_and_evaluate_all(X_train_full, y_train_full)

    subject = "DF"
    subject_dir = base_dir / subject

    print(f"Loading raw physiological E4 signals for subject '{subject}'...")
    eda_data = load_e4_csv(subject_dir / "EDA.csv")
    temp_data = load_e4_csv(subject_dir / "TEMP.csv")
    hr_data = load_e4_csv(subject_dir / "HR.csv")

    print("Upsampling and temporally aligning signals to 4Hz...")
    ref_time, df_aligned = align_signals(eda_data, temp_data, hr_data, target_freq=4.0)

    print("Extracting windowed features...")
    df_features = extract_window_features(df_aligned, window_size=40, step=20)

    print("Generating lag features...")
    df_lag_features = create_lag_features(df_features, lag_steps=10)

    print("Concatenating features and dropping NaN rows...")
    df_total = pd.concat([df_lag_features, df_features], axis=1)
    df_total = df_total.dropna()

    print("Scaling features using MinMaxScaler...")
    scaler = MinMaxScaler()
    x_scaled = scaler.fit_transform(df_total.iloc[:, 0:48])

    df_scaled = pd.DataFrame(x_scaled, columns=df_total.columns[:48])
    df_scaled = df_scaled.fillna(0.0)

    feature_names_mapping = {
        "EDA_Mean": "EDAR_Mean",
        "EDA_Min": "EDAR_Min",
        "EDA_Max": "EDAR_Max",
        "EDA_Std": "EDAR_Std",
        "EDA_Kurtosis": "EDAR_Kurtosis",
        "EDA_Skew": "EDAR_Skew",
        "EDA_Num_Peaks": "Num_PeaksR",
        "EDA_Amphitude": "EDAR_Amphitude",
        "EDA_Duration": "EDAR_Duration",
        "HR_Mean": "HRR_Mean",
        "HR_Min": "HRR_Min",
        "HR_Max": "HRR_Max",
        "HR_Std": "HRR_Std",
        "HR_RMS": "HRR_RMS",
        "temp_Mean": "TEMPR_Mean",
        "temp_Min": "TEMPR_Min",
        "temp_Max": "TEMPR_Max",
        "temp_Std": "TEMPR_Std",
    }
    df_scaled = df_scaled.rename(columns=feature_names_mapping)
    df_scaled = df_scaled[X_train_full.columns]

    print("Predicting stress levels using RandomForestClassifier...")
    rf_model = trained_models["RF"]
    rf_preds = rf_model.predict(df_scaled)

    df_total_rf = df_total.copy()
    df_total_rf["pred"] = rf_preds
    rf_out_path = base_dir / f"pred{subject}.csv"
    df_total_rf.to_csv(rf_out_path, index=False)
    print(f"RandomForest predictions saved to {rf_out_path.name}")

    print("Predicting stress levels using Support Vector Classifier (SVM)...")
    svm_model = trained_models["SVM"]
    svm_preds = svm_model.predict(df_scaled)

    df_total_svm = df_total.copy()
    df_total_svm["pred"] = svm_preds
    svm_out_path = base_dir / f"predSVM{subject}.csv"
    df_total_svm.to_csv(svm_out_path, index=False)
    print(f"SVM predictions saved to {svm_out_path.name}")

    print("Running change point detection on RandomForest predictions...")
    change_points = detect_change_points(df_total_rf["pred"], penalty=10.0)
    print(f"Detected change point indices: {change_points}")

    print("Classifying and merging stress segments...")
    df_events = classify_and_merge_segments(
        df_total_rf, change_points, reference_time=ref_time, window_step_seconds=5.0
    )

    print("\n=== Detected Stress Event Intervals ===")
    for _, row in df_events.iterrows():
        print(
            f"Interval: {row['start_time']} - {row['end_time']} | "
            f"Duration: {row['duration']} | "
            f"Stress Level: {row['stress_level']}"
        )


if __name__ == "__main__":
    main()
