# config.py
# This file contains configuration (target variables, paths, etc), also included the size of test/val splits and random seed for reproducibility

from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class Paths:
    root: Path = Path(__file__).resolve().parents[1]
    data_raw: Path = root / "data" / "raw"
    data_processed: Path = root / "data" / "processed"
    outputs: Path = root / "outputs"
    models: Path = outputs / "models"
    metrics: Path = outputs / "metrics"
    figures: Path = outputs / "figures"

TARGETS = ["Addiction Level", "ProductivityLoss"]
ID_COLS = ["UserID"]  # exclude from features

# Columns that are often ID-like / high-cardinality.
# set INCLUDE_VIDEO_ID=False to drop it from baseline.
VIDEO_ID_COL = "Video ID"
INCLUDE_VIDEO_ID = False

# Feature selection mode:
# False -> use all valid columns except EXCLUDE_FEATURE_COLS
# True  -> use only INCLUDE_FEATURE_COLS
USE_INCLUDE_FEATURE_COLS = False
INCLUDE_FEATURE_COLS = [
    "Income", "Debt", "Owns Property", "Profession", "Demographics"
]

# When True, run_all.py trains/evaluates all predefined FEATURE_SETS below.
USE_PREDEFINED_FEATURE_SETS = True
FEATURE_SETS = {
    "Set_A_All_Features": [
        "Income",
        "Debt",
        "Owns Property",
        "Profession",
        "Demographics",
        "Platform",
        "Total Time Spent",
        "Number of Sessions",
        "Video Category",
        "Video Length",
        "Engagement",
        "Importance Score",
        "Time Spent On Video",
        "Number of Videos Watched",
        "Scroll Rate",
        "Frequency",
        "Watch Reason",
        "Watch Time",
        "CurrentActivity",
        "Gender",
        "Location",
        "Age",
    ],
    "Set_B_Behavior_Macro": [
        "Total Time Spent",
        "Number of Sessions",
        "Frequency",
        "Watch Time",
        "CurrentActivity",
    ],
    # "Set_K_HighImpact_Behavior": [
    #     "Number of Videos Watched",
    #     "Importance Score",
    #     "Time Spent On Video",
    #     "Engagement"
    # ],
    "Set_L_Content_Behavior_Interaction": [
        "Video Category",
        "Watch Time",
        "Frequency",
        "Number of Videos Watched",
        "Time Spent On Video",
    ],
    "Set_M_Platform_Pattern": [
        "Platform",
        "Total Time Spent",
        "Number of Sessions",
        "Watch Time",
    ],
    # "Set_N_Engagement_Addiction": [
    #     "Importance Score",
    #     "Engagement",
    #     "Scroll Rate",
    #     "Time Spent On Video",
    #     "Number of Videos Watched",
    # ],
    # "Set_O_Context_Behavior": [
    #     "CurrentActivity",
    #     "Watch Reason",
    #     "Frequency",
    #     "Number of Sessions",
    #     "Total Time Spent",
    # ],
    "Set_P_Persona_Core": [
        "Number of Videos Watched",
        "Frequency",
        "Engagement",
        "Video Category",
        "Watch Time",
        "Importance Score",
    ],
}


# 'Income', 'Debt', 'Owns Property', 'Profession', 'Demographics', 'Platform', 'Total Time Spent', 'Number of Sessions', 
# 'Video Category', 'Video Length', 'Engagement', 'Importance Score', 'Time Spent On Video', 'Number of Videos Watched', 
# 'Scroll Rate', 'Frequency', 'Watch Reason', 'Watch Time', 'CurrentActivity', "Gender", "Location", "Age"
EXCLUDE_FEATURE_COLS = [
    "DeviceType", "OS", "ConnectionType", "Self Control", 
    "Satisfaction"
]

RANDOM_SEED = 42
TEST_SIZE = 0.2
VAL_SIZE = 0.2  # fraction of train used as validation (train -> train/val split)

# Test set held to do final estimating, determining the performance of the model
# Val set used for tuning, model selection, early stopping, etc. 验证集
