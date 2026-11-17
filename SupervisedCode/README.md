# Training and evaluation pipeline

Reproduce the saved experiments for two dataset-defined targets: **Addiction Level** and **ProductivityLoss**. The pipeline compares multinomial Logistic Regression, Random Forest, and CatBoost over the configured feature sets.

## Environment

Use Python 3.11 and install the root dependencies:

```bash
python -m pip install -r requirements.txt
```

## Data source

Download **Dark Side of Social Media** by Muhammad Roshan Riaz from [Kaggle](https://www.kaggle.com/datasets/muhammadroshaanriaz/time-wasters-on-social-media), and place the CSV at:

```text
SupervisedCode/data/raw/time_wasters.csv
```

The CSV is deliberately not committed because it is third-party content.

## Run

From the repository root:

```bash
python SupervisedCode/scripts/run_all.py
```

The runner uses the fixed random seed and split sizes in `src/config.py`, saves metrics under `outputs/metrics/`, writes generated model artifacts under `outputs/models/`, and adds a summary to `result.txt` for new feature signatures. Full retraining can take substantial time.

## Configuration

Edit `src/config.py` to choose targets, feature sets, included/excluded features, and train/validation/test split sizes. `EXCLUDE_FEATURE_COLS` excludes `Self Control` and `Satisfaction` from the default non-predefined feature path. The bundled `FEATURE_SETS` are explicit and may include different columns, so review them individually.

`outputs/metrics/` contains saved validation and test results supplied with the original archive, while large trained `outputs/models/` artifacts are ignored by Git. The small demonstration model is bundled separately at `../final_models/`.
