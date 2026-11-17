import sys
from pathlib import Path
import hashlib
import re

ROOT = Path(__file__).resolve().parents[1]  # project root
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import (
    Paths,
    TARGETS,
    ID_COLS,
    VIDEO_ID_COL,
    INCLUDE_VIDEO_ID,
    EXCLUDE_FEATURE_COLS,
    USE_INCLUDE_FEATURE_COLS,
    INCLUDE_FEATURE_COLS,
    USE_PREDEFINED_FEATURE_SETS,
    FEATURE_SETS,
)
from src.data import load_csv, basic_clean
from src.train import train_for_target
from src.evaluate import evaluate, save_metrics
from src.utils import ensure_dirs, save_model

RESULT_FILE = ROOT / "result.txt"
FEATURE_SIG_PREFIX = "feature_signature_sha1="


def _feature_signature(feature_cols: list[str]) -> str:
    """Stable signature for comparing feature sets, regardless of column order."""
    payload = "|".join(sorted(feature_cols)).encode("utf-8")
    return hashlib.sha1(payload).hexdigest()


def _load_existing_feature_signatures(result_file: Path) -> set[str]:
    if not result_file.exists():
        return set()

    signatures: set[str] = set()
    with open(result_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith(FEATURE_SIG_PREFIX):
                signatures.add(line.strip().split("=", maxsplit=1)[1])
    return signatures


def _append_results(
    result_file: Path,
    set_name: str,
    feature_cols: list[str],
    rows: list[dict[str, str | float]],
) -> None:
    result_file.parent.mkdir(parents=True, exist_ok=True)

    signature = _feature_signature(feature_cols)

    with open(result_file, "a", encoding="utf-8") as f:
        f.write("=== RUN START ===\n")
        f.write(f"set_name={set_name}\n")
        f.write(f"{FEATURE_SIG_PREFIX}{signature}\n")
        f.write(f"feature_count={len(feature_cols)}\n")
        f.write(f"features={feature_cols}\n")
        for row in rows:
            f.write(
                "set={set_name} target={target} model={model} split={split} "
                "accuracy={accuracy:.6f} macro_f1={macro_f1:.6f} mae={mae:.6f}\n".format(
                    set_name=set_name,
                    target=row["target"],
                    model=row["model"],
                    split=row["split"],
                    accuracy=row["accuracy"],
                    macro_f1=row["macro_f1"],
                    mae=row["mae"],
                )
            )
        f.write("=== RUN END ===\n\n")


def _safe_slug(name: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", name.strip()).strip("_").lower()
    return slug or "feature_set"


def _resolve_feature_cols(
    df_cols: list[str],
    requested_cols: list[str],
    protected_drop_cols: set[str],
) -> tuple[list[str], list[str], list[str]]:
    feature_cols = [c for c in requested_cols if c in df_cols and c not in protected_drop_cols]
    missing_cols = [c for c in requested_cols if c not in df_cols]
    dropped_protected = [c for c in requested_cols if c in protected_drop_cols]
    return feature_cols, missing_cols, dropped_protected


def main():
    P = Paths()
    ensure_dirs(P.data_processed, P.outputs, P.models, P.metrics, P.figures)

    # load kaggle data, perform basic cleaning (strip whitespace, drop duplicates)
    csv_path = P.data_raw / "time_wasters.csv"
    df = basic_clean(load_csv(csv_path))

    # Build protected drop columns: targets + IDs (and optionally Video ID)
    protected_drop_cols = set(TARGETS + ID_COLS)
    if not INCLUDE_VIDEO_ID and VIDEO_ID_COL in df.columns: #drop video id column
        protected_drop_cols.add(VIDEO_ID_COL)

    if USE_PREDEFINED_FEATURE_SETS:
        requested_feature_sets = FEATURE_SETS
        # print("Feature mode: predefined feature sets (USE_PREDEFINED_FEATURE_SETS=True)")
    else:
        if USE_INCLUDE_FEATURE_COLS:
            requested_feature_sets = {"Configured_Features": INCLUDE_FEATURE_COLS}
            # print("Feature mode: include-only (USE_INCLUDE_FEATURE_COLS=True)")
        else:
            drop_cols = set(EXCLUDE_FEATURE_COLS) | protected_drop_cols
            requested_cols = [c for c in df.columns if c not in drop_cols]
            requested_feature_sets = {"Configured_Features": requested_cols}
            # print("Feature mode: exclude-list (USE_INCLUDE_FEATURE_COLS=False)")

    existing_sigs = _load_existing_feature_signatures(RESULT_FILE)

    for set_name, requested_cols in requested_feature_sets.items():
        feature_cols, missing_cols, dropped_protected = _resolve_feature_cols(
            df.columns.tolist(),
            requested_cols,
            protected_drop_cols,
        )
        if missing_cols:
            # print(f"[{set_name}] Warning: features not found in dataframe: {missing_cols}")
            pass
        if dropped_protected:
            # print(f"[{set_name}] Note: protected columns were removed: {dropped_protected}")
            pass
        if not feature_cols:
            # print(f"[{set_name}] No valid features selected. Skipping set.")
            continue

        # print(f"\n=== Running {set_name} ===")
        # print(f"Feature columns ({len(feature_cols)}): {feature_cols}")
        set_slug = _safe_slug(set_name)
        run_rows: list[dict[str, str | float]] = []

        # Train + evaluate per target (you can change model_name to "rf")
        for target in TARGETS:
            for model_name in ["logreg", "rf", "catboost"]:
                out = train_for_target(df, feature_cols, target_col=target, model_name=model_name)

                val_metrics = evaluate(out.model, out.X_val, out.y_val)
                test_metrics = evaluate(out.model, out.X_test, out.y_test)
                run_rows.append(
                    {
                        "target": target,
                        "model": model_name,
                        "split": "val",
                        "accuracy": val_metrics.accuracy,
                        "macro_f1": val_metrics.macro_f1,
                        "mae": val_metrics.mae_ordinal,
                    }
                )
                run_rows.append(
                    {
                        "target": target,
                        "model": model_name,
                        "split": "test",
                        "accuracy": test_metrics.accuracy,
                        "macro_f1": test_metrics.macro_f1,
                        "mae": test_metrics.mae_ordinal,
                    }
                )

                safe_target = target.replace(" ", "_").lower()
                if model_name == "catboost":
                    model_path = P.models / f"{set_slug}_{safe_target}_{model_name}.cbm"
                    out.model.save_model(model_path)
                else:
                    model_path = P.models / f"{set_slug}_{safe_target}_{model_name}.joblib"
                    save_model(out.model, model_path)

                save_metrics(val_metrics, P.metrics / f"{set_slug}_{safe_target}_{model_name}_val.json")
                save_metrics(test_metrics, P.metrics / f"{set_slug}_{safe_target}_{model_name}_test.json")

                # print(f"[{set_name} | {target} | {model_name}] VAL acc={val_metrics.accuracy:.3f} mf1={val_metrics.macro_f1:.3f} mae={val_metrics.mae_ordinal:.3f}")
                # print(f"[{set_name} | {target} | {model_name}] TEST acc={test_metrics.accuracy:.3f} mf1={test_metrics.macro_f1:.3f} mae={test_metrics.mae_ordinal:.3f}")
                # print(f"Saved: {model_path}")

        current_sig = _feature_signature(feature_cols)
        if current_sig in existing_sigs:
            # print(f"[{set_name}] Feature set already exists in {RESULT_FILE}. Skipping append.")
            pass
        else:
            _append_results(RESULT_FILE, set_name, feature_cols, run_rows)
            existing_sigs.add(current_sig)
            # print(f"[{set_name}] Appended run summary to {RESULT_FILE}")
        print(f"Completed {set_name}")

if __name__ == "__main__":
    main()
