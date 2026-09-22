"""Train a reproducible loan default baseline and write a Tianchi submission.

Usage: py baseline.py
       py baseline.py --sample-rows 20000 --max-iter 20 --output-dir outputs/smoke
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

# Windows sandbox environments can deny the extra joblib worker handles.
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
TEXT_COLUMNS = ("grade", "subGrade", "employmentLength")
FREQUENCY_COLUMNS = ("employmentTitle", "postCode", "title")


def read_data(path: Path) -> pd.DataFrame:
    columns = pd.read_csv(path, nrows=0).columns
    text_columns = set(TEXT_COLUMNS) | {"issueDate", "earliesCreditLine"}
    dtypes = {
        column: ("int32" if column == "id" else "int8" if column == "isDefault" else str if column in text_columns else "float32")
        for column in columns
    }
    return pd.read_csv(path, dtype=dtypes)


def make_features(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_x = train.drop(columns=["id", "isDefault"]).copy()
    test_x = test.drop(columns=["id"]).copy()

    for frame in (train_x, test_x):
        issued = pd.to_datetime(frame.pop("issueDate"), errors="coerce")
        credit = pd.to_datetime(frame.pop("earliesCreditLine"), format="%b-%Y", errors="coerce")
        frame["issue_year"] = issued.dt.year
        frame["issue_month"] = issued.dt.month
        frame["credit_history_months"] = (
            (issued.dt.year - credit.dt.year) * 12 + issued.dt.month - credit.dt.month
        )
        frame["loan_to_income"] = frame["loanAmnt"] / frame["annualIncome"].replace(0, np.nan)

    for column in TEXT_COLUMNS:
        if column not in train_x:
            continue
        labels = pd.Index(train_x[column].dropna().astype(str).unique())
        mapping = {label: number for number, label in enumerate(labels)}
        for frame in (train_x, test_x):
            frame[column] = frame[column].astype("string").map(mapping).astype("float32")

    for column in FREQUENCY_COLUMNS:
        counts = train_x[column].value_counts(dropna=False)
        for frame in (train_x, test_x):
            frame[f"{column}_frequency"] = frame[column].map(counts).fillna(0)

    for frame in (train_x, test_x):
        for column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("float32")
        frame.replace([np.inf, -np.inf], np.nan, inplace=True)

    if list(train_x.columns) != list(test_x.columns):
        raise ValueError("Training and test feature columns differ")
    return train_x, test_x


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-rows", type=int, help="Use a random training subset for a quick run")
    parser.add_argument("--max-iter", type=int, default=150)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs")
    args = parser.parse_args()

    train = read_data(DATA / "train.csv")
    test = read_data(DATA / "testA.csv")
    sample = pd.read_csv(DATA / "sample_submit.csv")
    if args.sample_rows:
        train = train.sample(n=min(args.sample_rows, len(train)), random_state=42)
    if "isDefault" not in train or "isDefault" in test:
        raise ValueError("Unexpected target column in input files")
    if not train["isDefault"].isin([0, 1]).all():
        raise ValueError("Training target must be binary")
    if not np.array_equal(test["id"].to_numpy(), sample["id"].to_numpy()):
        raise ValueError("Test IDs do not match sample submission")

    x, x_test = make_features(train, test)
    y = train["isDefault"].astype("int8")
    fit_idx, valid_idx = train_test_split(
        np.arange(len(y)), test_size=0.2, random_state=42, stratify=y
    )
    model = HistGradientBoostingClassifier(
        max_iter=args.max_iter,
        max_leaf_nodes=31,
        learning_rate=0.06,
        l2_regularization=1.0,
        early_stopping=True,
        random_state=42,
    )
    print(f"Training on {len(fit_idx):,} rows; validating on {len(valid_idx):,} rows", flush=True)
    model.fit(x.iloc[fit_idx], y.iloc[fit_idx])
    valid_probability = model.predict_proba(x.iloc[valid_idx])[:, 1]
    auc = roc_auc_score(y.iloc[valid_idx], valid_probability)
    print(f"Validation AUC: {auc:.6f}; iterations: {model.n_iter_}", flush=True)

    # Refit with all available labels, using the iteration count selected above.
    final_model = HistGradientBoostingClassifier(
        max_iter=model.n_iter_,
        max_leaf_nodes=31,
        learning_rate=0.06,
        l2_regularization=1.0,
        early_stopping=False,
        random_state=42,
    )
    print(f"Refitting on all {len(y):,} training rows", flush=True)
    final_model.fit(x, y)
    submission = sample[["id"]].copy()
    submission["isDefault"] = final_model.predict_proba(x_test)[:, 1]
    if submission.isna().any().any() or not submission["isDefault"].between(0, 1).all():
        raise ValueError("Submission contains invalid probabilities")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_dir / "submission.csv", index=False)
    validation = train.iloc[valid_idx][
        ["id", "isDefault", "grade", "term", "annualIncome", "loanAmnt", "interestRate", "dti", "ficoRangeLow", "issueDate"]
    ].copy()
    validation["probability"] = valid_probability
    validation.to_csv(output_dir / "validation.csv", index=False)
    joblib.dump(final_model, output_dir / "model.joblib")
    report = {
        "train_rows": len(train),
        "test_rows": len(test),
        "feature_count": x.shape[1],
        "validation_auc": auc,
        "validation_iterations": model.n_iter_,
        "random_state": 42,
    }
    (output_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {output_dir / 'submission.csv'}", flush=True)


if __name__ == "__main__":
    main()
