"""Train comparable LR and XGBoost models and write a Tianchi submission.

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
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


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
    parser.add_argument("--max-iter", type=int, default=500, help="Maximum XGBoost boosting rounds")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
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
    print(f"Training on {len(fit_idx):,} rows; validating on {len(valid_idx):,} rows", flush=True)
    lr = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=300, random_state=42))
    lr.fit(x.iloc[fit_idx], y.iloc[fit_idx])
    lr_probability = lr.predict_proba(x.iloc[valid_idx])[:, 1]
    print(f"LR validation AUC: {roc_auc_score(y.iloc[valid_idx], lr_probability):.6f}", flush=True)

    xgb_params = dict(
        n_estimators=args.max_iter, max_depth=6, learning_rate=.05, subsample=.8,
        colsample_bytree=.8, objective="binary:logistic", eval_metric="auc",
        tree_method="hist", device=args.device, n_jobs=4, random_state=42,
        enable_categorical=False,
    )
    inner_fit, stop_idx = train_test_split(fit_idx, test_size=.1, random_state=43, stratify=y.iloc[fit_idx])
    tuner = XGBClassifier(**xgb_params, early_stopping_rounds=min(30, max(5, args.max_iter // 4)))
    tuner.fit(x.iloc[inner_fit], y.iloc[inner_fit], eval_set=[(x.iloc[stop_idx], y.iloc[stop_idx])], verbose=False)
    selected_rounds = int(tuner.best_iteration + 1)
    # The reported holdout is never used for early stopping or parameter selection.
    model = XGBClassifier(**(xgb_params | {"n_estimators": selected_rounds}))
    model.fit(x.iloc[fit_idx], y.iloc[fit_idx], verbose=False)
    valid_probability = model.predict_proba(x.iloc[valid_idx])[:, 1]
    print(f"XGBoost validation AUC: {roc_auc_score(y.iloc[valid_idx], valid_probability):.6f}; rounds: {selected_rounds}", flush=True)

    # Refit both models on the full labeled set after selecting XGBoost rounds on the holdout.
    print(f"Refitting both models on all {len(y):,} training rows", flush=True)
    final_lr = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=300, random_state=42))
    final_lr.fit(x, y)
    final_model = XGBClassifier(**(xgb_params | {"n_estimators": selected_rounds}))
    final_model.fit(x, y, verbose=False)
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
    validation["lr_probability"] = lr_probability
    validation.to_csv(output_dir / "validation.csv", index=False)
    joblib.dump(final_model, output_dir / "model.joblib")
    joblib.dump(final_lr, output_dir / "lr_model.joblib")

    def score_report(probability: np.ndarray) -> dict:
        truth = y.iloc[valid_idx]
        fpr, tpr, _ = roc_curve(truth, probability)
        return {"auc": float(roc_auc_score(truth, probability)), "ks": float(np.max(tpr - fpr)),
                "pr_auc": float(average_precision_score(truth, probability)),
                "brier": float(brier_score_loss(truth, probability))}

    report = {
        "train_rows": len(train),
        "test_rows": len(test),
        "feature_count": x.shape[1],
        "validation_auc": float(roc_auc_score(y.iloc[valid_idx], valid_probability)),
        "validation_iterations": selected_rounds,
        "main_model": "XGBoost",
        "models": {"Logistic Regression": score_report(lr_probability), "XGBoost": score_report(valid_probability)},
        "device": args.device,
        "random_state": 42,
    }
    (output_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {output_dir / 'submission.csv'}", flush=True)


if __name__ == "__main__":
    main()
