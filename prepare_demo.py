"""Build small, real-data aggregates for the local RiskLens demo."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import shap

from baseline import DATA, ROOT, make_features, read_data


OUT = ROOT / "outputs"
GROUPS = ["grade", "term", "purpose", "homeOwnership"]
DRIFT_COLUMNS = ["loanAmnt", "annualIncome", "interestRate", "dti", "ficoRangeLow", "grade", "term"]
DETAIL_COLUMNS = ["loanAmnt", "annualIncome", "interestRate", "dti", "ficoRangeLow", "term", "grade", "employmentLength"]


def number(value: object) -> float | None:
    if pd.isna(value):
        return None
    return float(value)


def psi_numeric(reference: pd.Series, current: pd.Series) -> float:
    ref = pd.to_numeric(reference, errors="coerce").dropna().to_numpy()
    cur = pd.to_numeric(current, errors="coerce").dropna().to_numpy()
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, 11)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    a = np.histogram(ref, bins=edges)[0] / len(ref)
    b = np.histogram(cur, bins=edges)[0] / len(cur)
    return float(np.sum((b - a) * np.log((b + 1e-6) / (a + 1e-6))))


def psi_category(reference: pd.Series, current: pd.Series) -> float:
    a = reference.fillna("缺失").astype(str).value_counts(normalize=True)
    b = current.fillna("缺失").astype(str).value_counts(normalize=True)
    labels = a.index.union(b.index)
    av = a.reindex(labels, fill_value=0).to_numpy()
    bv = b.reindex(labels, fill_value=0).to_numpy()
    return float(np.sum((bv - av) * np.log((bv + 1e-6) / (av + 1e-6))))


def main() -> None:
    print("Reading training and test data", flush=True)
    train = read_data(DATA / "train.csv")
    test = read_data(DATA / "testA.csv")
    predictions = pd.read_csv(OUT / "submission.csv", dtype={"id": "int32", "isDefault": "float32"})
    metrics = json.loads((OUT / "metrics.json").read_text(encoding="utf-8"))
    if not np.array_equal(test.id.to_numpy(), predictions.id.to_numpy()):
        raise ValueError("Test IDs differ from submission IDs")

    print("Calculating data quality and portfolio aggregates", flush=True)
    column_stats = []
    for name in train.columns:
        series = train[name]
        numeric = pd.api.types.is_numeric_dtype(series)
        stat = {
            "name": name,
            "type": "target" if name == "isDefault" else "numeric" if numeric else "category",
            "missing_pct": round(float(series.isna().mean() * 100), 3),
            "unique": int(series.nunique(dropna=True)),
        }
        if numeric:
            stat.update({
                "p01": number(series.quantile(0.01)),
                "median": number(series.median()),
                "mean": number(series.mean()),
                "p99": number(series.quantile(0.99)),
            })
        column_stats.append(stat)

    groups = {}
    for group in GROUPS:
        table = train.groupby(group, dropna=False, observed=True).isDefault.agg(["size", "mean"])
        table = table.sort_values("size", ascending=False).head(20)
        groups[group] = [
            {"label": str(index), "count": int(row["size"]), "default_rate": float(row["mean"])}
            for index, row in table.iterrows()
        ]

    bands = pd.cut(train.annualIncome, [-np.inf, 30000, 60000, 100000, 200000, np.inf], labels=["≤3万", "3–6万", "6–10万", "10–20万", ">20万"])
    income = train.groupby(bands, observed=False).isDefault.agg(["size", "mean"])
    groups["incomeBand"] = [
        {"label": str(index), "count": int(row["size"]), "default_rate": float(row["mean"])}
        for index, row in income.iterrows()
    ]

    drift = []
    for name in DRIFT_COLUMNS:
        value = psi_category(train[name], test[name]) if name == "grade" else psi_numeric(train[name], test[name])
        drift.append({"name": name, "psi": round(value, 4), "status": "漂移" if value >= 0.25 else "关注" if value >= 0.1 else "正常"})

    detail = {}
    for name in DETAIL_COLUMNS:
        s = train[name]
        if pd.api.types.is_numeric_dtype(s):
            edges = np.unique(np.quantile(s.dropna(), np.linspace(0, 1, 11)))
            if len(edges) >= 3:
                edges[0], edges[-1] = -np.inf, np.inf
                bucket = pd.cut(s, edges, include_lowest=True)
                counts = train.groupby([bucket, "isDefault"], observed=False).size().unstack(fill_value=0)
                detail[name] = [{"label": str(index), "good": int(row.get(0, 0)), "bad": int(row.get(1, 0))} for index, row in counts.iterrows()]
        else:
            counts = train.groupby([name, "isDefault"], dropna=False, observed=True).size().unstack(fill_value=0)
            counts = counts.loc[counts.sum(axis=1).sort_values(ascending=False).head(12).index]
            detail[name] = [{"label": str(index), "good": int(row.get(0, 0)), "bad": int(row.get(1, 0))} for index, row in counts.iterrows()]

    dataset = {
        "train_rows": len(train),
        "test_rows": len(test),
        "column_count": len(train.columns),
        "default_rate": float(train.isDefault.mean()),
        "missing_cells": int(train.isna().sum().sum()),
        "duplicate_rows": int(train.duplicated().sum()),
        "validation_auc": metrics["validation_auc"],
        "high_risk_count": int((predictions.isDefault >= 0.35).sum()),
        "risk_distribution": {
            "低风险": int((predictions.isDefault < 0.15).sum()),
            "中风险": int(((predictions.isDefault >= 0.15) & (predictions.isDefault < 0.35)).sum()),
            "高风险": int((predictions.isDefault >= 0.35).sum()),
        },
        "columns": column_stats,
        "groups": groups,
        "drift": drift,
        "detail": detail,
    }
    (OUT / "demo.json").write_text(json.dumps(dataset, ensure_ascii=False), encoding="utf-8")
    print("Wrote outputs/demo.json", flush=True)
    print("Preparing model features and probability-space SHAP explanations", flush=True)
    train_features, test_features = make_features(train, test)
    test_features.to_parquet(OUT / "test_features.parquet", index=False)
    background = train_features.sample(n=min(100, len(train_features)), random_state=42)
    background.to_parquet(OUT / "shap_background.parquet", index=False)
    model = joblib.load(OUT / "model.joblib")
    explainer = shap.TreeExplainer(model, background, feature_perturbation="interventional", model_output="probability")
    sample = test_features.sample(n=min(300, len(test_features)), random_state=42)
    explanation = explainer(sample)
    values = np.asarray(explanation.values)
    base = float(np.asarray(explanation.base_values).mean())
    predicted = model.predict_proba(sample)[:, 1]
    if not np.allclose(base + values.sum(axis=1), predicted, atol=1e-4):
        raise ValueError("SHAP contributions do not reconstruct XGBoost probabilities")
    mean_abs = np.mean(np.abs(values), axis=0)
    order = np.argsort(mean_abs)[::-1]
    importance = [{"feature": sample.columns[i], "mean_abs": float(mean_abs[i]),
                   "mean_signed": float(values[:, i].mean())} for i in order]
    beeswarm = {
        sample.columns[i]: [{"value": number(value), "contribution": float(contribution)}
                            for value, contribution in zip(sample.iloc[:100, i], values[:100, i])]
        for i in order[:8]
    }
    cohort = {}
    sampled_grades = test.iloc[sample.index].grade.reset_index(drop=True)
    for grade in "ABCDEFG":
        mask = sampled_grades.eq(grade).to_numpy()
        if not mask.any():
            continue
        signed = values[mask].mean(axis=0)
        ranked = np.argsort(np.abs(signed))[::-1][:6]
        cohort[grade] = {"sample_rows": int(mask.sum()), "average_pd": float(predicted[mask].mean()),
                         "contributors": [{"feature": sample.columns[i], "contribution": float(signed[i])} for i in ranked]}
    high_mask = predicted >= .35
    if high_mask.any():
        signed = values[high_mask].mean(axis=0)
        ranked = np.argsort(np.abs(signed))[::-1][:6]
        cohort["high_risk"] = {"sample_rows": int(high_mask.sum()), "average_pd": float(predicted[high_mask].mean()),
                               "contributors": [{"feature": sample.columns[i], "contribution": float(signed[i])} for i in ranked]}
    (OUT / "shap_summary.json").write_text(json.dumps({
        "method": "TreeExplainer / interventional / probability", "background_rows": len(background),
        "sample_rows": len(sample), "base_probability": base, "feature_importance": importance,
        "beeswarm": beeswarm, "cohorts": cohort,
        "note": "SHAP 表示特征对当前模型预测的贡献，非因果关系；全局和客群结果来自测试集 A 的固定随机样本。",
    }, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
