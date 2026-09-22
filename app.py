"""Local RiskLens demo. Start with: py -m uvicorn app:app --reload"""

from __future__ import annotations

import asyncio
import io
import json
import os
import re
import threading
import urllib.error
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field
from sklearn.calibration import calibration_curve
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from baseline import DATA, ROOT, make_features, read_data


OUT = ROOT / "outputs"
STATIC = ROOT / "static"
app = FastAPI(title="RiskLens", version="0.1.0")
executor = ThreadPoolExecutor(max_workers=1)
jobs: dict[str, dict] = {}
jobs_lock = threading.Lock()
FEATURES = {
    "loan_income_ratio": {"label": "贷款收入比", "formula": "loanAmnt / annualIncome", "description": "贷款金额相对年收入的比例"},
    "monthly_income": {"label": "月收入", "formula": "annualIncome / 12", "description": "年收入按 12 个月折算"},
    "credit_history_years": {"label": "信用历史年数", "formula": "issueDate - earliesCreditLine", "description": "放款时已有的信用记录年限"},
    "issue_month": {"label": "放款月份", "formula": "month(issueDate)", "description": "从放款日期提取月份"},
    "installment_income_ratio": {"label": "月供收入比", "formula": "installment / (annualIncome / 12)", "description": "月供相对估算月收入的比例"},
}
BUSINESS_FEATURE_LABELS = {
    "interestRate": "贷款利率", "dti": "债务收入比", "ficoRangeLow": "FICO 下限",
    "grade": "信用等级", "subGrade": "细分信用等级", "loan_to_income": "贷款收入比",
    "annualIncome": "年收入", "loanAmnt": "贷款金额", "credit_history_months": "信用历史",
    "revolUtil": "循环额度利用率", "term": "贷款期限", "employmentLength": "工作年限",
    "homeOwnership": "住房情况", "installment": "月供", "issue_year": "贷款发放年份",
    "issue_month": "贷款发放月份",
}


@lru_cache(maxsize=1)
def demo() -> dict:
    path = OUT / "demo.json"
    if not path.exists():
        raise HTTPException(503, "请先运行 py prepare_demo.py")
    return json.loads(path.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def validation() -> pd.DataFrame:
    path = OUT / "validation.csv"
    if not path.exists():
        raise HTTPException(503, "请先运行 py baseline.py")
    return pd.read_csv(path)


@lru_cache(maxsize=1)
def customers() -> pd.DataFrame:
    cols = ["id", "loanAmnt", "term", "interestRate", "grade", "annualIncome", "dti", "ficoRangeLow", "employmentLength", "purpose", "issueDate"]
    frame = pd.read_csv(DATA / "testA.csv", usecols=cols)
    scores = pd.read_csv(OUT / "submission.csv", usecols=["id", "isDefault"])
    if not np.array_equal(frame.id.to_numpy(), scores.id.to_numpy()):
        raise HTTPException(500, "客户 ID 与预测结果不一致")
    frame["probability"] = scores.isDefault.to_numpy()
    return frame.set_index("id", drop=False)


@lru_cache(maxsize=1)
def model_and_features():
    return (joblib.load(OUT / "model.joblib"), joblib.load(OUT / "lr_model.joblib"),
            pd.read_parquet(OUT / "test_features.parquet"))


@lru_cache(maxsize=1)
def shap_explainer():
    model, _, _ = model_and_features()
    background = pd.read_parquet(OUT / "shap_background.parquet")
    return shap.TreeExplainer(model, background, feature_perturbation="interventional", model_output="probability")


@lru_cache(maxsize=1)
def shap_summary() -> dict:
    return json.loads((OUT / "shap_summary.json").read_text(encoding="utf-8"))


def explain_customer(position: int) -> dict:
    model, lr, features = model_and_features()
    sample = features.iloc[[position]]
    explanation = shap_explainer()(sample)
    contributions = np.asarray(explanation.values)[0]
    base = float(np.asarray(explanation.base_values).reshape(-1)[0])
    probability = float(model.predict_proba(sample)[0, 1])
    reconstructed = base + float(contributions.sum())
    if not np.isclose(reconstructed, probability, atol=1e-4):
        raise HTTPException(500, "SHAP 解释与模型预测不一致")
    rows = sorted([
        {"feature": name, "value": None if pd.isna(sample.iloc[0][name]) else float(sample.iloc[0][name]),
         "contribution": float(value)} for name, value in zip(features.columns, contributions)
    ], key=lambda item: abs(item["contribution"]), reverse=True)
    return {"base_probability": base, "prediction": probability,
            "lr_probability": float(lr.predict_proba(sample)[0, 1]), "contributions": rows,
            "method": "TreeExplainer / interventional / probability",
            "background_rows": len(shap_explainer().data),
            "note": "SHAP 反映特征对当前模型预测的贡献，不代表因果关系或独立的信贷决策依据。"}


def thin_curve(x: np.ndarray, y: np.ndarray, limit: int = 120) -> list[dict]:
    indices = np.unique(np.linspace(0, len(x) - 1, min(limit, len(x))).astype(int))
    return [{"x": float(x[i]), "y": float(y[i])} for i in indices]


@lru_cache(maxsize=1)
def evaluation() -> dict:
    frame = validation()
    truth = frame.isDefault.to_numpy()
    score = frame.probability.to_numpy()
    fpr, tpr, _ = roc_curve(truth, score)
    precision, recall, _ = precision_recall_curve(truth, score)
    cal_true, cal_pred = calibration_curve(truth, score, n_bins=10, strategy="quantile")
    return {
        "auc": float(roc_auc_score(truth, score)),
        "ks": float(np.max(tpr - fpr)),
        "pr_auc": float(average_precision_score(truth, score)),
        "roc": thin_curve(fpr, tpr),
        "pr": thin_curve(recall[::-1], precision[::-1]),
        "calibration": thin_curve(cal_pred, cal_true, 10),
        "validation_rows": len(frame),
        "brier": float(np.mean((score - truth) ** 2)),
    }


def threshold_metrics(threshold: float, bad_cost: float, good_cost: float) -> dict:
    frame = validation()
    truth = frame.isDefault.to_numpy().astype(bool)
    flagged = frame.probability.to_numpy() >= threshold
    tp = int(np.sum(flagged & truth))
    fp = int(np.sum(flagged & ~truth))
    fn = int(np.sum(~flagged & truth))
    tn = int(np.sum(~flagged & ~truth))
    return {
        "threshold": threshold, "flagged": tp + fp, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": tp / (tp + fp) if tp + fp else 0,
        "recall": tp / (tp + fn) if tp + fn else 0,
        "false_reject_rate": fp / (fp + tn) if fp + tn else 0,
        "estimated_cost": fn * bad_cost + fp * good_cost,
        "bad_cost": bad_cost, "good_cost": good_cost,
        "basis": "16 万条留出验证集；成本为输入假设，不代表真实贷款损失",
    }


def risk_level(score: pd.Series) -> pd.Series:
    return pd.cut(score, [-np.inf, .15, .35, np.inf], labels=["低风险", "中风险", "高风险"], right=False)


def filter_customers(grade: str = "all", purpose: str = "all", income: str = "all", risk: str = "all") -> pd.DataFrame:
    if grade not in {"all", "A", "B", "C", "D", "E", "F", "G"} or risk not in {"all", "low", "medium", "high"}:
        raise HTTPException(422, "无效的等级或风险筛选")
    if income not in {"all", "low", "middle", "high"}:
        raise HTTPException(422, "无效的收入筛选")
    table = customers()
    if purpose != "all":
        if not purpose.isdigit() or int(purpose) not in table.purpose.unique():
            raise HTTPException(422, "无效的贷款用途编码")
        table = table[table.purpose.eq(int(purpose))]
    if grade != "all":
        table = table[table.grade.eq(grade)]
    if income == "low":
        table = table[table.annualIncome.le(60000)]
    elif income == "middle":
        table = table[table.annualIncome.gt(60000) & table.annualIncome.le(120000)]
    elif income == "high":
        table = table[table.annualIncome.gt(120000)]
    if risk != "all":
        level = risk_level(table.probability)
        table = table[level.eq({"low": "低风险", "medium": "中风险", "high": "高风险"}[risk])]
    return table


def risk_clues(row: pd.Series) -> list[str]:
    clues = []
    if row.interestRate >= 18:
        clues.append("利率较高")
    if row.dti >= 30:
        clues.append("债务收入比较高")
    if row.ficoRangeLow < 670:
        clues.append("信用评分偏低")
    if row.annualIncome > 0 and row.loanAmnt / row.annualIncome >= .5:
        clues.append("贷款收入比较高")
    if row.grade in {"D", "E", "F", "G"}:
        clues.append("信用等级偏低")
    return clues[:2] or ["查看客户详情"]


def risk_overview_data(grade: str = "all", purpose: str = "all", income: str = "all", risk: str = "all") -> dict:
    table = filter_customers(grade, purpose, income, risk)
    reference = filter_customers(grade, purpose, income, "all")
    distribution = risk_level(table.probability).value_counts().reindex(["低风险", "中风险", "高风险"], fill_value=0)
    high = table[table.probability.ge(.35)]
    factors = []
    labels = {"interestRate": "贷款利率", "dti": "债务收入比", "ficoRangeLow": "FICO 下限", "annualIncome": "年收入"}
    if len(high):
        for column, label in labels.items():
            factors.append({"label": label, "high_median": float(high[column].median()), "population_median": float(reference[column].median())})
    top = []
    for row in table.nlargest(10, "probability").itertuples():
        top.append({"id": int(row.id), "probability": float(row.probability), "grade": row.grade, "clues": risk_clues(row)})
    return {
        "count": len(table), "high_count": int(distribution["高风险"]), "medium_count": int(distribution["中风险"]),
        "high_share": float(distribution["高风险"] / len(table)) if len(table) else 0,
        "distribution": {name: int(count) for name, count in distribution.items()},
        "factors": factors, "top_customers": top, "threshold": .35,
        "filter": {"grade": grade, "purpose": purpose, "income": income, "risk": risk},
        "basis": "测试集 A 的模型预测；风险线索与组间中位数是描述性对照，不是 SHAP 或因果归因。",
    }


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/static/{filename}")
def static_file(filename: str):
    if filename not in {"app.js", "style.css"}:
        raise HTTPException(404)
    return FileResponse(STATIC / filename)


@app.get("/api/overview")
def overview():
    d = demo()
    return {key: d[key] for key in ["train_rows", "test_rows", "column_count", "default_rate", "missing_cells", "duplicate_rows", "high_risk_count", "risk_distribution"]} | {"evaluation": evaluation(), "drift": d["drift"][:4]}


@app.get("/api/risk-overview")
def risk_overview(grade: str = "all", purpose: str = "all", income: str = "all", risk: str = "all"):
    return risk_overview_data(grade, purpose, income, risk)


@app.get("/api/risk-filters")
def risk_filters():
    return {"purposes": [int(value) for value in sorted(customers().purpose.dropna().unique())]}


@app.get("/api/columns")
def columns():
    return {"columns": demo()["columns"], "detail_available": list(demo()["detail"])}


@app.get("/api/columns/{name}")
def column_detail(name: str):
    column = next((row for row in demo()["columns"] if row["name"] == name), None)
    if not column:
        raise HTTPException(404, "字段不存在")
    return {"column": column, "distribution": demo()["detail"].get(name, [])}


@app.get("/api/features")
def features():
    return FEATURES


@app.get("/api/features/{name}/preview")
def feature_preview(name: str):
    if name not in FEATURES:
        raise HTTPException(404, "特征不存在")
    df = pd.read_csv(DATA / "train.csv", nrows=50000, usecols=["loanAmnt", "annualIncome", "installment", "issueDate", "earliesCreditLine", "isDefault"])
    if name == "loan_income_ratio":
        value = df.loanAmnt / df.annualIncome.replace(0, np.nan)
    elif name == "monthly_income":
        value = df.annualIncome / 12
    elif name == "installment_income_ratio":
        value = df.installment / (df.annualIncome / 12).replace(0, np.nan)
    elif name == "issue_month":
        value = pd.to_datetime(df.issueDate).dt.month
    else:
        issued = pd.to_datetime(df.issueDate)
        credit = pd.to_datetime(df.earliesCreditLine, format="%b-%Y")
        value = ((issued.dt.year - credit.dt.year) * 12 + issued.dt.month - credit.dt.month) / 12
    valid = value.replace([np.inf, -np.inf], np.nan).dropna()
    return {"feature": FEATURES[name], "sample_rows": len(df), "missing_pct": float(value.isna().mean() * 100), "p01": float(valid.quantile(.01)), "median": float(valid.median()), "p99": float(valid.quantile(.99)), "mean_good": float(value[df.isDefault == 0].mean()), "mean_bad": float(value[df.isDefault == 1].mean())}


@app.get("/api/models")
def models():
    with jobs_lock:
        completed = [job.copy() for job in jobs.values() if job["status"] == "completed"]
    report = json.loads((OUT / "metrics.json").read_text(encoding="utf-8"))
    return {"baseline": {"algorithm": "XGBoost", "scope": "同一 16 万条留出验证", **evaluation()},
            "comparison": [{"algorithm": name, "purpose": "当前主模型" if name == "XGBoost" else "透明基准",
                            **scores} for name, scores in report["models"].items()],
            "selected_rounds": report["validation_iterations"], "experiments": completed}


@app.get("/api/shap/global")
def global_shap():
    return shap_summary()


class TrainingRequest(BaseModel):
    algorithm: str = Field(pattern="^(hist_gradient_boosting|logistic|random_forest)$")
    sample_rows: int = Field(default=30000, ge=5000, le=50000)


def run_training(job_id: str, request: TrainingRequest) -> None:
    try:
        frame = read_data_subset(request.sample_rows)
        x, _ = make_features(frame, frame.drop(columns="isDefault"))
        y = frame.isDefault.astype("int8")
        a, b = train_test_split(np.arange(len(y)), test_size=.2, random_state=42, stratify=y)
        if request.algorithm == "logistic":
            model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=300, random_state=42))
        elif request.algorithm == "random_forest":
            model = make_pipeline(SimpleImputer(strategy="median"), RandomForestClassifier(n_estimators=60, max_depth=12, n_jobs=1, random_state=42))
        else:
            model = HistGradientBoostingClassifier(max_iter=100, max_leaf_nodes=31, random_state=42)
        model.fit(x.iloc[a], y.iloc[a])
        probabilities = model.predict_proba(x.iloc[b])[:, 1]
        auc = float(roc_auc_score(y.iloc[b], probabilities))
        fpr, tpr, _ = roc_curve(y.iloc[b], probabilities)
        with jobs_lock:
            jobs[job_id].update(status="completed", auc=auc, ks=float(np.max(tpr - fpr)), pr_auc=float(average_precision_score(y.iloc[b], probabilities)))
    except Exception as exc:
        with jobs_lock:
            jobs[job_id].update(status="failed", error=str(exc)[:200])


def read_data_subset(rows: int) -> pd.DataFrame:
    columns = pd.read_csv(DATA / "train.csv", nrows=0).columns
    text = {"grade", "subGrade", "employmentLength", "issueDate", "earliesCreditLine"}
    dtypes = {name: ("int32" if name == "id" else "int8" if name == "isDefault" else str if name in text else "float32") for name in columns}
    return pd.read_csv(DATA / "train.csv", nrows=rows, dtype=dtypes)


@app.post("/api/training/jobs")
def create_training_job(request: TrainingRequest):
    with jobs_lock:
        if any(job["status"] in {"queued", "running"} for job in jobs.values()):
            raise HTTPException(409, "已有训练任务正在运行")
        job_id = uuid.uuid4().hex[:10]
        jobs[job_id] = {"id": job_id, "algorithm": request.algorithm, "sample_rows": request.sample_rows, "status": "running"}
    executor.submit(run_training, job_id, request)
    return jobs[job_id]


@app.get("/api/training/jobs/{job_id}")
def get_training_job(job_id: str):
    with jobs_lock:
        if job_id not in jobs:
            raise HTTPException(404, "任务不存在")
        return jobs[job_id].copy()


@app.get("/api/threshold")
def threshold(value: float = .35, bad_cost: float = 20000, good_cost: float = 800):
    if not 0 <= value <= 1 or not 0 <= bad_cost <= 1e8 or not 0 <= good_cost <= 1e8:
        raise HTTPException(422, "阈值或成本超出范围")
    return threshold_metrics(value, bad_cost, good_cost)


@app.get("/api/customers/{customer_id}")
def customer(customer_id: int):
    table = customers()
    if customer_id not in table.index:
        raise HTTPException(404, "仅支持查询测试集 A 的客户 ID")
    row = table.loc[customer_id]
    position = int(table.index.get_loc(customer_id))
    probability = float(row.probability)
    result = {"customer": {name: (None if pd.isna(value) else value.item() if hasattr(value, "item") else value) for name, value in row.items()}, "risk_level": "高" if probability >= .35 else "中" if probability >= .15 else "低"}
    result["shap"] = explain_customer(position)
    return result


@app.get("/api/portfolio")
def portfolio(dimension: str = "grade", risk: str = "all", limit: int = 30):
    groups = demo()["groups"]
    if dimension not in groups:
        raise HTTPException(422, "不支持的分组维度")
    table = customers()
    filtered = table if risk == "all" else table[table.probability.ge(.35) if risk == "high" else table.probability.lt(.15) if risk == "low" else table.probability.between(.15, .35, inclusive="left")]
    return {"dimension": dimension, "groups": groups[dimension], "risk": risk, "customer_count": len(filtered), "sample_customers": [{"id": int(row.id), "grade": row.grade, "probability": float(row.probability)} for row in filtered.nlargest(min(limit, 100), "probability").itertuples()]}


@app.get("/api/portfolio/export")
def export_portfolio(risk: str = "all"):
    if risk not in {"all", "high", "medium", "low"}:
        raise HTTPException(422, "不支持的风险等级")
    table = customers()
    score = table.probability
    filtered = table if risk == "all" else table[score.ge(.35) if risk == "high" else score.lt(.15) if risk == "low" else score.between(.15, .35, inclusive="left")]
    content = filtered[["id", "grade", "loanAmnt", "annualIncome", "probability"]].to_csv(index=False)
    return StreamingResponse(iter([content]), media_type="text/csv; charset=utf-8", headers={"Content-Disposition": f'attachment; filename="risklens_{risk}.csv"'})


@app.get("/api/monitor")
def monitor():
    return {"drift": demo()["drift"], "reference": "80 万训练集", "current": "20 万测试集 A", "note": "PSI 比较输入特征分布。测试集无标签，无法计算线上 AUC 或违约率趋势。"}


class AnalystRequest(BaseModel):
    question: str = Field(min_length=2, max_length=500)


def analyst_evidence(question: str) -> tuple[str, dict, str, dict]:
    q = question.upper()
    if any(word in question for word in ["阈值", "误伤", "漏掉", "成本", "审核策略"]):
        values = re.findall(r"(?<!\d)(?:0?\.\d+|1\.0+)(?!\d)", question)
        thresholds = [float(value) for value in values[:2] if 0 <= float(value) <= 1]
        if not thresholds:
            thresholds = [.35]
        evidence = {"source": "16 万条有标签留出验证集", "scenarios": [threshold_metrics(value, 20000, 800) for value in thresholds]}
        return "策略工具", evidence, "逐个阈值计算审核量、识别率、误伤率与假设成本", {"actions": [{"label": "打开策略模拟", "page": "strategy", "threshold": thresholds[-1]}]}

    customer_match = re.search(r"(?:客户|ID|#)\s*#?(\d{6,})", question, re.IGNORECASE)
    if customer_match:
        identifier = int(customer_match.group(1))
        table = customers()
        if identifier in table.index:
            row = table.loc[identifier]
            explanation = explain_customer(int(table.index.get_loc(identifier)))
            drivers = [x for x in explanation["contributions"] if x["contribution"] > 0 and x["feature"] in BUSINESS_FEATURE_LABELS][:3]
            listed = "、".join(f"{BUSINESS_FEATURE_LABELS[x['feature']]} +{x['contribution']*100:.1f} 个百分点" for x in drivers)
            local = f"客户 #{identifier} 的 XGBoost 预测违约概率为 {explanation['prediction']:.2%}。可解读字段的模型风险贡献：{listed or '未见明显上升项'}。匿名字段不推断业务含义；SHAP 不证明违约原因，最终由人工审核。"
            evidence = {"source": "本地测试集 A / XGBoost / SHAP", "customer_id": identifier,
                        "probability": explanation["prediction"], "base_probability": explanation["base_probability"],
                        "top_contributors": drivers, "note": explanation["note"]}
            return "客户预测 → SHAP 工具", evidence, "本地预测并计算概率空间 SHAP；单客户数据不发送在线模型", {"local_answer": local, "actions": [{"label": "查看风险解释", "page": "customer", "customer_id": identifier}]}
        return "客户工具", {"source": "本地测试集 A", "found": False}, "按客户 ID 查询", {"local_answer": "未在测试集 A 找到该客户 ID。"}

    grade_reason = re.search(r"([A-G])\s*级", q)
    if grade_reason and any(word in question for word in ["为什么", "驱动", "原因", "解释"]):
        grade = grade_reason.group(1)
        cohort = shap_summary()["cohorts"].get(grade)
        if cohort:
            evidence = {"source": "测试集 A 固定样本 / XGBoost / 概率空间 SHAP", "grade": grade,
                        "cohort": cohort, "base_probability": shap_summary()["base_probability"],
                        "caveat": shap_summary()["note"]}
            return "客群 SHAP 工具", evidence, "聚合该等级抽样客户的平均 SHAP 模型贡献", {"actions": [{"label": "查看模型风险洞察", "page": "models"}]}

    grade_pair = re.search(r"([A-G])\s*[/、和及]\s*([A-G])\s*级?", q)
    grades = list(dict.fromkeys(grade_pair.groups())) if grade_pair else re.findall(r"([A-G])\s*级", q)
    if not grades:
        grade_match = re.search(r"GRADE\s*([A-G])", q)
        grades = [grade_match.group(1)] if grade_match else []
    pd_match = re.search(r"(?:PD|违约概率)\s*(?:>|＞|大于|超过)\s*(0?\.\d+)", q)
    if pd_match or (len(grades) > 1 and any(word in question for word in ["筛", "找", "客户"])):
        cutoff = float(pd_match.group(1)) if pd_match else .35
        if not 0 <= cutoff <= 1:
            return "客群工具", {"error": "PD 阈值应在 0 到 1 之间"}, "检查筛选条件", {"local_answer": "PD 阈值应在 0 到 1 之间。"}
        table = customers()
        filtered = table[table.probability.gt(cutoff)]
        if grades:
            filtered = filtered[filtered.grade.isin(grades)]
        summary = filtered.groupby("grade").probability.agg(["size", "mean"])
        evidence = {"source": "20 万条无标签测试集 A 的模型预测", "grades": grades or "全部", "pd_above": cutoff, "count": len(filtered), "local_examples_below": min(10, len(filtered)), "by_grade": [{"grade": str(index), "count": int(row["size"]), "average_pd": float(row["mean"])} for index, row in summary.iterrows()]}
        matches = [{"id": int(row.id), "grade": row.grade, "probability": float(row.probability)} for row in filtered.nlargest(10, "probability").itertuples()]
        return "客群工具", evidence, "按信用等级与模型预测概率筛选测试集 A", {"matches": matches, "actions": [{"label": "查看风险总览", "page": "overview"}]}

    if any(word in question for word in ["为什么高风险", "高风险客户这么多", "风险因素", "风险原因"]):
        view = risk_overview_data()
        evidence = {"source": "20 万条无标签测试集 A 的模型预测 / 抽样 SHAP", "count": view["count"],
                    "high_count": view["high_count"], "high_share": view["high_share"],
                    "shap_cohort": shap_summary()["cohorts"].get("high_risk"), "caveat": shap_summary()["note"]}
        return "组合风险 → SHAP 工具", evidence, "汇总风险占比与高风险抽样客户的平均模型贡献", {"actions": [{"label": "查看风险总览", "page": "overview"}]}

    if any(word in question for word in ["哪个客群", "风险最高", "最近"]):
        group = customers().groupby("grade").probability.agg(["size", "mean"]).sort_values("mean", ascending=False)
        evidence = {"source": "20 万条无标签测试集 A 的模型预测", "groups": [{"grade": str(index), "count": int(row["size"]), "average_pd": float(row["mean"])} for index, row in group.iterrows()], "caveat": "没有新近生产数据和测试集真实标签，不能判断最近风险变化。"}
        return "客群工具", evidence, "按信用等级比较测试集 A 的模型平均风险", {"actions": [{"label": "查看风险总览", "page": "overview"}]}

    match = re.search(r"([A-G])\s*级|GRADE\s*([A-G])", q)
    if match:
        grade = (match.group(1) or match.group(2)).upper()
        item = next((row for row in demo()["groups"]["grade"] if row["label"] == grade), None)
        if item:
            return "客群工具", {"source": "80 万条有标签训练集", "grade": item, "overall_default_rate": demo()["default_rate"]}, "按信用等级计算历史样本量与违约率", {"actions": [{"label": "查看风险总览", "page": "overview", "grade": grade}]}
    if any(word in question for word in ["漂移", "稳定", "PSI", "psi", "模型状态"]):
        return "模型健康工具", {"source": "80 万训练集 vs 20 万无标签测试集 A", "drift": demo()["drift"]}, "比较输入变量分布 PSI", {"actions": [{"label": "查看模型健康", "page": "monitor"}]}
    view = risk_overview_data()
    return "组合风险工具", {"source": "20 万条无标签测试集 A 的模型预测", "count": view["count"], "distribution": view["distribution"], "high_share": view["high_share"]}, "汇总当前模型风险分层", {"actions": [{"label": "查看风险总览", "page": "overview"}]}


def call_deepseek(question: str, evidence: dict) -> str:
    key = os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        raise RuntimeError("未配置 DEEPSEEK_API_KEY")
    payload = {
        "model": os.environ.get("DEEPSEEK_MODEL", "deepseek-flash"),
        "messages": [
            {"role": "system", "content": "你是金融风控数据分析助手。只依据提供的聚合证据回答，使用中文，明确数据来源，百分比保留两位小数。SHAP 贡献表示对模型预测的推动或抵消，绝不能写成导致违约的原因，也不能编造特征贡献。local_examples_below 表示界面下方另有本地可点击客户示例，不要说无法列出名单，也不要编造 ID。不得编造其他数值、因果结论或线上表现；证据不足时明确说不知道。回答控制在150字以内。"},
            {"role": "user", "content": json.dumps({"question": question, "evidence": evidence}, ensure_ascii=False)},
        ],
        "temperature": 0.2,
        "max_tokens": 300,
        "stream": False,
    }
    request = urllib.request.Request(
        "https://api.deepseek.com/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            result = json.load(response)
        return result["choices"][0]["message"]["content"].strip()
    except (urllib.error.URLError, KeyError, IndexError, ValueError) as exc:
        raise RuntimeError(f"DeepSeek 调用失败：{type(exc).__name__}") from None


@app.post("/api/analyst")
async def analyst(request: AnalystRequest):
    route, evidence, calculation, ui = analyst_evidence(request.question)
    if "local_answer" in ui:
        answer, mode = ui["local_answer"], "local_only"
    else:
        try:
            answer = await asyncio.to_thread(call_deepseek, request.question, evidence)
            mode = "deepseek"
        except RuntimeError as exc:
            answer = f"已完成本地指标查询；在线总结暂不可用（{exc}）。请查看下方证据。"
            mode = "evidence_only"
    return {"answer": answer, "mode": mode, "route": route, "calculation": calculation, "evidence": evidence, "actions": ui.get("actions", []), "matches": ui.get("matches", [])}


@app.post("/api/upload-preview")
async def upload_preview(file: UploadFile = File(...)):
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in {".csv", ".parquet"}:
        raise HTTPException(422, "仅支持 CSV 或 Parquet")
    content = await file.read(25 * 1024 * 1024 + 1)
    if len(content) > 25 * 1024 * 1024:
        raise HTTPException(413, "演示预览上限为 25 MB")
    try:
        frame = pd.read_csv(io.BytesIO(content), nrows=10000) if suffix == ".csv" else pd.read_parquet(io.BytesIO(content)).head(10000)
    except Exception:
        raise HTTPException(422, "文件解析失败") from None
    return {"filename": file.filename, "preview_rows": len(frame), "columns": [{"name": name, "type": str(frame[name].dtype), "missing_pct": round(float(frame[name].isna().mean() * 100), 2), "unique": int(frame[name].nunique())} for name in frame.columns], "note": "仅预览前 1 万行，不保存上传文件，也不会替换内置天池数据集。"}
