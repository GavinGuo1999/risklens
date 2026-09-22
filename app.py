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
    return (joblib.load(OUT / "model.joblib"), pd.read_parquet(OUT / "test_features.parquet"), json.loads((OUT / "feature_medians.json").read_text()))


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
        "estimated_cost": fn * bad_cost + fp * good_cost,
        "bad_cost": bad_cost, "good_cost": good_cost,
        "basis": "16 万条留出验证集；成本为输入假设，不代表真实贷款损失",
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
    return {"baseline": {"algorithm": "HistGradientBoosting", "scope": "80 万条训练 / 16 万条留出验证", **evaluation()}, "experiments": completed}


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
    model, features, medians = model_and_features()
    original = features.iloc[[position]].copy()
    candidates = ["interestRate", "dti", "ficoRangeLow", "loan_to_income", "annualIncome", "loanAmnt", "credit_history_months", "revolUtil", "term"]
    changed = pd.concat([original] * (len(candidates) + 1), ignore_index=True)
    for i, feature in enumerate(candidates, start=1):
        changed.loc[i, feature] = medians.get(feature)
    predicted = model.predict_proba(changed)[:, 1]
    result["sensitivity"] = sorted([
        {"feature": feature, "value": None if pd.isna(original.iloc[0][feature]) else float(original.iloc[0][feature]), "reference": medians.get(feature), "delta": float(predicted[0] - predicted[i])}
        for i, feature in enumerate(candidates, start=1)
    ], key=lambda item: abs(item["delta"]), reverse=True)[:6]
    result["explanation_method"] = "单变量替换敏感度：将该特征替换成训练集中位数，观察模型概率变化；不是 SHAP，也不表示因果关系。"
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


def analyst_evidence(question: str) -> tuple[str, dict, str]:
    match = re.search(r"([A-Ga-g])\s*级|grade\s*([A-Ga-g])", question, re.IGNORECASE)
    if match:
        grade = (match.group(1) or match.group(2)).upper()
        item = next((row for row in demo()["groups"]["grade"] if row["label"] == grade), None)
        if item:
            return "客群查询", {"source": "80 万条有标签训练集", "grade": item, "overall_default_rate": demo()["default_rate"]}, "从训练集按 grade 分组，计算样本量与历史违约率"
    if any(word in question for word in ["漂移", "稳定", "PSI", "psi"]):
        return "分布漂移", {"source": "80 万训练集 vs 20 万无标签测试集 A", "drift": demo()["drift"]}, "对训练集与测试集 A 的输入变量计算 PSI"
    if any(word in question for word in ["阈值", "误伤", "漏掉", "成本"]):
        return "阈值模拟", {"source": "16 万条有标签留出验证集", **threshold_metrics(.35, 20000, 800)}, "在留出验证集上计算 TP、FP、FN、TN 和假设成本"
    return "整体概览", {"source": "训练集标签与留出验证结果", "training_rows": demo()["train_rows"], "default_rate": demo()["default_rate"], "validation_auc": evaluation()["auc"], "grade": demo()["groups"]["grade"][:7]}, "汇总训练标签、留出验证 AUC 与等级分组"


def call_deepseek(question: str, evidence: dict) -> str:
    key = os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        raise RuntimeError("未配置 DEEPSEEK_API_KEY")
    payload = {
        "model": os.environ.get("DEEPSEEK_MODEL", "deepseek-flash"),
        "messages": [
            {"role": "system", "content": "你是金融风控数据分析助手。只依据提供的聚合证据回答，使用中文，明确数据来源，百分比保留两位小数。不得编造数值、因果结论或线上表现；证据不足时明确说不知道。回答控制在150字以内。"},
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
    route, evidence, calculation = analyst_evidence(request.question)
    try:
        answer = await asyncio.to_thread(call_deepseek, request.question, evidence)
        mode = "deepseek"
    except RuntimeError as exc:
        answer = f"已完成本地指标查询；在线总结暂不可用（{exc}）。请查看下方证据。"
        mode = "evidence_only"
    return {"answer": answer, "mode": mode, "route": route, "calculation": calculation, "evidence": evidence}


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
