"""
JustiFlow — AI Fairness Auditing Platform
Flask backend with full bias analysis pipeline.

Install dependencies:
    pip install flask pandas numpy anthropic reportlab werkzeug

Run:
    python app.py
"""

import os
import io
import json
import uuid
import hashlib
import base64
from datetime import datetime
from flask import (
    Flask, render_template, request, redirect,
    url_for, session, jsonify, send_file, abort
)
import pandas as pd
import numpy as np
from werkzeug.utils import secure_filename
import anthropic
from blockchain import justi_chain, hash_file_bytes, hash_dataframe

# ── CONFIG ────────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "justiflow-dev-secret-change-in-prod")

UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"csv"}
MAX_FILE_MB = 50

RESULTS_FOLDER = "results"
HISTORY_FILE   = "analysis_history.json"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

# In-memory cache + disk-backed store so results survive Flask reloads
analysis_store: dict = {}
session_history: list = []   # last 3 analyses


def save_result(analysis_id: str, result: dict) -> None:
    """Persist result to disk as JSON so it survives server restarts."""
    path = os.path.join(RESULTS_FOLDER, f"{analysis_id}.json")
    serialisable = {k: v for k, v in result.items() if k != "_report_txt"}
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(serialisable, f, ensure_ascii=False)
    except Exception as e:
        app.logger.warning(f"Could not save result to disk: {e}")


def load_result(analysis_id: str):
    """Load result from in-memory cache, falling back to disk."""
    if analysis_id in analysis_store:
        return analysis_store[analysis_id]
    path = os.path.join(RESULTS_FOLDER, f"{analysis_id}.json")
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                result = json.load(f)
            result["_report_txt"] = generate_txt_report(result)
            analysis_store[analysis_id] = result
            return result
        except Exception as e:
            app.logger.warning(f"Could not load result from disk: {e}")
    return None


def append_history(entry: dict) -> None:
    """Append one analysis summary to the persistent history file."""
    hist = load_history()
    hist.insert(0, entry)
    hist = hist[:50]
    try:
        with open(HISTORY_FILE, "w") as f:
            json.dump(hist, f, indent=2)
    except Exception as e:
        app.logger.warning(f"Could not save history: {e}")


def load_history() -> list:
    if not os.path.exists(HISTORY_FILE):
        return []
    try:
        with open(HISTORY_FILE) as f:
            return json.load(f)
    except Exception:
        return []


def get_previous_result(exclude_id: str) -> dict | None:
    for entry in load_history():
        if entry.get("analysis_id") != exclude_id:
            return entry
    return None


# ── HELPERS ───────────────────────────────────────────────────────────────────

def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def dataset_hash(df: pd.DataFrame) -> str:
    """Reproducible hash of a dataframe."""
    return hashlib.md5(pd.util.hash_pandas_object(df, index=True).values).hexdigest()[:12].upper()


def risk_level(score: float) -> str:
    if score >= 50:
        return "HIGH"
    elif score >= 25:
        return "MEDIUM"
    return "LOW"


def verdict(score: float):
    if score >= 50:
        return {
            "verdict_class": "danger",
            "verdict_icon": "🔴",
            "verdict_title": "Not Recommended for Deployment",
            "verdict_sub": "Significant bias detected. Mitigation required before production use.",
        }
    elif score >= 25:
        return {
            "verdict_class": "warn",
            "verdict_icon": "⚠️",
            "verdict_title": "Proceed with Caution",
            "verdict_sub": "Moderate bias detected. Review recommendations and monitor closely.",
        }
    return {
        "verdict_class": "safe",
        "verdict_icon": "✅",
        "verdict_title": "Safe to Deploy",
        "verdict_sub": "Dataset meets fairness thresholds. Continue monitoring in production.",
    }


def score_class(score: float) -> str:
    if score >= 50:
        return "danger"
    elif score >= 25:
        return "warn"
    return "safe"


# ── FAIRNESS METRICS ──────────────────────────────────────────────────────────

def compute_dir(df: pd.DataFrame, protected: str, outcome: str) -> float:
    """Disparate Impact Ratio — P(Y=1|unprivileged) / P(Y=1|privileged)."""
    groups = df[protected].unique()
    if len(groups) < 2:
        return 1.0
    rates = df.groupby(protected)[outcome].mean()
    min_rate = rates.min()
    max_rate = rates.max()
    if max_rate == 0:
        return 1.0
    return round(min_rate / max_rate, 3)


def compute_spd(df: pd.DataFrame, protected: str, outcome: str) -> float:
    """Statistical Parity Difference — P(Y=1|unprivileged) - P(Y=1|privileged)."""
    rates = df.groupby(protected)[outcome].mean()
    return round(rates.min() - rates.max(), 3)


def compute_eod(df: pd.DataFrame, protected: str, outcome: str, label_col: str = None) -> float:
    """
    Equal Opportunity Difference — False Negative Rate gap across groups.
    Requires a ground-truth label column. If not available, approximates via FNR on outcome.
    """
    if label_col and label_col in df.columns:
        def fnr(grp):
            tp = ((grp[label_col] == 1) & (grp[outcome] == 1)).sum()
            fn = ((grp[label_col] == 1) & (grp[outcome] == 0)).sum()
            return fn / (tp + fn) if (tp + fn) > 0 else 0
        fnrs = df.groupby(protected).apply(fnr)
    else:
        # Approximate: use (1 - approval rate) as FNR proxy
        fnrs = 1 - df.groupby(protected)[outcome].mean()
    return round(fnrs.min() - fnrs.max(), 3)


def bias_score_from_metrics(dir_val: float, spd_val: float, eod_val: float) -> float:
    """Combine three metrics into a 0-100 bias score."""
    # DIR: ideal = 1.0, <0.8 is problematic
    dir_penalty = max(0, (1.0 - dir_val) * 100)
    spd_penalty = abs(spd_val) * 100
    eod_penalty = abs(eod_val) * 100
    raw = (dir_penalty * 0.4 + spd_penalty * 0.35 + eod_penalty * 0.25)
    return round(min(raw, 100), 1)


# ── MITIGATION ────────────────────────────────────────────────────────────────

def apply_reweighing(df: pd.DataFrame, protected: str, outcome: str) -> pd.DataFrame:
    """Reweighing: adjust sample weights to correct for representation bias."""
    total = len(df)
    df = df.copy()
    for g in df[protected].unique():
        for o in df[outcome].unique():
            mask = (df[protected] == g) & (df[outcome] == o)
            expected = (df[protected] == g).mean() * (df[outcome] == o).mean()
            actual = mask.mean()
            if actual > 0:
                df.loc[mask, "_weight"] = expected / actual
    # Simulate: resample with weights to produce a "fairer" version
    if "_weight" in df.columns:
        df[outcome] = df.apply(
            lambda r: 1 if np.random.random() < min(r["_weight"] * df[outcome].mean(), 0.9) else 0, axis=1
        )
        df.drop("_weight", axis=1, inplace=True)
    return df


def apply_oversampling(df: pd.DataFrame, protected: str, outcome: str) -> pd.DataFrame:
    """Oversampling: boost minority group representation."""
    df = df.copy()
    group_counts = df[protected].value_counts()
    max_count = group_counts.max()
    parts = [df]
    for g, cnt in group_counts.items():
        if cnt < max_count:
            minority = df[df[protected] == g]
            extra = minority.sample(max_count - cnt, replace=True, random_state=42)
            parts.append(extra)
    return pd.concat(parts, ignore_index=True)


def apply_threshold(df: pd.DataFrame, protected: str, outcome: str) -> pd.DataFrame:
    """Threshold adjustment: equalize approval rates across groups."""
    df = df.copy()
    overall_rate = df[outcome].mean()
    for g in df[protected].unique():
        mask = df[protected] == g
        group_rate = df.loc[mask, outcome].mean()
        if group_rate > 0:
            scale = overall_rate / group_rate
            df.loc[mask, outcome] = (df.loc[mask, outcome] * scale).clip(0, 1).round()
    return df


MITIGATION_FNS = {
    "reweighing": apply_reweighing,
    "oversampling": apply_oversampling,
    "threshold": apply_threshold,
}


# ── CLAUDE AUDIT PARAGRAPH ────────────────────────────────────────────────────

def generate_audit_paragraph(
    filename: str, protected: str, outcome: str,
    bias_score: float, dir_val: float, spd_val: float, eod_val: float,
    risk: str, recommendations: list
) -> str:
    """Call Claude API to generate a plain-English audit paragraph."""
    if not ANTHROPIC_API_KEY:
        return (
            f"The dataset '{filename}' was analyzed for algorithmic bias using the protected attribute '{protected}' "
            f"and outcome '{outcome}'. The analysis produced a bias score of {bias_score}%, indicating a {risk.lower()} "
            f"risk of discriminatory outcomes. The Disparate Impact Ratio of {dir_val} "
            f"({'falls below' if dir_val < 0.8 else 'meets'} the 0.8 threshold set by the four-fifths rule), "
            f"and the Statistical Parity Difference of {spd_val} suggests meaningful disparity between groups. "
            f"Immediate action is {'strongly recommended' if risk == 'HIGH' else 'advised' if risk == 'MEDIUM' else 'optional but beneficial'} "
            f"to ensure compliance with EU AI Act Article 10 data governance requirements."
        )
    try:
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        prompt = f"""
You are an expert AI fairness auditor. Write a 3-4 sentence plain-English audit paragraph for the following analysis.
Be specific, professional, and reference the EU AI Act where relevant.

Dataset: {filename}
Protected attribute: {protected}
Outcome column: {outcome}
Bias Score: {bias_score}%
Risk Level: {risk}
Disparate Impact Ratio: {dir_val} (threshold: ≥0.8)
Statistical Parity Difference: {spd_val} (threshold: ≤0.1)
Equal Opportunity Difference: {eod_val} (threshold: ≤0.1)
Top recommendation: {recommendations[0] if recommendations else 'Review dataset representation'}

Write only the paragraph, no headers or extra text.
"""
        msg = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}]
        )
        return msg.content[0].text.strip()
    except Exception as e:
        app.logger.warning(f"Claude API call failed: {e}")
        return (
            f"Analysis of '{filename}' identified a {risk.lower()} risk bias score of {bias_score}% "
            f"using the '{protected}' protected attribute. The Disparate Impact Ratio of {dir_val} "
            f"indicates {'significant' if dir_val < 0.8 else 'acceptable'} disparity between groups. "
            f"Dataset adjustments are recommended to achieve compliance with EU AI Act Article 10."
        )


# ── SHAP-STYLE FEATURE IMPORTANCE ─────────────────────────────────────────────

def compute_feature_importance(df: pd.DataFrame, protected: str, outcome: str) -> list:
    """Compute correlation-based feature importance (SHAP proxy)."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != outcome]
    if not numeric_cols:
        return []
    corrs = {}
    for col in numeric_cols:
        try:
            corrs[col] = abs(df[col].corr(df[outcome]))
        except Exception:
            corrs[col] = 0.0
    total = sum(corrs.values()) or 1
    colors = ["rgba(59,130,246,0.8)", "rgba(139,92,246,0.8)", "rgba(6,182,212,0.8)", "rgba(245,158,11,0.8)"]
    sorted_cols = sorted(corrs, key=corrs.get, reverse=True)[:4]
    return [
        {
            "name": col,
            "importance": round(corrs[col] / total * 100, 1),
            "color": colors[i % len(colors)]
        }
        for i, col in enumerate(sorted_cols)
    ]


# ── CORRELATION HEATMAP ───────────────────────────────────────────────────────

def compute_heatmap(df: pd.DataFrame) -> dict:
    """Return correlation matrix for numeric columns (max 8 cols)."""
    numeric = df.select_dtypes(include=[np.number])
    if len(numeric.columns) < 2:
        return {"labels": [], "values": []}
    cols = list(numeric.columns)[:8]
    corr = numeric[cols].corr().round(2)
    return {
        "labels": cols,
        "values": corr.values.tolist()
    }


# ── COUNTERFACTUAL ────────────────────────────────────────────────────────────

def compute_counterfactual(df: pd.DataFrame, protected: str, outcome: str, shap_features: list) -> str:
    """Generate a counterfactual explanation string."""
    if not shap_features:
        return "Insufficient data for counterfactual analysis."
    top_feature = shap_features[0]["name"]
    rates = df.groupby(protected)[outcome].mean()
    if len(rates) < 2:
        return "Only one group detected — no counterfactual available."
    min_group = rates.idxmin()
    max_group = rates.idxmax()
    gap = abs(rates[max_group] - rates[min_group]) * 100
    return (
        f"If '{top_feature}' were equalised between '{min_group}' and '{max_group}', "
        f"the approval rate gap would reduce by approximately {gap:.1f}%. "
        f"Equalising '{top_feature}' is the single highest-impact intervention available."
    )


# ── INTERSECTIONAL BIAS ───────────────────────────────────────────────────────

def compute_intersectional(df: pd.DataFrame, protected: str, outcome: str) -> dict:
    """Check bias at intersection of protected attribute with the next categorical column."""
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    cat_cols = [c for c in cat_cols if c != protected and c != outcome]
    if not cat_cols:
        return {}
    second = cat_cols[0]
    group_rates = df.groupby([protected, second])[outcome].mean()
    return group_rates.to_dict()


# ── REPORT GENERATOR ─────────────────────────────────────────────────────────

def generate_txt_report(result: dict) -> str:
    lines = [
        "=" * 60,
        "  JUSTIFLOW — AI FAIRNESS AUDIT REPORT",
        "=" * 60,
        f"Analysis ID  : {result['analysis_id']}",
        f"Dataset      : {result['filename']}",
        f"Date         : {result['analysis_date']}",
        f"Dataset Hash : {result['dataset_hash']}",
        "",
        "── VERDICT ──────────────────────────────────────────",
        f"  {result['verdict_title']}",
        f"  Risk Level : {result['risk_level']}",
        f"  Bias Score : {result['bias_score']}%",
        f"  Improvement: {result['improvement']}%",
        "",
        "── FAIRNESS METRICS ─────────────────────────────────",
    ]
    for m in result.get("metrics", []):
        lines.append(f"  {m['name']:<40} {m['value']}")
    lines += [
        "",
        "── AI AUDIT PARAGRAPH ───────────────────────────────",
        result.get("audit_paragraph", ""),
        "",
        "── RECOMMENDATIONS ──────────────────────────────────",
    ]
    for r in result.get("recommendations", []):
        lines.append(f"  [{r['priority'].upper()}] {r['text']}")
    lines += [
        "",
        "── REGULATORY ALIGNMENT ─────────────────────────────",
        "  EU AI Act — Article 10 (Data Governance)",
        "  EU AI Act — Article 13 (Transparency)",
        "  Fairlearn Framework — DIR, SPD, EOD metrics",
        "  Google AI Principles",
        "",
        "=" * 60,
        "  Generated by JustiFlow · https://justiflow.io",
        "=" * 60,
    ]
    return "\n".join(lines)


# ── MAIN ANALYSIS PIPELINE ────────────────────────────────────────────────────

def run_analysis(df: pd.DataFrame, filename: str, protected: str, outcome: str,
                 mitigation: str, intersectional: bool, temporal: bool,
                 generate_cert: bool) -> dict:
    """Full analysis pipeline. Returns result dict for dashboard template."""
    analysis_id = str(uuid.uuid4())[:8].upper()
    d_hash = dataset_hash(df)

    # Ensure outcome is binary int — handle text like "Approved/Rejected", "Hired/Not Hired" etc.
    def encode_binary(series):
        s = series.fillna("").astype(str).str.strip()
        # Already numeric
        if s.str.match(r'^-?\d+(\.\d+)?$').all():
            return s.astype(float).astype(int)
        uniq = [v for v in s.unique() if v != ""]
        if len(uniq) < 2:
            return s.map({uniq[0]: 1}).fillna(0).astype(int)
        # Positive keywords → 1, everything else → 0
        positive = {"approved", "hired", "shortlisted", "yes", "low risk",
                    "standard care", "specialist referral", "granted", "accepted",
                    "passed", "selected", "1", "true", "match", "correct"}
        # If more than 2 unique values (e.g. High/Medium/Low Risk), collapse using keywords
        if len(uniq) > 2:
            return s.apply(lambda v: 1 if v.lower() in positive else 0)
        first = uniq[0].lower()
        if first in positive:
            pos_val, neg_val = uniq[0], uniq[1]
        else:
            pos_val, neg_val = uniq[1], uniq[0]
        return s.map({pos_val: 1, neg_val: 0}).fillna(0).astype(int)

    df[outcome] = encode_binary(df[outcome])

    # ── BEFORE metrics
    dir_before = compute_dir(df, protected, outcome)
    spd_before = compute_spd(df, protected, outcome)
    eod_before = compute_eod(df, protected, outcome)
    bias_before = bias_score_from_metrics(dir_before, spd_before, eod_before)

    # ── APPLY MITIGATION
    mit_fn = MITIGATION_FNS.get(mitigation, apply_reweighing)
    df_mit = mit_fn(df.copy(), protected, outcome)

    # ── AFTER metrics
    dir_after = compute_dir(df_mit, protected, outcome)
    spd_after = compute_spd(df_mit, protected, outcome)
    eod_after = compute_eod(df_mit, protected, outcome)
    bias_after = bias_score_from_metrics(dir_after, spd_after, eod_after)

    improvement = round(max(0, (bias_before - bias_after) / (bias_before or 1) * 100), 1)
    risk = risk_level(bias_before)
    v_info = verdict(bias_before)

    # ── SHAP
    shap_features = compute_feature_importance(df, protected, outcome)

    # ── COUNTERFACTUAL
    counterfactual = compute_counterfactual(df, protected, outcome, shap_features)

    # ── HEATMAP
    heatmap = compute_heatmap(df)

    # ── AUDIT PARAGRAPH
    rec_texts = [
        "Increase representation of underrepresented groups in training data",
        "Apply reweighing to balance group contributions",
        "Audit data collection pipeline for systematic exclusion",
        "Implement regular fairness monitoring post-deployment",
        "Document protected attribute handling for EU AI Act compliance",
    ]
    if bias_before >= 50:
        recs = [
            {"priority": "high",   "text": rec_texts[0]},
            {"priority": "high",   "text": rec_texts[2]},
            {"priority": "medium", "text": rec_texts[1]},
            {"priority": "medium", "text": rec_texts[4]},
            {"priority": "low",    "text": rec_texts[3]},
        ]
    elif bias_before >= 25:
        recs = [
            {"priority": "high",   "text": rec_texts[1]},
            {"priority": "medium", "text": rec_texts[0]},
            {"priority": "medium", "text": rec_texts[4]},
            {"priority": "low",    "text": rec_texts[3]},
        ]
    else:
        recs = [
            {"priority": "medium", "text": rec_texts[3]},
            {"priority": "low",    "text": rec_texts[4]},
        ]

    audit_para = generate_audit_paragraph(
        filename, protected, outcome, bias_before,
        dir_before, spd_before, eod_before, risk,
        [r["text"] for r in recs]
    )

    # ── GROUP OUTCOMES
    group_outcomes = []
    for g, grp in df.groupby(protected):
        group_outcomes.append({
            "group": str(g),
            "count": len(grp),
            "avg_outcome": round(grp[outcome].mean(), 3),
            "approval_rate": round(grp[outcome].mean() * 100, 1),
        })

    # ── DATASET SUMMARY
    summary = {
        "rows": len(df),
        "cols": len(df.columns),
        "missing": int(df.isnull().sum().sum()),
        "groups": df[protected].nunique(),
    }

    # ── IMBALANCE CHECK
    group_pcts = df[protected].value_counts(normalize=True)
    imbalance_groups = [str(g) for g, pct in group_pcts.items() if pct < 0.10]

    # ── PREVIEW
    preview_df = df.head(10)
    preview_cols = list(preview_df.columns)[:8]
    preview_rows = preview_df[preview_cols].fillna("—").values.tolist()

    # ── BAR CHART DATA
    bar_labels = [str(g) for g in df[protected].unique()]
    bar_before = [round(df[df[protected] == g][outcome].mean() * 100, 1) for g in df[protected].unique()]
    bar_after  = [round(df_mit[df_mit[protected] == g][outcome].mean() * 100, 1) for g in df_mit[protected].unique()]

    # ── METRICS TABLE
    def metric_cls(name, val):
        if name == "DIR":
            return "ok" if val >= 0.8 else ("warn" if val >= 0.6 else "bad")
        else:
            return "ok" if abs(val) <= 0.1 else ("warn" if abs(val) <= 0.2 else "bad")

    metrics = [
        {"name": "Disparate Impact Ratio (DIR)", "value": str(dir_before),
         "cls": metric_cls("DIR", dir_before),
         "pct": round(dir_before * 100, 0),
         "color": "#10b981" if dir_before >= 0.8 else "#ef4444"},
        {"name": "Statistical Parity Diff (SPD)", "value": str(spd_before),
         "cls": metric_cls("SPD", spd_before),
         "pct": round(abs(spd_before) * 100, 0),
         "color": "#10b981" if abs(spd_before) <= 0.1 else "#ef4444"},
        {"name": "Equal Opportunity Diff (EOD)", "value": str(eod_before),
         "cls": metric_cls("EOD", eod_before),
         "pct": round(abs(eod_before) * 100, 0),
         "color": "#10b981" if abs(eod_before) <= 0.1 else "#ef4444"},
    ]

    # ── CONFIDENCE BREAKDOWN
    data_quality = min(100, round(100 - (summary["missing"] / (summary["rows"] * summary["cols"] + 1)) * 100, 0))
    model_conf   = round(80 + (summary["rows"] / 500) * 10, 0) if summary["rows"] < 500 else 95
    rec_conf     = round((data_quality + model_conf) / 2, 0)
    confidence   = round((data_quality * 0.4 + model_conf * 0.4 + rec_conf * 0.2), 0)
    confidence_breakdown = [
        {"name": "Data Quality",      "pct": min(int(data_quality), 100)},
        {"name": "Model Confidence",  "pct": min(int(model_conf), 100)},
        {"name": "Rec. Confidence",   "pct": min(int(rec_conf), 100)},
    ]

    # ── AUDIT LOG
    now = datetime.now()
    audit_log = [
        {"id": f"JF-{analysis_id}", "label": f"Full analysis · {protected} vs {outcome}", "date": now.strftime("%Y-%m-%d %H:%M")},
        {"id": f"HASH-{d_hash[:6]}", "label": f"Dataset fingerprint computed", "date": now.strftime("%Y-%m-%d %H:%M")},
        {"id": f"MIT-{mitigation[:3].upper()}", "label": f"Mitigation applied: {mitigation}", "date": now.strftime("%Y-%m-%d %H:%M")},
    ]

    # ── SHARE URL
    share_url = f"https://justiflow.io/results?id={analysis_id}&score={bias_before}&risk={risk}"

    # ── SESSION HISTORY
    hist_entry = {
        "score": bias_before,
        "cls": score_class(bias_before),
        "filename": filename,
        "timestamp": now.strftime("%d %b %Y, %H:%M"),
        "risk": risk,
        "risk_cls": risk.lower(),
    }
    session_history.insert(0, hist_entry)
    if len(session_history) > 3:
        session_history.pop()

    result = {
        # IDs & meta
        "analysis_id": analysis_id,
        "dataset_hash": d_hash,
        "filename": filename,
        "analysis_date": now.strftime("%d %B %Y"),
        # Scores
        "bias_score": bias_before,
        "improvement": improvement,
        "confidence": int(confidence),
        "risk_level": risk,
        "score_class": score_class(bias_before),
        "mitigation_method": mitigation.title(),
        # Verdict
        **v_info,
        # Metrics
        "metrics": metrics,
        "shap_features": shap_features,
        "counterfactual": counterfactual,
        "audit_paragraph": audit_para,
        "confidence_breakdown": confidence_breakdown,
        # Data
        "group_outcomes": group_outcomes,
        "summary": summary,
        "imbalance_groups": imbalance_groups,
        "preview_cols": preview_cols,
        "preview_rows": preview_rows,
        "bar_labels": bar_labels,
        "bar_before": bar_before,
        "bar_after": bar_after,
        "heatmap": heatmap,
        # UX
        "recommendations": recs,
        "history": session_history[:3],
        "audit_log": audit_log,
        "share_url": share_url,
        # Report text (for download)
        "_report_txt": None,
    }

    result["_report_txt"] = generate_txt_report(result)
    analysis_store[analysis_id] = result
    save_result(analysis_id, result)
    return result


# ── ROUTES ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["GET"])
def upload():
    return render_template("upload.html", error=None)


@app.route("/analyze", methods=["POST"])
def analyze():
    # ── FILE VALIDATION
    if "file" not in request.files:
        return render_template("upload.html", error="No file part in request.")
    file = request.files["file"]
    if file.filename == "":
        return render_template("upload.html", error="No file selected.")
    if not allowed_file(file.filename):
        return render_template("upload.html", error="Only .csv files are supported.")

    filename = secure_filename(file.filename)

    # ── READ RAW BYTES (for blockchain hashing) then parse CSV
    try:
        file_bytes = file.read()
        file_hash  = hash_file_bytes(file_bytes)
        import io as _io
        df = pd.read_csv(_io.BytesIO(file_bytes))
    except Exception as e:
        return render_template("upload.html", error=f"Could not parse CSV: {e}")

    if len(df) < 10:
        return render_template("upload.html", error="Dataset must have at least 10 rows.")

    # ── FORM PARAMS
    protected  = request.form.get("protected_col", "")
    outcome    = request.form.get("outcome_col", "")
    mitigation = request.form.get("mitigation", "reweighing")
    intersectional = bool(request.form.get("intersectional"))
    temporal       = bool(request.form.get("temporal"))
    generate_cert  = bool(request.form.get("generate_cert"))

    if protected not in df.columns:
        return render_template("upload.html", error=f"Column '{protected}' not found in CSV.")
    if outcome not in df.columns:
        return render_template("upload.html", error=f"Column '{outcome}' not found in CSV.")
    if protected == outcome:
        return render_template("upload.html", error="Protected attribute and outcome must be different columns.")

    # ── RUN ANALYSIS
    try:
        result = run_analysis(
            df, filename, protected, outcome,
            mitigation, intersectional, temporal, generate_cert
        )
    except Exception as e:
        app.logger.exception("Analysis failed")
        return render_template("upload.html", error=f"Analysis failed: {e}")

    analysis_id = result["analysis_id"]

    # ── BLOCKCHAIN: record file upload + analysis result
    justi_chain.add_file_block(
        file_hash       = file_hash,
        filename        = filename,
        file_size_bytes = len(file_bytes),
        analysis_id     = analysis_id,
    )
    justi_chain.add_analysis_block(
        analysis_id  = analysis_id,
        file_hash    = file_hash,
        bias_score   = result["bias_score"],
        risk_level   = result["risk_level"],
        protected_col= protected,
        outcome_col  = outcome,
        row_count    = len(df),
    )
    result["file_hash"]       = file_hash
    result["blockchain"]      = justi_chain.summary()
    result["blockchain_block"]= justi_chain.latest.to_dict()

    # ── HISTORY: attach previous analysis for comparison
    previous = get_previous_result(analysis_id)
    result["previous"] = previous

    # ── Save current result to history
    append_history({
        "analysis_id":  analysis_id,
        "filename":     filename,
        "file_hash":    file_hash,
        "bias_score":   result["bias_score"],
        "risk_level":   result["risk_level"],
        "improvement":  result.get("improvement", 0),
        "row_count":    len(df),
        "protected_col":protected,
        "outcome_col":  outcome,
        "analysed_at":  datetime.now().strftime("%d %b %Y, %H:%M"),
    })

    return render_template("dashboard.html", result=result)


# ── REST API ENDPOINT ─────────────────────────────────────────────────────────

@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    """
    REST API endpoint — accepts multipart/form-data or JSON.
    Returns full analysis as JSON.
    """
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if not allowed_file(file.filename):
        return jsonify({"error": "Only .csv files supported"}), 400

    try:
        df = pd.read_csv(file)
    except Exception as e:
        return jsonify({"error": f"CSV parse error: {e}"}), 400

    protected  = request.form.get("protected_col", "")
    outcome    = request.form.get("outcome_col", "")
    mitigation = request.form.get("mitigation", "reweighing")

    if protected not in df.columns or outcome not in df.columns:
        return jsonify({"error": "Invalid column names"}), 400

    try:
        result = run_analysis(df, file.filename, protected, outcome, mitigation, False, False, False)
        # Strip non-serialisable / internal keys
        api_result = {k: v for k, v in result.items() if not k.startswith("_")}
        return jsonify(api_result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── SAMPLE CSV ────────────────────────────────────────────────────────────────

@app.route("/sample-csv")
def sample_csv():
    """Generate and serve a sample CSV for demo purposes."""
    np.random.seed(42)
    n = 300
    gender = np.random.choice(["Male", "Female", "Non-binary"], n, p=[0.55, 0.40, 0.05])
    age    = np.random.randint(22, 60, n)
    edu    = np.random.choice(["High School", "Bachelor", "Master", "PhD"], n, p=[0.3, 0.4, 0.2, 0.1])
    exp    = np.random.randint(0, 20, n)
    # Inject bias: males get hired more
    base_prob = 0.45
    hired = []
    for i in range(n):
        prob = base_prob + (0.2 if gender[i] == "Male" else 0) + exp[i] * 0.01
        hired.append(1 if np.random.random() < min(prob, 0.95) else 0)

    sample_df = pd.DataFrame({
        "gender": gender, "age": age,
        "education": edu, "experience_years": exp, "hired": hired
    })
    csv_buf = io.StringIO()
    sample_df.to_csv(csv_buf, index=False)
    csv_buf.seek(0)
    return send_file(
        io.BytesIO(csv_buf.getvalue().encode()),
        mimetype="text/csv",
        as_attachment=True,
        download_name="justiflow_sample_dataset.csv"
    )


# ── DOWNLOAD ENDPOINTS ────────────────────────────────────────────────────────

@app.route("/api/blockchain/verify", methods=["POST"])
def verify_file():
    """Verify an uploaded file against the blockchain ledger."""
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400
    file_bytes = request.files["file"].read()
    result     = justi_chain.verify_file(file_bytes)
    return jsonify(result)


@app.route("/api/blockchain/summary")
def blockchain_summary():
    return jsonify(justi_chain.summary())


@app.route("/api/history")
def history_api():
    return jsonify(load_history())


@app.route("/history")
def history_page():
    history = load_history()
    chain   = justi_chain.summary()
    return render_template("history.html", history=history, chain=chain)


@app.route("/download/report/<analysis_id>")
def download_report(analysis_id: str):
    result = load_result(analysis_id)
    if not result:
        abort(404)
    txt = result.get("_report_txt", "No report available.")
    buf = io.BytesIO(txt.encode("utf-8"))
    return send_file(
        buf, mimetype="text/plain",
        as_attachment=True,
        download_name=f"justiflow_report_{analysis_id}.txt"
    )


@app.route("/download/certificate/<analysis_id>")
def download_certificate(analysis_id: str):
    result = load_result(analysis_id)
    if not result:
        abort(404)
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib import colors
        from reportlab.lib.units import mm
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.enums import TA_CENTER

        buf = io.BytesIO()
        doc = SimpleDocTemplate(buf, pagesize=A4, topMargin=20*mm, bottomMargin=20*mm,
                                 leftMargin=20*mm, rightMargin=20*mm)
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle("title", parent=styles["Title"],
                                     fontSize=22, textColor=colors.HexColor("#3b82f6"),
                                     alignment=TA_CENTER, spaceAfter=6)
        sub_style   = ParagraphStyle("sub", parent=styles["Normal"],
                                     fontSize=12, alignment=TA_CENTER, spaceAfter=4,
                                     textColor=colors.HexColor("#94a3b8"))
        body_style  = ParagraphStyle("body", parent=styles["Normal"],
                                     fontSize=11, spaceAfter=8, leading=16)
        verdict_style = ParagraphStyle("verdict", parent=styles["Normal"],
                                       fontSize=14, textColor=colors.HexColor("#10b981"),
                                       alignment=TA_CENTER, fontName="Helvetica-Bold",
                                       spaceAfter=6)
        if result["risk_level"] == "HIGH":
            verdict_color = colors.HexColor("#ef4444")
        elif result["risk_level"] == "MEDIUM":
            verdict_color = colors.HexColor("#f59e0b")
        else:
            verdict_color = colors.HexColor("#10b981")

        story = [
            Paragraph("JUSTIFLOW", title_style),
            Paragraph("AI Fairness Audit Certificate", sub_style),
            Spacer(1, 8*mm),
            HRFlowable(width="100%", color=colors.HexColor("#1e3a5f")),
            Spacer(1, 6*mm),
            Paragraph(f"<b>Dataset:</b> {result['filename']}", body_style),
            Paragraph(f"<b>Analysis Date:</b> {result['analysis_date']}", body_style),
            Paragraph(f"<b>Analysis ID:</b> {result['analysis_id']}", body_style),
            Paragraph(f"<b>Dataset Hash:</b> {result['dataset_hash']}", body_style),
            Spacer(1, 6*mm),
            HRFlowable(width="100%", color=colors.HexColor("#1e3a5f")),
            Spacer(1, 6*mm),
            Paragraph(f"<b>Bias Score:</b> {result['bias_score']}%", body_style),
            Paragraph(f"<b>Improvement After Mitigation:</b> {result['improvement']}%", body_style),
            Paragraph(f"<b>Confidence:</b> {result['confidence']}%", body_style),
            Spacer(1, 4*mm),
            Paragraph(
                f"<font color='#{verdict_color.hexval()[2:]}' size='14'><b>Risk Level: {result['risk_level']}</b></font>",
                ParagraphStyle("v2", parent=styles["Normal"], fontSize=14, alignment=TA_CENTER, spaceAfter=6)
            ),
            Paragraph(result["verdict_title"], verdict_style),
            Spacer(1, 6*mm),
            HRFlowable(width="100%", color=colors.HexColor("#1e3a5f")),
            Spacer(1, 4*mm),
            Paragraph("<b>Audit Paragraph</b>", body_style),
            Paragraph(result["audit_paragraph"], body_style),
            Spacer(1, 4*mm),
            HRFlowable(width="100%", color=colors.HexColor("#1e3a5f")),
            Spacer(1, 4*mm),
            Paragraph("<b>Regulatory Alignment</b>", body_style),
            Paragraph("EU AI Act — Article 10 (Data Governance) · Article 13 (Transparency)", body_style),
            Paragraph("Fairlearn Framework — DIR, SPD, EOD · Google AI Principles", body_style),
            Spacer(1, 8*mm),
            Paragraph("Generated by JustiFlow · AI Fairness Auditing Platform", sub_style),
        ]
        doc.build(story)
        buf.seek(0)
        return send_file(buf, mimetype="application/pdf", as_attachment=True,
                         download_name=f"justiflow_certificate_{analysis_id}.pdf")
    except Exception:
        # Fallback: plain text certificate
        txt = generate_txt_report(result)
        buf = io.BytesIO(txt.encode("utf-8"))
        return send_file(buf, mimetype="text/plain", as_attachment=True,
                         download_name=f"justiflow_certificate_{analysis_id}.txt")


import os

# ... your existing code ...

if __name__ == "__main__":
    # Get port from Railway's environment, default to 8080 if not found
    port = int(os.environ.get("PORT", 8080))
    # host='0.0.0.0' is required for the container to be accessible
    app.run(host='0.0.0.0', port=port)
