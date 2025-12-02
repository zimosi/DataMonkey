# pip install langgraph langchain-core langchain-openai pandas scikit-learn matplotlib duckdb ydata-profiling
from __future__ import annotations
import logging
from typing import TypedDict, Annotated, Dict, Any, List, Optional, Literal
import operator, os, uuid, json, joblib
from pathlib import Path
import pandas as pd, numpy as np, matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import f1_score, roc_auc_score, r2_score, mean_squared_error
from sklearn.linear_model import LogisticRegression, LinearRegression, ElasticNet
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
from langchain_openai import ChatOpenAI
from config import OPENAI_API_KEY, OPENAI_MODEL, OPENAI_MAX_TOKENS, OPENAI_TEMPERATURE
ART = "artifacts"; os.makedirs(ART, exist_ok=True)
logger = logging.getLogger(__name__)
DATA: dict[str, pd.DataFrame] = {}

def seed_data():
    """Initialize artifacts directory"""
    os.makedirs("static", exist_ok=True)
    os.makedirs("static/plots", exist_ok=True)  # For serving plots to frontend

class State(TypedDict):
    messages: Annotated[list, operator.add]
    job_id: str
    df_ref: str  # Reference to dataframe in DATA dict
    schema: Dict[str, Any] | None
    task: Optional[Literal["classification","regression"]]
    target: Optional[str]
    user_prompt: Optional[str]  # Original user prompt
    reason_profile: Optional[str]
    preprocess_plan: Optional[dict]
    pipeline_path: Optional[str]
    splits: Optional[dict]
    candidates: Optional[List[str]]  # Algorithm candidates for search
    hpo_results: Optional[List[dict]]  # Hyperparameter optimization results
    best_model_path: Optional[str]  # Path to best trained model

def save_json(obj, name):
    path = f"{ART}/{name}"
    with open(path, "w") as f: json.dump(obj, f, indent=2)
    return path

@tool
def profile(df_ref: str) -> dict:
    """Quick schema + basic stats; saves small artifacts (csv head, corr heatmap)."""
    df = DATA[df_ref]
    schema = {c:str(df[c].dtype) for c in df.columns}
    return {"schema": schema}

@tool
def infer_target(df_ref: str, hint: Optional[str]=None) -> dict:
    """Infer target column + task from hint or heuristics (last col, low-cardinality numeric→reg/class)."""
    df = DATA[df_ref]
    target = hint if hint in df.columns else df.columns[-1]
    y = df[target]
    if y.dtype.kind in "ifu":
        task = "classification" if y.nunique()<=10 else "regression"
    else:
        task = "classification"
    return {"target": target, "task": task}

@tool
def split_data(df_ref: str, target: str, test_size: float=0.2, seed: int=42) -> dict:
    """Split data into train and test sets"""
    df = DATA[df_ref]
    strat = df[target] if df[target].nunique()<=20 and df[target].dtype.kind not in "f" else None
    train_idx, test_idx = train_test_split(df.index, test_size=test_size, random_state=seed, stratify=strat)
    out = {"train_idx": train_idx.tolist(), "test_idx": test_idx.tolist(), "stratified": strat is not None}
    save_json(out, f"{df_ref}_split.json")
    return out

@tool
def build_preprocess(df_ref: str, target: str, task: str) -> dict:
    """Build preprocessing pipeline"""
    df = DATA[df_ref]
    X = df.drop(columns=[target])
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]
    pre = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("scaler", StandardScaler())]), num_cols),
            ("cat", Pipeline([("onehot", OneHotEncoder(handle_unknown="ignore"))]), cat_cols),
        ]
    )
    spec = {"num_cols": num_cols, "cat_cols": cat_cols}
    spec_path = save_json(spec, f"{df_ref}_preprocess.json")
    return {"spec": spec, "spec_path": spec_path}

@tool
def search_models(df_ref: str, target: str, task: str, split: dict, preprocess_spec_path: str, n_trials: int=12, seed: int=42) -> dict:
    """ Search for best model """
    df = DATA[df_ref]
    train = df.loc[split["train_idx"]]
    X, y = train.drop(columns=[target]), train[target]
    num_cols = json.load(open(preprocess_spec_path))["num_cols"]
    cat_cols = json.load(open(preprocess_spec_path))["cat_cols"]
    pre = ColumnTransformer([
        ("num", Pipeline([("scaler", StandardScaler())]), num_cols),
        ("cat", Pipeline([("onehot", OneHotEncoder(handle_unknown="ignore"))]), cat_cols),
    ])
    # candidates
    if task=="classification":
        models = [
            ("logreg", LogisticRegression(max_iter=1000)),
            ("rf", RandomForestClassifier(random_state=seed))
        ]
        scorer = lambda yt, yp: roc_auc_score(yt, yp, multi_class="ovr") if len(np.unique(yt))>2 else roc_auc_score(yt, yp)
        proba = True
    else:
        models = [
            ("linreg", LinearRegression()),
            ("enet", ElasticNet(random_state=seed)),
            ("rfr", RandomForestRegressor(random_state=seed))
        ]
        scorer = lambda yt, yp: -np.sqrt(mean_squared_error(yt, yp))  # higher is better
        proba = False
    kf = StratifiedKFold(5, shuffle=True, random_state=seed) if task=="classification" and y.nunique()<=20 else KFold(5, shuffle=True, random_state=seed)

    trials, best = [], {"score": -1e9, "name": None, "path": None}
    for name, estimator in models:
        pipe = Pipeline([("pre", pre), ("mdl", estimator)])
        cv_scores=[]
        for tr, va in kf.split(X, y):
            Xtr, Xva = X.iloc[tr], X.iloc[va]; ytr, yva = y.iloc[tr], y.iloc[va]
            pipe.fit(Xtr, ytr)
            if task=="classification":
                if len(np.unique(y))>2:
                    proba_va = pipe.predict_proba(Xva)
                    score = scorer(yva, proba_va, )
                else:
                    proba_va = pipe.predict_proba(Xva)[:,1]
                    score = scorer(yva, proba_va)
            else:
                pred = pipe.predict(Xva)
                score = scorer(yva, pred)
            cv_scores.append(float(score))
        avg = float(np.mean(cv_scores))
        trials.append({"model": name, "cv_scores": cv_scores, "mean": avg})
        if avg > best["score"]:
            best.update({"score": avg, "name": name})
            # fit on all train
            pipe.fit(X, y)
            path = f"{ART}/{df_ref}_{name}_best.pkl"; joblib.dump(pipe, path)
            best["path"] = path

    trials_path = save_json(trials, f"{df_ref}_trials.json")
    return {"trials_path": trials_path, "leader": best}

@tool
def evaluate_model(df_ref: str, target: str, split: dict, leader_path: str, task: str) -> dict:
    """evalute model"""
    df = DATA[df_ref]
    test = df.loc[split["test_idx"]]
    Xte, yte = test.drop(columns=[target]), test[target]
    pipe = joblib.load(leader_path)
    pred = pipe.predict(Xte)
    if task=="classification":
        metric = {"f1": float(f1_score(yte, pred, average="weighted"))}
    else:
        metric = {"rmse": float(np.sqrt(mean_squared_error(yte, pred))), "r2": float(r2_score(yte, pred))}
    mp = save_json(metric, f"{df_ref}_metrics.json")
    return {"metrics": metric, "metrics_path": mp}

TOOLS = [profile, infer_target, split_data, build_preprocess, search_models, evaluate_model]
tool_node = ToolNode(TOOLS)
llm = ChatOpenAI(
    model=OPENAI_MODEL,
    api_key=OPENAI_API_KEY,
    temperature=OPENAI_TEMPERATURE,
    max_tokens=OPENAI_MAX_TOKENS
).bind_tools(TOOLS)

# Separate LLM without tools for profile analysis (we don't want it to call tools here)
llm_no_tools = ChatOpenAI(
    model=OPENAI_MODEL,
    api_key=OPENAI_API_KEY,
    temperature=OPENAI_TEMPERATURE,
    max_tokens=OPENAI_MAX_TOKENS
)

SYSTEM = SystemMessage(
    content=(
        "You are an expert machine learning engineer specializing in automated ML pipeline construction. "
        "Your role is to analyze datasets and user requests to determine the target column and task type (classification or regression).\n\n"
        "Guidelines:\n"
        "- Identify the target column from user's prompt (what they want to predict)\n"
        "- Classification: discrete categories, strings, or integers with <10 unique values\n"
        "- Regression: continuous values, floats, or integers with >10 unique values\n"
        "- Return ONLY valid JSON: {\"target\": \"column_name\", \"task\": \"classification\" or \"regression\"}\n"
        "- Be precise and logical in your analysis"
    )
)

# ---- Graph nodes (each emits artifacts; keep it short) ----
def n_profile(s: State) -> State:
    """Profile the dataset and determine target column and task type using LLM"""
    # Get user message
    user_message = s["messages"][-1].content if s["messages"] else s.get("user_prompt", "")
    
    # Get dataframe from DATA dict using df_ref (which should be job_id)
    df_ref = s.get("df_ref", s["job_id"])
    if df_ref not in DATA:
        logger.error(f"DataFrame '{df_ref}' not found in DATA")
        raise ValueError(f"DataFrame '{df_ref}' not found")
    
    df = DATA[df_ref]
    csv_preview = df.head(20).to_string()
    
    # Build enhanced prompt
    enhanced_prompt = f"""You are analyzing a dataset with {len(df)} rows and {len(df.columns)} columns.

Columns: {', '.join(df.columns.tolist())}

Data preview:
{csv_preview}

User question: {user_message}

Please analyze the dataset and user question to determine:
1. The target column (what the user wants to predict)
2. The task type (classification or regression)

Then give the reason for your answer. Why you choose this target column and task type.
Return ONLY a valid JSON object with these exact keys:
{{"target": "column_name", "task": "classification" or "regression", "reason": "reason for your answer"}}

Do not include any additional text, just the JSON."""
    
    logger.info(f"💬 User question: {user_message}")
    logger.info(f"📊 Dataset: {len(df)} rows x {len(df.columns)} columns")
    
    # Invoke LLM WITHOUT tools (we just want text response, not tool calls)
    ai = llm_no_tools.invoke([SYSTEM, HumanMessage(content=enhanced_prompt)])
    
    # Parse JSON response
    import json
    import re
    # Initialize reason to None in case of fallback
    reason = None
    try:
        # Check if content exists
        if not ai.content or not ai.content.strip():
            logger.error("LLM returned empty content")
            raise ValueError("Empty LLM response")
        
        # Extract JSON from response (might have markdown code blocks)
        content = ai.content.strip()
        logger.info(f"💬 ai output: {content[:500]}...")  # Log first 500 chars

        # Strategy 1: Try to parse the entire content as JSON
        try:
            result = json.loads(content)
        except json.JSONDecodeError:
            # Strategy 2: Try to extract JSON from markdown code blocks (```json ... ```)
            json_code_block = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', content, re.DOTALL)
            if json_code_block:
                result = json.loads(json_code_block.group(1))
            else:
                logger.info(f"No JSON code block found in response")
                # Strategy 3: Try to find the first complete JSON object
                # Find the first { and match to the last }
                start_idx = content.find('{')
                if start_idx == -1:
                    raise ValueError("No JSON object found in response")
                
                # Find matching closing brace
                brace_count = 0
                end_idx = start_idx
                for i in range(start_idx, len(content)):
                    if content[i] == '{':
                        brace_count += 1
                    elif content[i] == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            end_idx = i + 1
                            break
                
                if brace_count != 0:
                    raise ValueError("Incomplete JSON object in response")
                
                json_str = content[start_idx:end_idx]
                result = json.loads(json_str)
            
        # Extract values from result
        target = result.get("target")
        task = result.get("task")
        reason = result.get("reason")
        
        # Validate target exists in dataframe
        if target not in df.columns:
            logger.warning(f"Target '{target}' not found in columns, using fallback")
            raise ValueError(f"Target column '{target}' not found")
        
        logger.info(f"✅ Determined target: {target}, task: {task}")
    except (json.JSONDecodeError, ValueError, KeyError) as e:
        logger.error(f"Failed to parse LLM response: {e}")
        logger.info("Falling back to infer_target tool")
        # Fallback to infer_target tool
        out = infer_target.invoke({"df_ref": df_ref})
        target = out["target"]
        task = out["task"]
        reason = f"Fallback: Selected {target} (last column) with task {task} based on heuristics."
        logger.info(f"✅ Fallback target: {target}, task: {task}")
    
    # Run profile tool to get schema
    profile_result = profile.invoke({"df_ref": df_ref})
    
    return {
        "target": target,
        "task": task,
        "schema": profile_result["schema"],
        "reason_profile": reason
    }

def n_split(s: State) -> State:
    """Split data into train/test sets"""
    logger.info("Splitting data...")
    # Return empty state for now (placeholder)
    return {}

def n_preprocess(s: State) -> State:
    """Preprocess data"""
    from sklearn.model_selection import train_test_split
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.impute import SimpleImputer

    logger.info("Preprocessing data...")
    
    # Get dataframe from DATA dict using df_ref (which should be job_id)
    df_ref = s.get("df_ref", s["job_id"])
    if df_ref not in DATA:
        logger.error(f"DataFrame '{df_ref}' not found in DATA")
        raise ValueError(f"DataFrame '{df_ref}' not found")
    
    df = DATA[df_ref]
    
    # Create run directory using job_id
    run_dir = Path(ART) / s["job_id"]
    run_dir.mkdir(parents=True, exist_ok=True)

    # Get target and task from state
    target = s.get("target")
    task = s.get("task")
    
    if not target:
        raise ValueError("Target column not set in state. Run PROFILE node first.")
    if not task:
        raise ValueError("Task type not set in state. Run PROFILE node first.")
    
    y = df[target]
    X = df.drop(columns=[target])

    # Identify columns
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    # Create preprocessing pipelines
    num_tr = Pipeline([("imp", SimpleImputer(strategy="median")),
                       ("sc", StandardScaler())])
    cat_tr = Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                       ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])

    ct = ColumnTransformer([("num", num_tr, num_cols),
                            ("cat", cat_tr, cat_cols)])

    # Split data into train/valid/test (60/20/20)
    Xtr, Xtmp, ytr, ytmp = train_test_split(
        X, y, test_size=0.4, random_state=42,
        stratify=y if task=="classification" and y.nunique() <= 20 else None
    )
    Xva, Xte, yva, yte = train_test_split(
        Xtmp, ytmp, test_size=0.5, random_state=42,
        stratify=ytmp if task=="classification" and ytmp.nunique() <= 20 else None
    )

    # Fit preprocessing pipeline on training data
    ct.fit(Xtr)

    # Materialize transformed splits
    def tx(df_part): 
        arr = ct.transform(df_part)
        return pd.DataFrame(arr, columns=num_cols + [f"cat_{i}" for i in range(len(cat_cols))] if cat_cols else num_cols)

    # Save transformed splits (using CSV if parquet not available, otherwise parquet)
    try:
        # Try parquet first
        paths = {
            "train": str(run_dir / "train.parquet"),
            "valid": str(run_dir / "valid.parquet"),
            "test":  str(run_dir / "test.parquet"),
        }
        tx(Xtr).assign(_y=ytr.values).to_parquet(paths["train"])
        tx(Xva).assign(_y=yva.values).to_parquet(paths["valid"])
        tx(Xte).assign(_y=yte.values).to_parquet(paths["test"])
        logger.info("Saved splits in Parquet format")
    except (ImportError, AttributeError, Exception) as e:
        # Fallback to CSV if parquet library not available or any error occurs
        logger.warning(f"Parquet save failed ({e}), using CSV instead")
        paths = {
            "train": str(run_dir / "train.csv"),
            "valid": str(run_dir / "valid.csv"),
            "test":  str(run_dir / "test.csv"),
        }
        tx(Xtr).assign(_y=ytr.values).to_csv(paths["train"], index=False)
        tx(Xva).assign(_y=yva.values).to_csv(paths["valid"], index=False)
        tx(Xte).assign(_y=yte.values).to_csv(paths["test"], index=False)
        logger.info("Saved splits in CSV format")

    # Save preprocessing pipeline
    pipe_path = str(run_dir / "pipeline.pkl")
    joblib.dump(ct, pipe_path)

    plan = {
        "impute": {"num": "median", "cat": "most_frequent"},
        "scale": {"num": "standard"},
        "encode": {"cat": "onehot"},
        "split": {"test_size": 0.2, "valid_size": 0.2}
    }
    
    logger.info(f"✅ Preprocessing completed. Pipeline saved to {pipe_path}")
    
    return {
        "preprocess_plan": plan,
        "pipeline_path": pipe_path,
        "splits": paths,
    }

def _load_split(paths: Dict[str, str]):
    """Load train and validation splits from parquet or CSV files"""
    try:
        # Try parquet first
        tr = pd.read_parquet(paths["train"])
        va = pd.read_parquet(paths["valid"])
    except Exception:
        # Fallback to CSV
        tr = pd.read_csv(paths["train"])
        va = pd.read_csv(paths["valid"])
    
    Xtr, ytr = tr.drop(columns=["_y"]), tr["_y"]
    Xva, yva = va.drop(columns=["_y"]), va["_y"]
    return Xtr, ytr, Xva, yva

def _score(metric: str, y_true, y_pred, y_proba=None) -> float:
    """Calculate metric score"""
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, f1_score, roc_auc_score
    import numpy as np
    if metric == "rmse": 
        return float((mean_squared_error(y_true, y_pred)) ** 0.5)
    if metric == "mae":  
        return float(mean_absolute_error(y_true, y_pred))
    if metric == "r2":   
        return float(r2_score(y_true, y_pred))
    if metric == "accuracy": 
        return float(accuracy_score(y_true, y_pred))
    if metric == "f1":
        # Handle multiclass with average parameter
        if len(np.unique(y_true)) > 2:
            return float(f1_score(y_true, y_pred, average="weighted"))
        else:
            return float(f1_score(y_true, y_pred, average="binary"))
    if metric == "auc":
        if y_proba is None: 
            return 0.0
        # Handle multiclass AUC
        if len(np.unique(y_true)) > 2:
            return float(roc_auc_score(y_true, y_proba, multi_class="ovr", average="weighted"))
        else:
            return float(roc_auc_score(y_true, y_proba))
    raise ValueError(f"Unknown metric: {metric}")

def _higher_is_better(metric: str) -> bool:
    return metric in {"accuracy", "f1", "auc", "r2"}


def n_search(s: State) -> State:
    """Search for best model with hyperparameter optimization"""
    import random
    from sklearn.dummy import DummyRegressor, DummyClassifier
    from sklearn.linear_model import Ridge, LogisticRegression
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
    
    logger.info("Searching for best model and hyperparameters...")
    
    # Get task from state
    task = s.get("task")
    if not task:
        raise ValueError("Task type not set in state. Run PROFILE node first.")
    
    # Get splits from state
    splits = s.get("splits")
    if not splits:
        raise ValueError("Splits not found in state. Run PREPROCESS node first.")
    
    # Load preprocessed splits
    try:
        Xtr, ytr, Xva, yva = _load_split(splits)
        logger.info(f"Loaded splits: Train={len(Xtr)}, Valid={len(Xva)}")
    except Exception as e:
        logger.error(f"Failed to load splits: {e}")
        raise
    
    # Determine metric
    metric = "accuracy" if task == "classification" else "rmse"
    
    # Define algorithm candidates based on task
    if task == "regression":
        candidates = ["baseline", "ridge", "random_forest"]
    else:
        candidates = ["baseline", "logreg", "random_forest"]
    
    # Try to add xgboost if available
    try:
        from xgboost import XGBRegressor, XGBClassifier
        candidates.append("xgboost")
        logger.info("XGBoost available, including in search")
    except ImportError:
        logger.warning("XGBoost not available, skipping")
    
    results = []
    best_overall = None
    best_model = None
    best_algo = None
    
    for algo in candidates:
        logger.info(f"🔍 Testing algorithm: {algo}")
        trials = 3 if algo != "baseline" else 1
        best_for_algo = None
        best_score = float('-inf') if _higher_is_better(metric) else float('inf')
        
        for trial in range(trials):
            try:
                # Create model with random hyperparameters
                if task == "regression":
                    if algo == "baseline": 
                        model = DummyRegressor()
                    elif algo == "ridge":   
                        model = Ridge(alpha=10 ** random.uniform(-4, 2), random_state=42)
                    elif algo == "random_forest": 
                        model = RandomForestRegressor(
                            n_estimators=random.randint(60, 200),
                            max_depth=random.choice([None, 5, 10]),
                            random_state=42
                        )
                    elif algo == "xgboost": 
                        model = XGBRegressor(
                            tree_method="hist",
                            n_estimators=random.randint(100, 300),
                            max_depth=random.randint(3, 8),
                            learning_rate=10 ** random.uniform(-3, -0.3),
                            random_state=42
                        )
                    else: 
                        continue
                else:  # classification
                    if algo == "baseline": 
                        model = DummyClassifier(strategy="most_frequent", random_state=42)
                    elif algo == "logreg":  
                        model = LogisticRegression(
                            max_iter=1000, 
                            C=10 ** random.uniform(-3, 2),
                            random_state=42
                        )
                    elif algo == "random_forest": 
                        model = RandomForestClassifier(
                            n_estimators=random.randint(60, 200),
                            max_depth=random.choice([None, 5, 10]),
                            random_state=42
                        )
                    elif algo == "xgboost": 
                        model = XGBClassifier(
                            tree_method="hist",
                            n_estimators=random.randint(100, 300),
                            max_depth=random.randint(3, 8),
                            learning_rate=10 ** random.uniform(-3, -0.3),
                            random_state=42
                        )
                    else: 
                        continue
                
                # Train model
                model.fit(Xtr, ytr)
                
                # Predict
                yhat = model.predict(Xva)
                yproba = None
                if hasattr(model, "predict_proba") and task == "classification":
                    proba = model.predict_proba(Xva)
                    # For binary classification, use probability of positive class
                    if proba.ndim == 2 and proba.shape[1] == 2:
                        yproba = proba[:, 1]
                
                # Calculate score
                val = _score(metric, yva, yhat, yproba)
                
                logger.info(f"  Trial {trial+1}/{trials}: {metric}={val:.4f}")
                
                # Update best for this algorithm
                is_better = False
                if best_for_algo is None:
                    is_better = True
                elif _higher_is_better(metric):
                    is_better = val > best_score
                else:
                    is_better = val < best_score
                
                if is_better:
                    best_score = val
                    best_for_algo = {
                        "algo": algo,
                        "params": model.get_params() if hasattr(model, "get_params") else {},
                        "score": val,
                        "trial": trial + 1
                    }
                    # Update overall best
                    if best_overall is None:
                        best_overall = best_for_algo
                        best_model = model
                        best_algo = algo
                    elif _higher_is_better(metric):
                        if val > best_overall["score"]:
                            best_overall = best_for_algo
                            best_model = model
                            best_algo = algo
                    else:
                        if val < best_overall["score"]:
                            best_overall = best_for_algo
                            best_model = model
                            best_algo = algo
                            
            except Exception as e:
                logger.error(f"  Trial {trial+1} failed for {algo}: {e}")
                continue
        
        if best_for_algo:
            results.append(best_for_algo)
            logger.info(f"✅ Best {algo}: {metric}={best_for_algo['score']:.4f}")
    
    if not results:
        raise ValueError("No models were successfully trained")
    
    # Save best model
    run_dir = Path(ART) / s["job_id"]
    best_model_path = str(run_dir / "best_model.pkl")
    joblib.dump(best_model, best_model_path)
    logger.info(f"✅ Best model saved to {best_model_path}")
    
    # Save results
    results_path = save_json(results, f"{s['job_id']}_search_results.json")
    
    logger.info(f"🎯 Best algorithm: {best_algo}, {metric}={best_overall['score']:.4f}")
    
    return {
        "hpo_results": results,
        "best_model_path": best_model_path,
        "candidates": candidates
    }


def n_eval(s: State) -> State:
    """Evaluate model"""
    logger.info("Evaluating models...")
    # Return empty state for now (placeholder)
    return {}
# ---- Build graph ----
def create_workflow() -> StateGraph:
    g = StateGraph(State)
    g.add_node("PROFILE", n_profile)
    g.add_node("SPLIT", n_split)
    g.add_node("PREPROCESS", n_preprocess)
    g.add_node("SEARCH", n_search)
    g.add_node("EVALUATE", n_eval)
    g.add_edge(START, "PROFILE")
    g.add_edge("PROFILE", "SPLIT")
    g.add_edge("SPLIT", "PREPROCESS")
    g.add_edge("PREPROCESS", "SEARCH")
    g.add_edge("SEARCH", "EVALUATE")
    g.add_edge("EVALUATE", END)

    return g.compile()

async def start_machine_learning_pipeline(df: pd.DataFrame, job_id: str, user_prompt: str) -> dict:
    try:
        logger.info(f"Starting machine learning for job {job_id} with prompt: {user_prompt}")
        
        # Initialize artifacts directory
        seed_data()
        
        # Store dataframe in DATA with job_id as key
        DATA[job_id] = df
        
        # Create workflow
        workflow = create_workflow()
        
        # Initial state with user message
        initial_state = {
            "messages": [HumanMessage(content=user_prompt)],
            "job_id": job_id,
            "df_ref": job_id,  # Reference to dataframe in DATA
            "schema": None,
            "task": None,
            "target": None,
            "user_prompt": user_prompt,
            "reason_profile": None,
            "preprocess_plan": None,
            "pipeline_path": None,
            "splits": None,
            "candidates": None,
            "hpo_results": None,
            "best_model_path": None
        }
        
        # Run workflow with config
        logger.info("Running workflow...")
        config = {"recursion_limit": 10}
        final_state = await workflow.ainvoke(initial_state, config=config)
        
        # Clean up: remove dataframe from DATA
        if job_id in DATA:
            del DATA[job_id]
        
        return final_state
        
    except Exception as e:
        logger.error(f"ML pipeline failed for job {job_id}: {str(e)}")
        # Clean up on error
        if job_id in DATA:
            del DATA[job_id]
        raise
    
'''
# ---- Example run ----
if __name__ == "__main__":
    df = pd.DataFrame({
        "age":[23,45,36,52,33,29,41,60,37,49],
        "bmi":[21.2,30.1,27.5,28.2,24.3,25.0,29.9,31.4,26.3,27.8],
        "smoker":[0,1,0,1,0,0,1,1,0,1],
        "charges":[2100,16800,9800,22000,4500,5200,15500,24000,11000,19000]
    })
    job = uuid.uuid4().hex[:8]
    DATA[job] = df
    state: State = {
        "messages":[HumanMessage("Train a model to predict charges")],
        "job_id": job,
        "df_ref": job,
        "schema": None, "profile": None,
        "task": None, "target": None,
        "split": None, "preprocess": None,
        "candidate_space": None, "trial_results": [],
        "best_model_path": None, "metrics": None,
        "artifacts": [], "step_count": 0
    }
    final = app.invoke(state)
    print("Best model:", final["best_model_path"])
    print("Metrics:", final["metrics"])
    print("Artifacts:", final["artifacts"])
'''