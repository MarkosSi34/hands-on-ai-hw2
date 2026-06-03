"""
LangGraph tool definitions for the conversational agent.

Three tools are exposed to the agent:

  1. retrieve_knowledge  — RAG retrieval over domain documents (Task 1)
  2. predict_income      — HW1 best model wrapped as a callable (Task 2)
  3. dataset_stats       — Summary statistics over the HW1 dataset (Task 5, bonus)

Every tool is implemented as its own class so the orchestration mirrors the
HW1 style (class-with-run_pipeline). A top-level ToolBuilder class wires all
three together and returns the list of LangChain Tool objects that the
LangGraph agent consumes.
"""

import os
import json
import logging
from pathlib import Path
from typing import List

import joblib
import numpy as np
import pandas as pd

from langchain_core.tools import StructuredTool, BaseTool

from src.rag import RAGPipeline

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


#  Tool 1 Retrieval (RAG)

class KnowledgeBaseTool:
    """
    Wraps the RAGPipeline as a LangChain tool.

    Domain  : Socioeconomic / Demographics knowledge base
    Task    : Answer factual or conceptual questions about income,
              education, demographics by retrieving the top-k most
              relevant passages from data/documents/.
    Source  : src.rag.RAGPipeline

    The agent invokes this tool when the user asks a "what / why / how"
    question that does not require running the HW1 model.
    """

    TOOL_NAME = "retrieve_knowledge"
    TOOL_DESCRIPTION = (
        "Search the domain knowledge base for information about US Census "
        "income data, demographics, education, and socioeconomic factors. "
        "Use this tool whenever the user asks a factual or conceptual "
        "question (e.g. 'what factors influence income', 'how does education "
        "relate to earnings', 'what is the gender wage gap'). "
        "Input: a natural-language query string. "
        "Output: the top-3 most relevant passages, each tagged with its "
        "source document."
    )

    def __init__(self, rag: RAGPipeline = None):
        self.rag = rag or RAGPipeline()

    def load_pipeline(self):
        """Load (or build) the persisted vector store."""
        logging.info("Initialising KnowledgeBaseTool...")
        self.rag.run_pipeline(rebuild=False)
        logging.info("KnowledgeBaseTool ready.")

    def _call(self, query: str):
        """Underlying retrieval function exposed to the agent."""
        return self.rag.retrieve(query)

    def as_langchain_tool(self):
        return StructuredTool.from_function(
            func=self._call,
            name=self.TOOL_NAME,
            description=self.TOOL_DESCRIPTION,
        )



#  Tool 2 Prediction (HW1 best model)

class PredictionTool:
    """
    Wraps the HW1 best model as a callable tool.

    Domain  : Adult Census Income (UCI dataset)
    Task    : Binary classification — predict whether annual income > $50K
    Source  : models/best_model.pkl   (carried over from Homework 1)

    Design choices
    ──────────────
    • Loads ALL four HW1 artifacts at startup (best_model, scaler,
      ohe_encoder, preprocessing_meta). The assignment spec mentions only
      best_model.pkl and scaler.pkl, but the HW1 pipeline also requires
      ohe_encoder.pkl and preprocessing_meta.pkl to reproduce predictions
      exactly. Without these the tool's output would not match HW1's
      /predict endpoint.
    • Preprocessing is identical to hands-on-ai-hw1/src/api.py:
        impute → clip → group rare countries → label-encode
        → one-hot → engineer features → align columns → scale.
    • Input accepted as either a JSON string or a dict. The LLM is
      instructed (via the tool description) to pass a JSON string.
    • Decision threshold = 0.5 (same as HW1).
    """

    TOOL_NAME = "predict_income"
    POSITIVE_THRESHOLD = 0.5
    ARTIFACTS_DIR = "models"

    REQUIRED_FIELDS = [
        "age", "workclass", "fnlwgt", "education", "education_num",
        "marital_status", "occupation", "relationship", "race", "sex",
        "capital_gain", "capital_loss", "hours_per_week", "native_country",
    ]

    TOOL_DESCRIPTION = (
        "Predict whether a US adult's annual income exceeds $50,000 based "
        "on US Census features (the Homework 1 model). "
        "Use this tool whenever the user provides specific demographic / "
        "employment values and wants a yes/no prediction or probability. "
        "Input MUST be a JSON string with all of these fields:\n"
        '  age (int), workclass (str), fnlwgt (int), education (str),\n'
        '  education_num (int), marital_status (str), occupation (str),\n'
        '  relationship (str), race (str), sex (str: "Male" or "Female"),\n'
        '  capital_gain (int), capital_loss (int), hours_per_week (int),\n'
        '  native_country (str).\n'
        "Example input: "
        '{"age": 39, "workclass": "State-gov", "fnlwgt": 77516, '
        '"education": "Bachelors", "education_num": 13, '
        '"marital_status": "Never-married", "occupation": "Adm-clerical", '
        '"relationship": "Not-in-family", "race": "White", "sex": "Male", '
        '"capital_gain": 2174, "capital_loss": 0, "hours_per_week": 40, '
        '"native_country": "United-States"}\n'
        "If the user has not given a value for a field, ask them — do not "
        "invent values. Output is a human-readable prediction with the "
        "predicted class (>50K / <=50K) and the probability of >50K."
    )

    def __init__(self, artifacts_dir: str = None):
        self.artifacts_dir = artifacts_dir or self.ARTIFACTS_DIR
        self.model   = None
        self.scaler  = None
        self.encoder = None
        self.meta    = None

    # Loading
    def load_artifacts(self):
        """Load the four HW1 artifacts from disk."""
        logging.info(f"Loading HW1 artifacts from '{self.artifacts_dir}'...")
        d = Path(self.artifacts_dir)
        self.model   = joblib.load(d / "best_model.pkl")
        self.scaler  = joblib.load(d / "scaler.pkl")
        self.encoder = joblib.load(d / "ohe_encoder.pkl")
        self.meta    = joblib.load(d / "preprocessing_meta.pkl")
        logging.info(
            "PredictionTool ready. "
            f"Model: {type(self.model).__name__} | "
            f"Features expected by scaler: {len(self.scaler.feature_names_in_)}"
        )

    # Input handling
    def _parse_input(self, features) -> dict:
        """Accept either a JSON string or a dict; validate required keys."""
        if isinstance(features, dict):
            payload = features
        else:
            try:
                payload = json.loads(features)
            except (json.JSONDecodeError, TypeError) as e:
                raise ValueError(f"Invalid JSON input: {e}")

        missing = [f for f in self.REQUIRED_FIELDS if f not in payload]
        if missing:
            raise ValueError(
                f"Missing required field(s): {', '.join(missing)}. "
                f"All 14 census fields must be provided."
            )
        return payload

    # Preprocessing — identical to HW1 api.py:preprocess_input
    def preprocess(self, payload: dict):
        """Replicate the HW1 training-time preprocessing for a single sample."""
        df = pd.DataFrame([payload])

        # 1. Impute '?' with training-mode values
        for col, val in self.meta["fill_values"].items():
            if col in df.columns:
                df[col] = df[col].replace("?", val)

        # 2. IQR outlier clipping (training-set bounds)
        for col, (lower, upper) in self.meta["iqr_bounds"].items():
            if col in df.columns:
                df[col] = df[col].clip(lower=lower, upper=upper)

        # 3. Group rare native_country values
        top_countries = self.meta["top_countries"]
        df["native_country"] = df["native_country"].where(
            df["native_country"].isin(top_countries), other="Other"
        )

        # 4. Label-encode binary categorical columns (sex)
        for col, le in self.meta["label_encoders"].items():
            df[col] = le.transform(df[col])

        # 5. One-hot encode multi-value nominal columns
        ohe_cols = list(self.encoder.feature_names_in_)
        encoded_array = self.encoder.transform(df[ohe_cols])
        encoded_df = pd.DataFrame(
            encoded_array,
            columns=self.encoder.get_feature_names_out(ohe_cols),
            index=df.index,
        )
        df = pd.concat([df.drop(columns=ohe_cols), encoded_df], axis=1)

        # 6. Feature engineering (identical to HW1)
        df["capital_net"] = df["capital_gain"] - df["capital_loss"]
        df["hours_education_interaction"] = (
            df["hours_per_week"] * df["education_num"]
        )

        # 7. Align column order with training, then scale
        df = df[self.scaler.feature_names_in_]
        df = pd.DataFrame(
            self.scaler.transform(df),
            columns=df.columns,
            index=df.index,
        )
        return df

    # Inference
    def _call(self, features_json: str):
        """
        Underlying function exposed to the agent. Returns a human-readable
        prediction string identical in content to the HW1 /predict response.
        """
        try:
            payload = self._parse_input(features_json)
            X = self.preprocess(payload)
            prob = float(self.model.predict_proba(X)[:, 1][0])
            pred = int(prob >= self.POSITIVE_THRESHOLD)
            label = ">50K" if pred == 1 else "<=50K"
            return (
                f"Prediction: {label} "
                f"(probability of >$50K: {prob:.1%}). "
                f"Decision threshold: {self.POSITIVE_THRESHOLD}."
            )
        except Exception as e:
            return f"Prediction failed: {e}"

    def as_langchain_tool(self):
        return StructuredTool.from_function(
            func=self._call,
            name=self.TOOL_NAME,
            description=self.TOOL_DESCRIPTION,
        )



#  Tool 3 Dataset statistics

class DatasetStatsTool:
    """
    Bonus third tool. Returns summary statistics for any column of the
    HW1 dataset on demand.

    Domain  : Adult Census Income (UCI dataset)
    Task    : Answer numerical questions about the HW1 dataset itself —
              e.g. "what is the average age in the dataset?",
              "how is education distributed?", "what's the class balance?".
    Source  : data/adult_census_income.csv

    Design choices
    ──────────────
    • Numerical columns return mean / median / std / min / max plus a
      sample size.
    • Categorical columns return the top-10 value counts and the number
      of unique categories.
    • Unknown columns return the list of valid column names so the LLM
      can self-correct.
    """

    TOOL_NAME = "dataset_stats"
    DATASET_PATH = "data/adult_census_income.csv"

    TOOL_DESCRIPTION = (
        "Return summary statistics for any column of the Homework 1 "
        "Adult Census Income dataset. "
        "Use this tool when the user asks about the dataset itself — for "
        "example: 'what is the average age in the dataset?', "
        "'how are occupations distributed?', 'what is the class balance?'. "
        "Input: a column name as a string. Valid columns include: "
        "age, workclass, fnlwgt, education, education_num, marital_status, "
        "occupation, relationship, race, sex, capital_gain, capital_loss, "
        "hours_per_week, native_country, income. "
        "Output: mean/median/std/min/max for numerical columns, or "
        "value counts for categorical columns."
    )

    def __init__(self, dataset_path: str = None):
        self.dataset_path = dataset_path or self.DATASET_PATH
        self.df: pd.DataFrame = None

    def load_dataset(self):
        logging.info(f"Loading dataset for stats tool from '{self.dataset_path}'...")
        if not Path(self.dataset_path).exists():
            raise FileNotFoundError(f"Dataset not found: {self.dataset_path}")
        self.df = pd.read_csv(self.dataset_path)
        # Clean: strip whitespace and replace '?' with NaN, matching HW1
        str_cols = self.df.select_dtypes("object").columns
        self.df[str_cols] = self.df[str_cols].apply(lambda c: c.str.strip())
        self.df = self.df.replace("?", np.nan)
        logging.info(
            f"DatasetStatsTool ready. Shape: {self.df.shape} | "
            f"Columns: {list(self.df.columns)}"
        )

    def _call(self, column: str):
        """Return summary statistics for one column."""
        col = column.strip()

        if col not in self.df.columns:
            return (
                f"Unknown column '{col}'. "
                f"Valid columns: {', '.join(self.df.columns)}."
            )

        series = self.df[col].dropna()
        n = len(series)

        if pd.api.types.is_numeric_dtype(series):
            return (
                f"Statistics for numerical column '{col}' (n={n}):\n"
                f"  mean   : {series.mean():.2f}\n"
                f"  median : {series.median():.2f}\n"
                f"  std    : {series.std():.2f}\n"
                f"  min    : {series.min():.2f}\n"
                f"  max    : {series.max():.2f}"
            )

        counts = series.value_counts().head(10)
        lines = [
            f"Value counts for categorical column '{col}' "
            f"(n={n}, unique={series.nunique()}):"
        ]
        for value, count in counts.items():
            pct = 100 * count / n
            lines.append(f"  {value:<25} {count:>7,}  ({pct:5.2f}%)")
        return "\n".join(lines)

    def as_langchain_tool(self):
        return StructuredTool.from_function(
            func=self._call,
            name=self.TOOL_NAME,
            description=self.TOOL_DESCRIPTION,
        )


#  Orchestrator, builds all three tools for the agent

class ToolBuilder:
    """
    Builds the list of LangChain tools that the LangGraph agent will use.

    run_pipeline() returns a list of three Tool objects:
        [retrieve_knowledge, predict_income, dataset_stats]

    The order matters only for the tool-selection prompt — the LLM picks
    based on the description, not the order.
    """

    def __init__(self):
        self.kb_tool      = KnowledgeBaseTool()
        self.pred_tool    = PredictionTool()
        self.stats_tool   = DatasetStatsTool()

    def run_pipeline(self):
        logging.info("=" * 60)
        logging.info("Building agent tools...")
        logging.info("=" * 60)

        self.kb_tool.load_pipeline()
        self.pred_tool.load_artifacts()
        self.stats_tool.load_dataset()

        tools = [
            self.kb_tool.as_langchain_tool(),
            self.pred_tool.as_langchain_tool(),
            self.stats_tool.as_langchain_tool(),
        ]

        logging.info(
            f"All tools ready: {[t.name for t in tools]}"
        )
        logging.info("=" * 60)
        return tools



# CLI smoke test (python -m src.tools)

if __name__ == "__main__":
    tools = ToolBuilder().run_pipeline()

    print("\n--- Smoke test: predict_income ---")
    sample = {
        "age": 39, "workclass": "State-gov", "fnlwgt": 77516,
        "education": "Bachelors", "education_num": 13,
        "marital_status": "Never-married", "occupation": "Adm-clerical",
        "relationship": "Not-in-family", "race": "White", "sex": "Male",
        "capital_gain": 2174, "capital_loss": 0,
        "hours_per_week": 40, "native_country": "United-States",
    }
    print(tools[1].invoke({"features_json": json.dumps(sample)}))

    print("\n--- Smoke test: dataset_stats ---")
    print(tools[2].invoke({"column": "age"}))
    print()
    print(tools[2].invoke({"column": "income"}))
