---
title: DataPrepEnv
emoji: 🧹
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# 🧪 DataPrepEnv: Intelligent Data Cleaning Environment

🚀 **Live Premium Interactive Dashboard:** [View on Hugging Face Spaces](https://kushluvdivya-dataprepenv.hf.space/)  
🔗 **Official OpenEnv Submission Space Base:** [Hugging Face Repository](https://huggingface.co/spaces/KushLuvdivya/DataPrepEnv)

DataPrepEnv is a highly robust, fully **OpenEnv-compliant** evaluation environment designed to train, assess, and benchmark Autonomous AI Agents on the chaotic real-world task of **tabular data cleaning**.

Instead of simple toy problems, DataPrepEnv models real pipeline engineering. Agents are given messy, organically corrupted Pandas DataFrames (with type drift, exact row duplicates, missing variables, and extreme outliers) and penalized or rewarded deterministically for their ability to synthesize a biologically clean, statistically valid output.

---

## 🎯 Hackathon Pre-Submission Compliance Verification

This repository was specifically engineered to achieve a **100% pass rate** on the strict OpenEnv evaluation mechanics:
1. **Hugging Face Spaces Native**: We bypass standard local execution and provide a hardened `Dockerfile` that maps `uvicorn` explicitly to port `7860`, allowing seamless Hugging Face Space generation.
2. **Server Architecture Matrix**: Unlike basic Gradio apps, our `app.py` operates a **FastAPI layer** to securely expose `/reset`, `/step`, and `/state` API routes for automated remote pings, while gracefully mounting an ultra-premium Gradio UI directly to the root `/` route.
3. **Bulletproof `inference.py`**:
   - Implements flawless `[START]`, `[STEP]`, `[END]` JSON log stream pipelines.
   - Restricts API credentials to exactly what the Auto-Grader injects (`API_BASE_URL`, `MODEL_NAME`, `HF_TOKEN`, `LOCAL_IMAGE_NAME`) deployed strictly at the global scope without indentation.
   - Defers exclusively to the `OpenAI()` client interface to run multi-turn agent feedback loops.

---

## 🏗️ Architecture & Technology Stack

- **Core Engine**: Python 3.10+, Pandas, Numpy
- **Server Framework**: FastAPI + Uvicorn
- **Observation Engine**: **Pydantic V2** Typed Models for airtight data validation.
- **Frontend / Dashboard**: **Gradio** powered by highly custom injected CSS (`Glassmorphism` styling, interactive `gr.Dataframe` structures, and dynamic web-fonts) for a visual experience that redefines what OpenEnv visualization should look like.
- **Deployment Strategy**: Docker (Ubuntu base -> Python environment -> Containerized Web Application).

## 🔌 Action & Observation Spaces

### Action Space
- `action_type`: one of `fill_missing`, `drop_duplicates`, `remove_outliers`, `normalize_column`, `convert_data_type`, or `submit`
- `column`: optional target column for column-specific cleaning operations
- `parameters`: free-form dictionary for strategies such as `{"fill_value": "median"}` or `{"target_type": "numeric"}`

### Observation Space
- `dataset_preview`: markdown preview of the current dataset snapshot
- `missing_value_counts`: missing values per column
- `duplicate_counts`: exact duplicate row count
- `column_types`: current inferred dtype for each column
- `summary_statistics`: descriptive statistics for numeric columns
- `reward` and `done`: transition signals returned by the OpenEnv server during `step()`

---

## 🏎️ The Tasks & Graders

DataPrepEnv features three highly calibrated difficulty tiers, automatically governed by `openenv.yaml`.

### 1. Easy Mode
- **Corruption Dynamics**: Introduces simple missing numerical thresholds (NaN insertion) and simple duplicate rows. No categorical corruption.
- **Grader Matrix**: Validates the correct row index deduplication against a hashed ground truth dataset. Checks basic count statistics.

### 2. Medium Mode
- **Corruption Dynamics**: Introduces aggressive feature corruption, including mapping `Int` types to `Strings`, injecting formatting inconsistencies, and heavy categorical omissions. 
- **Grader Matrix**: Evaluates the strict enforcement of statistical distributions. If an Agent uses a "Mean-Fill" strategy instead of a more appropriate "Median-Fill" for skewed columns, the Dense Reward system penalizes them gracefully.

### 3. Hard Mode
- **Corruption Dynamics**: Chaotic extreme outliers exceeding 6-sigma bounds are maliciously layered into standard deviation mappings. Compound row-level null propagation requires advanced iterative missing-value resolution.
- **Grader Matrix**: Programmatic deterministic grading utilizing rigorous MSE (Mean Squared Error) measurements of the normalized post-cleaned DataFrame against an untouched ground truth DataFrame.

---

## 🖥️ Local Usage & Testing

To test the multi-turn reinforcement loop locally:

1. **Install Dependencies**:
```bash
pip install -r requirements.txt
```

2. **Run The UI Server**:
```bash
python app.py
```
Visit `http://localhost:7860/` to manually explore the task physics and corrupted data geometries.

3. **Run AI Inference Agent**:
Configure your OpenAI endpoint or compatible proxy before firing off the testing agent.
```bash
export API_BASE_URL="https://api.openai.com/v1"
export MODEL_NAME="gpt-4o"
export HF_TOKEN="your_auth_key_here"

python inference.py
```
