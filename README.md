---
title: DataPrepEnv
emoji: 🧹
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---
# DataPrepEnv

🚀 **Live Interactive Dashboard:** [View Premium UI on Hugging Face](https://kushluvdivya-dataprepenv.hf.space/ui/)  
🔗 **Official OpenEnv Submission Space:** [Hugging Face Space](https://huggingface.co/spaces/KushLuvdivya/DataPrepEnv)

An OpenEnv-compliant reinforcement learning environment that trains AI agents to automatically clean messy datasets. Simulated tasks involve resolving issues such as exact row duplicates, missing numerical or categorical values, type corruption, and severe numerical outliers.

## Project Structure

- `env/`: Contains the core environment logic including the state tracking, action parsing logic, and the dense reward system.
- `tasks/`: Dynamically generated baseline datasets simulating "Easy", "Medium", and "Hard" scenarios of messy data.
- `graders/`: Deterministic programmatic graders comparing the AI agent's resulting datasets against perfectly cleaned ground truth datasets.
- `openenv.yaml`: The official metadata linking tasks to graders to be recognized by OpenEnv tools.

## Requirements

This environment runs with Python 3.10 and is compatible with `< 8GB RAM` requirements for HuggingFace Spaces.

```bash
pip install -r requirements.txt
```

## Running Inference Testing

The baseline inference tester interacts with the environment using the OpenAI API.

Ensure you set the required authentication prior to running inference:

```bash
export OPENAI_API_KEY="your-api-key"
export API_BASE_URL="https://api.openai.com/v1"
export MODEL_NAME="gpt-4o"
```

To run a test episode on the Hard Task:

```bash
python inference.py
```

The script adheres to the logging requirement structure (`[START]`, `[STEP]`, `[END]`).

## Deployment to Hugging Face Spaces

This project contains a `Dockerfile` customized for HuggingFace platform.
When creating your Space:
1. Choose "Docker" as the hardware backend format.
2. Select standard limits (vCPU 2, RAM 8GB limits).
3. The Space will automatically build the environment layer and run inference if passed parameters.
