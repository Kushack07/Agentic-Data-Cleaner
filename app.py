import gradio as gr
from fastapi import FastAPI, Request
from env.environment import DataPrepEnv
import pandas as pd
import json

# Global dict to store current environment
envs = {
    "Easy": DataPrepEnv({"module": "tasks.task_easy"}),
    "Medium": DataPrepEnv({"module": "tasks.task_medium"}),
    "Hard": DataPrepEnv({"module": "tasks.task_hard"})
}

custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');

/* Base theme */
.gradio-container {
    font-family: 'Outfit', sans-serif !important;
    background: linear-gradient(135deg, #09090b 0%, #171720 100%) !important;
    color: #e2e8f0 !important;
    border: none !important;
}

/* Glassmorphism Panels */
.gr-box, .gr-panel, .gr-form, .gr-block {
    background: rgba(30, 41, 59, 0.45) !important;
    backdrop-filter: blur(12px) !important;
    -webkit-backdrop-filter: blur(12px) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 16px !important;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3) !important;
}

/* Stunning Gradients for Titles */
h1 {
    background: -webkit-linear-gradient(45deg, #4facfe 0%, #00f2fe 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 800 !important;
    text-align: center;
    letter-spacing: -1px;
}
.subtitle {
    text-align: center;
    color: #94a3b8;
    margin-bottom: 2rem !important;
}

/* Micro-animated Buttons */
button.primary {
    background: linear-gradient(90deg, #3b82f6 0%, #8b5cf6 100%) !important;
    border: none !important;
    transition: all 0.3s ease !important;
    transform: scale(1);
    font-weight: 600 !important;
    box-shadow: 0 4px 15px rgba(139, 92, 246, 0.4) !important;
}

button.primary:hover {
    transform: scale(1.02) translateY(-2px);
    box-shadow: 0 8px 25px rgba(139, 92, 246, 0.6) !important;
}

/* Hacker-style Textboxes */
.gr-textbox input, .gr-textbox textarea {
    font-family: monospace !important;
    background: rgba(0, 0, 0, 0.3) !important;
    border: 1px solid rgba(139, 92, 246, 0.3) !important;
    color: #a78bfa !important;
}

/* Tab styling and DataFrame */
.gr-tabs > button {
    border-radius: 8px 8px 0 0 !important;
    transition: 0.2s;
}
.gr-tabs > button.selected {
    background: rgba(139, 92, 246, 0.2) !important;
    border-bottom: 3px solid #8b5cf6 !important;
    font-weight: 600;
}
"""

def get_preview(difficulty):
    env = envs[difficulty]
    obs = env.reset()
    
    # Return raw Pandas DataFrame for stunning gr.Dataframe rendering
    df = env.current_df
    
    # Format missing counts into string
    missing_str = "\n".join([f"{k}: {v}" for k, v in obs.missing_value_counts.items() if v > 0])
    if not missing_str: missing_str = "None"
        
    return df.head(50), str(obs.duplicate_counts), missing_str, json.dumps(obs.summary_statistics, indent=2)

with gr.Blocks(title="DataPrepEnv Command Center", css=custom_css) as demo:
    gr.Markdown("# 🧪 DataPrepEnv Command Center")
    gr.Markdown("<p class='subtitle'>Welcome! This premium dashboard gives you an unparalleled view into the DataPrepEnv physics engine. Select a difficulty below and hit <b>Generate Environment</b> to materialize a messy dataset.</p>")
    
    with gr.Row():
        diff_dropdown = gr.Dropdown(choices=["Easy", "Medium", "Hard"], value="Hard", label="Environment Difficulty Level", scale=2)
        btn = gr.Button("🚀 Generate / Reset Environment", variant="primary", scale=1)
        
    with gr.Tabs():
        with gr.TabItem("📊 Raw Data Preview (Interactive)"):
            preview_box = gr.Dataframe(interactive=False)
            
        with gr.TabItem("⚙️ Environment Metrics & Targets"):
            with gr.Row():
                with gr.Column():
                    dupes_box = gr.Textbox(label="Target exact duplicate rows in dataset", lines=2)
                    missing_box = gr.Textbox(label="Missing values per column target", lines=10)
                with gr.Column():
                    stats_box = gr.Code(label="Real-time Summary Statistics (Useful for identifying outliers)", language="json")
                    
    btn.click(fn=get_preview, inputs=diff_dropdown, outputs=[preview_box, dupes_box, missing_box, stats_box])


app = FastAPI(title="DataPrepEnv API")

api_env = envs["Hard"]

@app.get("/")
def health():
    return {"status": "ok", "message": "DataPrepEnv server running. Visit /ui for dashboard."}

@app.post("/reset")
def reset():
    obs = api_env.reset()
    return obs.model_dump()

@app.post("/step")
async def step(request: Request):
    action = await request.json()
    obs, reward, done, info = api_env.step(action)
    return {
        "observation": obs.model_dump(),
        "reward": reward.model_dump(),
        "done": done,
        "info": info
    }

@app.get("/state")
def state():
    obs = api_env.state()
    return obs.model_dump()

# Mount Gradio dashboard to /ui
app = gr.mount_gradio_app(app, demo, path="/ui")

if __name__ == "__main__":
    import uvicorn
    print("Launching FastAPI Server on port 7860...")
    uvicorn.run(app, host="0.0.0.0", port=7860)
