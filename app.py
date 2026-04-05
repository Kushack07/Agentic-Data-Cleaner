import gradio as gr
from fastapi import FastAPI, Request
from env.environment import DataPrepEnv
import pandas as pd

# Global dict to store current environment
envs = {
    "Easy": DataPrepEnv({"module": "tasks.task_easy"}),
    "Medium": DataPrepEnv({"module": "tasks.task_medium"}),
    "Hard": DataPrepEnv({"module": "tasks.task_hard"})
}

def get_preview(difficulty):
    env = envs[difficulty]
    obs = env.reset()
    
    # We can also render the dataframe as html using pandas if we want a nicer table
    df = env.current_df
    html_table = df.head(15).to_html(classes="table", index=False)
    
    # Format missing counts into string
    missing_str = "\n".join([f"{k}: {v}" for k, v in obs.missing_value_counts.items() if v > 0])
    if not missing_str: missing_str = "None"
        
    return html_table, str(obs.duplicate_counts), missing_str, str(obs.summary_statistics)

with gr.Blocks(title="DataPrepEnv Dashboard", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🧪 DataPrepEnv: Interactive Dashboard", elem_classes="text-center")
    gr.Markdown("Welcome! This UI allows you to visualize the output of your programmatic OpenEnv Tasks. Select a difficulty below and hit **Generate / Reset** to create a messy dataset.")
    
    with gr.Row():
        diff_dropdown = gr.Dropdown(choices=["Easy", "Medium", "Hard"], value="Hard", label="Task Difficulty Setup")
        btn = gr.Button("🔄 Generate / Reset Environment", variant="primary")
        
    with gr.Row():
        dupes_box = gr.Textbox(label="Total Exact Duplicate Rows")
        missing_box = gr.Textbox(label="Missing Values Count")
        
    with gr.Row():
        stats_box = gr.Textbox(label="General Summary Statistics (Useful for Outliers)")
        
    with gr.Row():
        gr.Markdown("### Dataset Preview (First 15 Rows)")
    with gr.Row():
        preview_box = gr.HTML()
        
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
