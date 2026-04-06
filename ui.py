import json

import gradio as gr

from env.environment import DataPrepEnv


envs = {
    "Easy": DataPrepEnv({"module": "tasks.task_easy"}),
    "Medium": DataPrepEnv({"module": "tasks.task_medium"}),
    "Hard": DataPrepEnv({"module": "tasks.task_hard"}),
}


custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');

.gradio-container {
    font-family: 'Outfit', sans-serif !important;
    background: linear-gradient(135deg, #09090b 0%, #171720 100%) !important;
    color: #e2e8f0 !important;
    border: none !important;
}

.gr-box, .gr-panel, .gr-form, .gr-block {
    background: rgba(30, 41, 59, 0.45) !important;
    backdrop-filter: blur(12px) !important;
    -webkit-backdrop-filter: blur(12px) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 16px !important;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3) !important;
}

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

.gr-textbox input, .gr-textbox textarea {
    font-family: monospace !important;
    background: rgba(0, 0, 0, 0.3) !important;
    border: 1px solid rgba(139, 92, 246, 0.3) !important;
    color: #a78bfa !important;
}

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
    df = env.current_df

    missing_str = "\n".join(
        [f"{key}: {value}" for key, value in obs.missing_value_counts.items() if value > 0]
    )
    if not missing_str:
        missing_str = "None"

    return (
        df.head(50),
        str(obs.duplicate_counts),
        missing_str,
        json.dumps(obs.summary_statistics, indent=2),
    )


with gr.Blocks(title="DataPrepEnv Command Center", css=custom_css) as demo:
    gr.Markdown("# DataPrepEnv Command Center")
    gr.Markdown(
        "<p class='subtitle'>Generate a messy dataset preview, inspect duplicates, "
        "missing values, and numeric summaries, and sanity-check the environment "
        "before submission.</p>"
    )

    with gr.Row():
        diff_dropdown = gr.Dropdown(
            choices=["Easy", "Medium", "Hard"],
            value="Hard",
            label="Environment Difficulty Level",
            scale=2,
        )
        btn = gr.Button("Generate / Reset Environment", variant="primary", scale=1)

    with gr.Tabs():
        with gr.TabItem("Raw Data Preview"):
            preview_box = gr.Dataframe(interactive=False)

        with gr.TabItem("Environment Metrics"):
            with gr.Row():
                with gr.Column():
                    dupes_box = gr.Textbox(label="Exact duplicate rows", lines=2)
                    missing_box = gr.Textbox(label="Missing values per column", lines=10)
                with gr.Column():
                    stats_box = gr.Code(
                        label="Summary statistics",
                        language="json",
                    )

    btn.click(
        fn=get_preview,
        inputs=diff_dropdown,
        outputs=[preview_box, dupes_box, missing_box, stats_box],
    )
