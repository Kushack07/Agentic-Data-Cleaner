import gradio as gr

from openenv.core.env_server import create_app

from models import DataPrepAction, DataPrepObservation
from server.dataprep_environment import DataPrepEnvironment
from ui import demo


api_app = create_app(
    DataPrepEnvironment,
    DataPrepAction,
    DataPrepObservation,
    env_name="DataPrepEnv",
)

app = gr.mount_gradio_app(api_app, demo, path="/")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=7860)
