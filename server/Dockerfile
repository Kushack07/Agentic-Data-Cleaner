FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV WANDB_DISABLED=true

RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR $HOME/app
COPY --chown=user . $HOME/app/
RUN pip install --no-cache-dir -r requirements.txt

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
