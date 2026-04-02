# Use a lightweight python image
FROM python:3.10-slim

# Set environment
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV WANDB_DISABLED=true

# Add a non-root user (important for HF Spaces security)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user
ENV PATH="/home/user/.local/bin:$PATH"

# Set working directory
WORKDIR $HOME/app

# Copy repo content
COPY --chown=user . $HOME/app/

# Install requirements
RUN pip install --no-cache-dir -r requirements.txt

# Run the inference test script by default
CMD ["python", "inference.py"]
