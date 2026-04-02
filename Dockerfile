# ── CodeReviewEnv Docker Image ───────────────────────────────────────────────
# Hugging Face Spaces Docker SDK – serves FastAPI on port 7860
#
# Build  : docker build -t code-review-env .
# Run    : docker run -p 7860:7860 code-review-env
# No API : docker run code-review-env python baseline/run_agent.py --no-llm
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.10-slim

# Metadata
LABEL maintainer="project contributors"
LABEL description="CodeReviewEnv-v0 – AI Code Review RL Environment"

# Prevent Python from writing .pyc files and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# HF Spaces requires a non-root user
RUN useradd -m -u 1000 appuser
WORKDIR /app

# Install dependencies first (layer-caching optimisation)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# Copy project source
COPY . .

# Set correct ownership
RUN chown -R appuser:appuser /app
USER appuser

# HF Spaces requires port 7860
EXPOSE 7860

# Default command: serve interactive Gradio UI on port 7860
# To serve the REST API instead: uvicorn app.main:app --host 0.0.0.0 --port 7860
CMD ["python", "gradio_app.py"]

# ── Instructions ─────────────────────────────────────────────────────────────
# REST API endpoints (once running):
#   GET  /           → welcome page
#   GET  /health     → health check
#   GET  /reset      → start new episode
#   POST /step       → submit action {"action": 3, "difficulty": "easy"}
#   GET  /render     → current state
#   GET  /scores     → leaderboard
#   GET  /actions    → list actions
#   GET  /docs       → Swagger UI
# ─────────────────────────────────────────────────────────────────────────────
