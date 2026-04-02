# ── CodeReviewEnv Docker Image ───────────────────────────────────────────────
# Build  : docker build -t code-review-env .
# Run    : docker run -e OPENAI_API_KEY=sk-... code-review-env
# No LLM : docker run code-review-env python baseline/run_agent.py --no-llm
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.10-slim

# Metadata
LABEL maintainer="project contributors"
LABEL description="CodeReviewEnv-v0 – AI Code Review RL Environment"

# Prevent Python from writing .pyc files and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install dependencies first (layer-caching optimisation)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# Copy project source
COPY . .

# Make the baseline script executable
RUN chmod +x baseline/run_agent.py

# Default command: run heuristic baseline (no API key required)
CMD ["python", "baseline/run_agent.py", "--no-llm", "--episodes", "5"]

# ── Instructions ─────────────────────────────────────────────────────────────
# To run with OpenAI LLM:
#   docker run -e OPENAI_API_KEY=sk-... code-review-env \
#     python baseline/run_agent.py --episodes 5 --seed 42
#
# To run a specific difficulty:
#   docker run code-review-env python -c "
#     import sys; sys.path.insert(0, '.')
#     import gym, code_review_env
#     env = gym.make('CodeReviewEnv-v0', difficulty='hard')
#     obs = env.reset(); print(obs['diff_patch'][:200])
#   "
# ─────────────────────────────────────────────────────────────────────────────
