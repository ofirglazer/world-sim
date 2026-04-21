# ── 3DVES Web Service — Dockerfile ───────────────────────────────────────
# Single-stage build; no GUI dependencies (matplotlib excluded).
# uvicorn runs with a single worker — the SimulationWorker is a thread,
# not a process, so multiple uvicorn workers would each have their own
# SessionRegistry with no shared state.  Use one worker until Redis
# pub/sub is introduced for horizontal scaling.

FROM python:3.11-slim

# System packages: none required beyond the slim base for this stack
WORKDIR /app

# Install Python dependencies first (layer cached unless requirements.txt changes)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the full project tree
COPY . .

# Ensure the JSONL log directory exists inside the container
RUN mkdir -p sim_logs

# Expose the uvicorn port
EXPOSE 8000

# Non-root user for security
RUN useradd -m simuser && chown -R simuser /app
USER simuser

CMD ["uvicorn", "sim3dves.web.server:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1"]
