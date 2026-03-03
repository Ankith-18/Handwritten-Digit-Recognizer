# -----------------------------
# Stage 1: Builder
# -----------------------------
FROM python:3.10-slim AS builder

WORKDIR /install

# Install system dependencies (only if needed)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for caching)
COPY requirements.txt .

# Upgrade pip and install dependencies into custom folder
RUN pip install --upgrade pip \
    && pip install --prefix=/install --no-cache-dir -r requirements.txt


# -----------------------------
# Stage 2: Final Runtime Image
# -----------------------------
FROM python:3.10-slim

WORKDIR /app

# Copy installed dependencies from builder stage
COPY --from=builder /install /usr/local

# Copy application code
COPY . .

# Expose Streamlit default port
EXPOSE 8501

# Streamlit config to allow external access
ENV PYTHONUNBUFFERED=1
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_ENABLECORS=false
ENV STREAMLIT_SERVER_PORT=8501

# Run Streamlit
# Run Streamlit web app (listen on all interfaces)
CMD ["streamlit", "run", "6_web_interface.py", "--server.address=0.0.0.0", "--server.port=8501"]