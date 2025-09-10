FROM python:3.12-slim

WORKDIR /app

# System deps for faiss/sentence-transformers
RUN apt-get update && apt-get install -y --no-install-recommends \
    g++ libopenblas-dev libomp-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app
COPY . .

# Default config expects ./data and ./artifacts directories
RUN mkdir -p data artifacts

# Streamlit networking config
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ENABLECORS=false
ENV STREAMLIT_SERVER_ENABLEXSRSFPROTECTION=false

EXPOSE 8501
CMD ["streamlit", "run", "streamlit_app.py"]
