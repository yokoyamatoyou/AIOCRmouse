FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System deps for opencv / streamlit
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN python -m pip install --upgrade pip && pip install -r requirements.txt

COPY src ./src
COPY templates ./templates
COPY sitecustomize.py ./sitecustomize.py
COPY AIOCR_GUI_使い方.md ./AIOCR_GUI_使い方.md

# Create writable dirs
RUN mkdir -p workspace database uploads

ENV PORT=8080
EXPOSE 8080

# Use Cloud Run provided $PORT
CMD ["sh", "-c", "python -m streamlit run src/app/main.py --server.headless=true --server.address=0.0.0.0 --server.port=${PORT}"]


