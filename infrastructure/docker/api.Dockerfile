FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml /app/
COPY packages/ /app/packages/
COPY apps/api/requirements.txt /app/apps/api/requirements.txt

RUN pip install --no-cache-dir -r /app/apps/api/requirements.txt \
    && pip install --no-cache-dir Pillow numpy scipy torch torchvision

COPY apps/api/ /app/apps/api/
COPY services/ /app/services/

EXPOSE 8000

CMD ["uvicorn", "apps.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
