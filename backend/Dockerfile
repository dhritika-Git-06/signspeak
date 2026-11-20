FROM python:3.11-slim

WORKDIR /app
ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1

COPY backend/requirements-serve.txt /tmp/requirements-serve.txt
RUN pip install --no-cache-dir -r /tmp/requirements-serve.txt

COPY signspeak /app/signspeak
COPY backend/app /app/backend/app
COPY models /app/models

EXPOSE 8765
CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8765"]
