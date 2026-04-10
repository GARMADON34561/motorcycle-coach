FROM python:3.11-slim

WORKDIR /app

COPY server ./server
COPY requirements.txt .
COPY pyproject.toml .
COPY uv.lock .
COPY inference.py .
COPY openenv.yaml .

RUN pip install --no-cache-dir -r requirements.txt
RUN pip install -e .

EXPOSE 7860

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
