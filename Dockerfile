FROM python:3.12-slim

WORKDIR /app

COPY requirements-ml.txt .
RUN pip install --no-cache-dir -r requirements-ml.txt

COPY app ./app
COPY ml ./ml

ENV VIGIA_WORKER=1

VOLUME ["/app/data"]
EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
