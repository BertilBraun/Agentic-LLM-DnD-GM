FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml .
RUN pip install --no-cache-dir . && opentelemetry-bootstrap -a install
COPY alembic.ini .
COPY alembic/ alembic/
COPY app/ app/
EXPOSE 8000
CMD ["sh", "-c", "alembic upgrade head && exec opentelemetry-instrument uvicorn app.main:app --host 0.0.0.0 --port 8000"]
