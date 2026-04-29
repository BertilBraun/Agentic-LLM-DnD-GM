FROM python:3.12-slim
WORKDIR /app
COPY state-mcp/pyproject.toml .
RUN pip install --no-cache-dir . && opentelemetry-bootstrap -a install
COPY shared/ shared/
COPY state-mcp/app/ app/
CMD ["sh", "-c", "exec opentelemetry-instrument uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8001}"]
