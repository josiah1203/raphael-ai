FROM python:3.11-slim
WORKDIR /app
RUN pip install --no-cache-dir uv
COPY pyproject.toml README.md ./
COPY src ./src
RUN uv pip install --system -e .
ENV RAPHAEL_SERVICE_PORT=8099
EXPOSE 8099
CMD ["uvicorn", "raphael_ai.app:app", "--host", "0.0.0.0", "--port", "8099"]
