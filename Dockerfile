FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY drift/ drift/
RUN useradd --create-home appuser
USER appuser
ENTRYPOINT ["python", "-m", "drift.detect"]
