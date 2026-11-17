FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

COPY agent/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt fastapi uvicorn google-adk

COPY agent/ agent/
COPY web/ web/

ENV PORT=8080
ENV GOOGLE_CLOUD_PROJECT=qwiklabs-gcp-04-4753f87c7a64

EXPOSE 8080

CMD ["python3", "-m", "uvicorn", "web.server:app", "--host", "0.0.0.0", "--port", "8080"]
