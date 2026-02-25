FROM python:3.12-slim

WORKDIR /app

# Install the hypervisor package and API dependencies
RUN pip install --no-cache-dir \
    agent-hypervisor[api] \
    redis \
    pyyaml \
    httpx

COPY app/ /app/app/
COPY config/ /app/config/

EXPOSE 8000 8501
