FROM python:3.12-slim

# Build tools are not needed: every dependency below ships manylinux wheels.
WORKDIR /app

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Long-lived dev container; work happens via `docker exec`.
CMD ["sleep", "infinity"]
