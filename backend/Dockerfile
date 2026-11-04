# Imagen de la API. Contexto de build: backend/. Sin PyTorch (ver requirements.txt).
FROM python:3.11-slim-bookworm AS deps
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN python -m venv /opt/venv
COPY requirements.txt /tmp/requirements.txt
RUN /opt/venv/bin/pip install -r /tmp/requirements.txt

FROM python:3.11-slim-bookworm AS runtime
# ffmpeg/ffprobe: los usa app/services/encoder.py para H.264 y para sondear videos.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 1000 kinetia \
    && useradd --uid 1000 --gid 1000 --no-create-home --home-dir /nonexistent --shell /usr/sbin/nologin kinetia \
    && mkdir -p /models /data/results /data/db /shared-tmp/api \
    && chown -R 1000:1000 /models /data \
    && chown 1000:1000 /shared-tmp/api \
    && chmod 1777 /shared-tmp
COPY --from=deps /opt/venv /opt/venv
WORKDIR /srv
COPY app/ ./app/
COPY animation_engine/ ./animation_engine/
RUN rm -rf animation_engine/tests animation_engine/requirements.txt \
    && find . -name '__pycache__' -type d -prune -exec rm -rf {} + \
    && /opt/venv/bin/python -m compileall -q app animation_engine
ENV PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TMPDIR=/shared-tmp/api \
    HOME=/shared-tmp/api
USER 1000:1000
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--no-access-log"]
