FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860 \
    QC_WORK_ROOT=/data/runs

WORKDIR /app

COPY requirements.txt pyproject.toml ./
RUN python -m pip install --upgrade pip \
    && python -m pip install -r requirements.txt

COPY . .

RUN useradd --create-home --uid 10001 qcapp \
    && mkdir -p /data/runs \
    && chown -R qcapp:qcapp /data

USER qcapp

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:7860/', timeout=3)" || exit 1

CMD ["python", "scripts/run_qc_web.py"]
