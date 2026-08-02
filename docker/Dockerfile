FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

COPY pyproject.toml README.md README_ZH.md ./
COPY src ./src

RUN pip install --upgrade pip \
    && pip install .

COPY alembic.ini ./
COPY migrations ./migrations
COPY scripts ./scripts

EXPOSE 8000

CMD ["uvicorn", "hvac_fdd.api:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
