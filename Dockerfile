FROM registry.suse.com/bci/python:3.12 AS loader

WORKDIR /app

RUN pip install uv

COPY pyproject.toml .

RUN uv pip install --system -r pyproject.toml

COPY . .

RUN uv run python loader.py

FROM registry.suse.com/bci/python:3.12

WORKDIR /app

RUN pip install uv

COPY pyproject.toml .

RUN uv pip install --system -r pyproject.toml

COPY . .

RUN mkdir chroma_db

COPY --from=loader  /app/chroma_db /app/chroma_db

RUN uv run python main.py