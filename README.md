# Eval RAG approaches

## Prepare data

```
BASE_URL="http://ollama:11434" uv run python loader.py --splitter=all  --persist-dir=emdebbing --persist-dir-docstore=docstore
```

This will create the embeddings for all the approaches inside the emdebbing and docstore folders

## Run retrievers

```
BASE_URL="http://ollama:11434"  uv run fastapi dev main.py
```

It will run the retrievers in the following urls:

- http://localhost:8000/markdown
- http://localhost:8000/recursive
- http://localhost:8000/hierarchical
- http://localhost:8000/summary

Example:

```
curl -X POST "http://localhost:8000/recursive" \                                                                                          
-H "Content-Type: application/json" \
-d '{"q": "what is a GitRepo??"}'
```

## Run measures

```
BASE_URL="http://ollama:11434" uv run python measurement.py
```