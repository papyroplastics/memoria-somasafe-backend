
api-run:
	uv run fastapi dev api --host 0.0.0.0

api-test:
	uv run pytest api/test.py

model-train:
	uv run train.py feature-mlp
