PY := python

setup:
	$(PY) -m venv .venv && . .venv/bin/activate && pip install -U pip && pip install -r requirements.txt

# Train with config-specified model (decision_tree or random_forest)
train:
	$(PY) src/train.py --config configs/config.yaml

# Train specific model type
train-dt:
	$(PY) src/train.py --config configs/config.yaml --model decision_tree

train-rf:
	$(PY) src/train.py --config configs/config.yaml --model random_forest

# Example: make eval MODEL=artifacts/model_random_forest_n100_depth10_20250101-120000.joblib
eval:
	@if [ -z "$(MODEL)" ]; then echo "Usage: make eval MODEL=path/to/model.joblib"; exit 1; fi
	$(PY) src/evaluate.py --config configs/config.yaml --model $(MODEL)

# Evaluate the latest model
eval-latest:
	@MODEL=$$(ls -t artifacts/*.joblib 2>/dev/null | head -n1); \
	if [ -z "$$MODEL" ]; then echo "No models found in artifacts/"; exit 1; fi; \
	echo "Evaluating latest model: $$MODEL"; \
	$(PY) src/evaluate.py --config configs/config.yaml --model "$$MODEL"

# Example: make demo MODEL=artifacts/xxx.joblib JSON=sample.json
demo:
	@if [ -z "$(MODEL)" ] || [ -z "$(JSON)" ]; then echo "Usage: make demo MODEL=... JSON=..."; exit 1; fi
	$(PY) src/predict.py --model $(MODEL) --input $(JSON)

# Compare models
compare:
	@echo "Available models:" && ls -lh artifacts/*.joblib 2>/dev/null || echo "No models found"

# Run streamlit app
app:
	streamlit run app.py

clean:
	rm -rf artifacts outputs .pytest_cache

clean-outputs:
	rm -rf outputs

test:
	$(PY) -m pytest tests/ -v