PY := python

setup:
	$(PY) -m venv .venv && . .venv/bin/activate && pip install -U pip && pip install -r requirements.txt

train:
	$(PY) src/train.py --config configs/config.yaml

# Example: make eval MODEL=artifacts/model_decision_tree_depth6_20250101-120000.joblib
eval:
	@if [ -z "$(MODEL)" ]; then echo "Usage: make eval MODEL=path/to/model.joblib"; exit 1; fi
	$(PY) src/evaluate.py --config configs/config.yaml --model $(MODEL)

# Example: make demo MODEL=artifacts/xxx.joblib JSON=sample.json
demo:
	@if [ -z "$(MODEL)" ] || [ -z "$(JSON)" ]; then echo "Usage: make demo MODEL=... JSON=..."; exit 1; fi
	$(PY) src/predict.py --model $(MODEL) --input $(JSON)

clean:
	rm -rf artifacts outputs .pytest_cache
