# v2 pipeline. `make all` is offline and rebuilds everything from data/raw/.
# Only `make fetch` touches the network (and MP needs MP_API_KEY in .env).
# No make on Windows? `python run_all.py <step>` does the same thing.

PY ?= python
export PYTHONPATH := src

.PHONY: all fetch data features train evaluate report uncertainty explain mlp test

all: data features train evaluate report uncertainty explain mlp

fetch:
	$(PY) -m a3bx3.fetch.oqmd
	$(PY) -m a3bx3.fetch.jarvis
	$(PY) -m a3bx3.fetch.mp

data:
	$(PY) -m a3bx3.curate

features: data
	$(PY) -m a3bx3.features

train: features
	$(PY) -m a3bx3.evaluate train

evaluate: train
	$(PY) -m a3bx3.evaluate evaluate

report:
	$(PY) -m a3bx3.evaluate report

uncertainty: evaluate
	$(PY) -m a3bx3.uncertainty

explain: evaluate
	$(PY) -m a3bx3.explain

mlp: evaluate
	$(PY) -m a3bx3.mlp

test:
	$(PY) -m pytest -q tests
