.PHONY: audit smoke test serve perf

PYTHON ?= python3
TI_PYTHONPATH ?= src

audit:
	PYTHONPATH=$(TI_PYTHONPATH) $(PYTHON) -m torchinferno.cli audit

test:
	$(PYTHON) -m pytest

smoke:
	PYTHONPATH=$(TI_PYTHONPATH) $(PYTHON) -m torchinferno.cli dsv4-smoke --device cpu --batch-size 1 --prompt-tokens 3 --new-tokens 2
	PYTHONPATH=$(TI_PYTHONPATH) $(PYTHON) -m torchinferno.cli deepseek-smoke --device cpu --batch-size 1 --prompt-tokens 3 --new-tokens 2 --cache-backend paged
	PYTHONPATH=$(TI_PYTHONPATH) $(PYTHON) -m torchinferno.cli serve-smoke --device cpu --cache-backend paged --page-size 2 --new-tokens 2

serve:
	PYTHONPATH=$(TI_PYTHONPATH) $(PYTHON) -m torchinferno.cli serve-smoke --device cpu --cache-backend paged --page-size 2 --new-tokens 2

perf:
	PYTHONPATH=$(TI_PYTHONPATH) $(PYTHON) -m torchinferno.cli perf-smoke --device cpu --heads 2 --seq-len 8 --head-dim 8 --value-dim 8
