.PHONY: audit smoke test serve perf profile disagg

PYTHON ?= python3
TI_PYTHONPATH ?= src
PROFILE_DIR ?= .torchinferno_runs/dsv4-cpu
DISAGG_DIR ?= .torchinferno_disagg

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

profile:
	PYTHONPATH=$(TI_PYTHONPATH) $(PYTHON) -m torchinferno.cli profile-run $(PROFILE_DIR) --device cpu --batch-size 1 --prompt-tokens 3 --new-tokens 2 --warmup 0

disagg:
	PYTHONPATH=$(TI_PYTHONPATH) $(PYTHON) -m torchinferno.cli disagg-init $(DISAGG_DIR) --prefill-ranks 1 --decode-ranks 1 --device cpu
