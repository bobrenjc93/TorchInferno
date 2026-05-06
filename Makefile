.PHONY: audit smoke test serve perf profile profile-region profile-pattern disagg

PYTHON ?= python3
TI_PYTHONPATH ?= src
PROFILE_DIR ?= .torchinferno_runs/dsv4-cpu
REGION_PROFILE_DIR ?= .torchinferno_runs/region-cpu
PATTERN_PROFILE_DIR ?= .torchinferno_runs/pattern-cpu
REGION ?= layers.0.attn
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

profile-region:
	PYTHONPATH=$(TI_PYTHONPATH) $(PYTHON) -m torchinferno.cli profile-region $(REGION_PROFILE_DIR) --device cpu --region $(REGION) --batch-size 1 --tokens 3 --warmup 0 --iters 1

profile-pattern:
	PYTHONPATH=$(TI_PYTHONPATH) $(PYTHON) -m torchinferno.cli profile-pattern $(PATTERN_PROFILE_DIR) --device cpu --batch-size 1 --tokens 3 --hidden-size 16 --warmup 0 --iters 1

disagg:
	PYTHONPATH=$(TI_PYTHONPATH) $(PYTHON) -m torchinferno.cli disagg-init $(DISAGG_DIR) --prefill-ranks 1 --decode-ranks 1 --device cpu
