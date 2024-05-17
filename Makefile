app = pipeline

# bootstrap
_conda = conda

# python + dependencies
conda_yml = environment.yml

workdir = .
conda_name = .conda
conda_dir = $(workdir)/$(conda_name)

dev_reqs = $(workdir)/requirements.dev.txt
dev_reqs_ok = $(conda_dir)/.dev_reqs_ok

scripts_dir = ./scripts

# logs
log_dir          = ./logs
TIMESTAMP        := $(shell date "+%Y-%m-%dT%H-%a")
run_logs         = $(log_dir)/$(TIMESTAMP)
err_file_name    = err.txt
err_file         = $(run_logs)/$(err_file_name)
output_file_name = res.txt
output_file      = $(run_logs)/$(output_file_name)

# includes SBATCH_MAIL, conda_yml, DSI_PARTITION and CGFP_DIR
include .env

# helper
RUNNING = @echo "+ $@\n"
LAST_LOGS:=$(shell find logs |grep -E "logs/[^/]*$$" |sort |tail -n 1)

# effectively exes
PYTHON = $(conda_dir)/bin/python
SBATCH = $(_conda) run -p $(conda_dir) sbatch


.PHONY: default run test lint py clean

default: run

run: cmd ?= -m $(app) $(args)
run: args ?=
run: $(PYTHON)
	$(RUNNING)
	$(PYTHON) $(cmd)

test: cmd  ?= -m pytest $(args)
test: args ?= -s
test: 
	$(RUNNING)
	$(running)
	python $(cmd)

clean: conda=false
clean:
	@$(RUNNING)
	@find . | grep -E ".*/__pycache__$$" | xargs rm -rf
	@find . |grep -E ".*\.egg-info$$" |xargs rm -rf
	@$(conda) && echo "cleaning conda" && rm -rf $(conda_dir) || echo "skipping conda"

lint: $(dev_reqs_ok)
	$(RUNNING)
	$(PYTHON) -m ruff format
	$(PYTHON) -m ruff check --fix

$(PYTHON): $(conda_yml)
	@$(RUNNING)
	$(_conda) env update -f $(conda_yml) -p $(conda_dir)
	@test -f $(PYTHON) && touch $(PYTHON)

$(dev_reqs_ok): $(PYTHON) $(dev_reqs)
	$(RUNNING)
	$(PYTHON) -m pip install -r $(dev_reqs)
	@touch $(dev_reqs_ok)

train: args ?=
train: cmd  ?= $(SBATCH) $(args)
train: $(PYTHON) $(run_logs)
	$(RUNNING)
	$(SBATCH) \
	--mem=64GB \
	--partition=$(DSI_PARTITION) \
	--nodes=1 \
	--gres=gpu:1 \
	--output=$(output_file) \
	--error=$(err_file) \
	--mail-user=$(SBATCH_MAIL) \
	$(scripts_dir)/train-cgfp.slurm

$(run_logs):
	$(RUNNING)
	mkdir -p $(run_logs)

###

.PHONY: last-logs
last-logs:
	$(RUNNING)
	cat $(LAST_LOGS)/$(output_file_name)

.PHONY: last-errs
last-errs:
	$(RUNNING)
	cat $(LAST_LOGS)/$(err_file_name)

##### HUGGINGFACE #####

# TODO: update this to take model as an argumentsq
# Usage: make update-dev-model MODEL_DIR=2024-02-05_10-56
update-dev-model:
	$(RUNNING)
	@echo "Moving model files from $(MODEL_DIR)..."
	cp $(CGFP_DIR)model-files/$(MODEL_DIR)/pytorch_model.bin $(CGFP_DIR)huggingface/cgfp-distilbert/
	cp $(CGFP_DIR)model-files/$(MODEL_DIR)/config.json $(CGFP_DIR)huggingface/cgfp-distilbert/
	@echo "Committing changes..."
	cd $(CGFP_DIR)huggingface/cgfp-distilbert/ && git add -u && git commit -m "update dev model" && git push

# Usage: This upgrades the dev model to production
update-prod-model:
	$(RUNNING)
	@echo "Moving model files from $(MODEL_DIR)..."
	cp $(CGFP_DIR)huggingface/cgfp-classifier-dev/pytorch_model.bin $(CGFP_DIR)huggingface/cgfp-classifier/
	cp $(CGFP_DIR)huggingface/cgfp-classifier-dev/config.json $(CGFP_DIR)huggingface/cgfp-classifier/
	@echo "Committing changes..."
	cd $(CGFP_DIR)huggingface/cgfp-classifier/ && git add -u && git commit -m "update prod model" && git push
