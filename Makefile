
SBATCH_MAIL:=credmond@uchicago.edu
DSI_PARTITION:=general

SCRIPTS_DIR:=scripts
LOG_DIR:=logs
SRC_DIR:=src

SRC_FILES := $(shell find src -type f -not -path "*/__pycache__/*" -not -name "*.egg-info")

OUTPUT_FILE:=res.txt
ERR_FILE:=err.txt

TIMESTAMP:=$(shell date '+%Y-%m-%d-%T-%a')
RUN_LOGS:=$(LOG_DIR)/$(TIMESTAMP)

LAST_LOGS:=$(shell find logs |grep -E "logs/[^/]*$$" |sort |tail -n 1)

# CONDA_ENV_PATH := ./tmp/conda/cgfp
CONDA_ENV_PATH := /net/projects/cgfp
CONDA_ENV_FILE := environment.yml

CONDA_RUN = conda run -p $(CONDA_ENV_PATH)

.PHONY: clean
clean:
	find . | grep -E ".*/__pycache__$$" | xargs rm -rf
	find . |grep -E ".*\.egg-info$$" |xargs rm -rf

##### ENVIRONMENT SETUP #####

$(CONDA_ENV_PATH):
	conda env create --file $(CONDA_ENV_FILE) --prefix $(CONDA_ENV_PATH) -v

$(CONDA_ENV_PATH)/.timestamp: $(CONDA_ENV_FILE) $(CONDA_ENV_PATH)
	conda env update --file $(CONDA_ENV_FILE) --prefix $(CONDA_ENV_PATH) -v
	touch $(CONDA_ENV_PATH)/.timestamp

src: $(SRC_FILES) $(CONDA_ENV_PATH)/.timestamp
	conda run -p $(CONDA_ENV_PATH) pip install .
	touch src

requirements.dev.txt: $(CONDA_ENV_PATH)/.timestamp
	conda run -p $(CONDA_ENV_PATH) pip install -r requirements.dev.txt
	touch requirements.dev.txt

.PHONY: env
env: $(CONDA_ENV_PATH)/.timestamp src requirements.dev.txt
	echo "Checking environment\n"

##### DEVELOPMENT #####

.PHONY: test
test: env
	$(CONDA_RUN) pytest tests -s

##### TRAINING #####

.PHONY: train
train: env
	mkdir -p $(RUN_LOGS)
	$(CONDA_RUN) sbatch \
	--mem=64GB \
	--partition=$(DSI_PARTITION) \
	--nodes=1 \
	--gres=gpu:1 \
	--output=$(RUN_LOGS)/$(OUTPUT_FILE) \
	--error=$(RUN_LOGS)/$(ERR_FILE) \
	--mail-user=$(SBATCH_MAIL) \
	$(SCRIPTS_DIR)/train.slurm

.PHONY: last-logs
last-logs:
	echo ""
	cat $(LAST_LOGS)/$(OUTPUT_FILE)

.PHONY: last-errs
last-errs:
	echo ""
	cat $(LAST_LOGS)/$(ERR_FILE)
