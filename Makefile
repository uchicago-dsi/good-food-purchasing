include .env # includes SBATCH_MAIL, CONDA_ENV_PATH, DSI_PARTITION and CGFP_DIR

SCRIPTS_DIR:=scripts
LOG_DIR:=logs
SRC_DIR:=src

SRC_FILES := $(shell find src -type f -not -path "*/__pycache__/*" -not -name "*.egg-info")

OUTPUT_FILE:=res.txt
ERR_FILE:=err.txt

TIMESTAMP:=$(shell date '+%Y-%m-%d-%T-%a')
RUN_LOGS:=$(LOG_DIR)/$(TIMESTAMP)

LAST_LOGS:=$(shell find logs |grep -E "logs/[^/]*$$" |sort |tail -n 1)

CONDA_ENV_FILE:=environment.yml
CONDA_RUN = conda run -p $(CONDA_ENV_PATH)

.PHONY: clean
clean:
	find . | grep -E ".*/__pycache__$$" | xargs rm -rf
	find . |grep -E ".*\.egg-info$$" |xargs rm -rf

# TODO: add something to clean model files and checkpoints

##### ENVIRONMENT SETUP #####

$(CONDA_ENV_PATH):
	conda env create --file $(CONDA_ENV_FILE)

$(CONDA_ENV_PATH)/.timestamp: $(CONDA_ENV_FILE) $(CONDA_ENV_PATH)
	conda env update --file $(CONDA_ENV_FILE) --prefix $(CONDA_ENV_PATH) -v
	touch $(CONDA_ENV_PATH)/.timestamp

src: $(SRC_FILES) $(CONDA_ENV_PATH)/.timestamp
	conda run -p $(CONDA_ENV_PATH) pip install -e .
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

##### HUGGINGFACE #####

# TODO: update this to take model as an argument
# Usage: make update-dev-model MODEL_DIR=2024-02-05_10-56
update-dev-model:
	@echo "Moving model files from $(MODEL_DIR)..."
	cp $(CGFP_DIR)model-files/$(MODEL_DIR)/pytorch_model.bin $(CGFP_DIR)huggingface/cgfp-distilbert/
	cp $(CGFP_DIR)model-files/$(MODEL_DIR)/config.json $(CGFP_DIR)huggingface/cgfp-distilbert/
	@echo "Committing changes..."
	cd $(CGFP_DIR)huggingface/cgfp-distilbert/ && git add -u && git commit -m "update dev model" && git push

# Usage: This upgrades the dev model to production
update-prod-model:
	@echo "Moving model files from $(MODEL_DIR)..."
	cp $(CGFP_DIR)huggingface/cgfp-classifier-dev/pytorch_model.bin $(CGFP_DIR)huggingface/cgfp-classifier/
	cp $(CGFP_DIR)huggingface/cgfp-classifier-dev/config.json $(CGFP_DIR)huggingface/cgfp-classifier/
	@echo "Committing changes..."
	cd $(CGFP_DIR)huggingface/cgfp-classifier/ && git add -u && git commit -m "update prod model" && git push

