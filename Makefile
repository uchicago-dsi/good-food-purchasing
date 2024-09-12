app = pipeline

# bootstrap
_conda = conda

# python + dependencies
conda_yml = environment.yml

workdir = ./
scripts_dir = ${workdir}scripts/
build_dir = ${workdir}build/
logs_dir = ${workdir}logs/
conda_udpated = ${build_dir}conda_updated.txt

DATE := $(shell date +"%Y%m%d_%H%M%S")
LOG_FILE_PREFIX = ${logs_dir}${DATE}
output_file = ${LOG_FILE_PREFIX}_res.txt
err_file = ${LOG_FILE_PREFIX}_err.txt

# includes SBATCH_MAIL, conda_yml, DSI_PARTITION and CGFP_DIR
include .env

# effectively exes
SBATCH = $(_conda) run -n ${ENV_NAME} sbatch

.PHONY: train validate_api_key

validate_api_key:
ifndef WANDB_API_KEY
	@echo "WARNING: WANDB_API_KEY IS NOT SET"
	@echo "ADD your wandb api key as an environment variable before continuining"
	$(error "WANDB_API_KEY is not set. Please set it to continue.")
endif

$(conda_updated): $(conda_yml)
	mkdir -p ${build_dir}
	$(_conda) env update -f $(conda_yml)
	@touch $(conda_updated)

train: ${conda_updated} validate_api_key
	${SBATCH} \
	--partition=$(DSI_PARTITION) \
	--output="$(output_file)" \
	--error="$(err_file)" \
	--mail-user=$(SBATCH_MAIL) \
	$(scripts_dir)train-cgfp.slurm


# TODO: add run-pipeline if we want later

##### HUGGINGFACE #####

# Usage: make update-huggingface MODEL_DIR=/full/path/to/model/ MODEL_NAME=distilbert
update-huggingface:
	$(RUNNING)
	@echo "Moving model files from $(MODEL_DIR)..."
	cp $(MODEL_DIR)/* $(CGFP_DIR)huggingface/cgfp-$(MODEL_NAME)/
	@echo "Committing changes..."
	cd $(CGFP_DIR)huggingface/cgfp-$(MODEL_NAME) && git add . && git commit -m "update model with $(MODEL_DIR)" && git push
