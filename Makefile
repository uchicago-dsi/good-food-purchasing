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

DATE := $(shell date)
LOG_FILE_PREFIX = ${logs_dir}${DATE}
output_file = ${LOG_FILE_PREFIX}_res.txt
err_file = ${LOG_FILE_PREFIX}_err.txt

# includes SBATCH_MAIL, conda_yml, DSI_PARTITION and CGFP_DIR
include .env

# effectively exes
SBATCH = $(_conda) run -n ${ENV_NAME} sbatch

.PHONY: train

$(conda_updated): $(conda_yml)
	mkdir -p ${build_dir}
	$(_conda) env update -f $(conda_yml)
	@touch $(conda_updated)

train: ${conda_updated}
	${SBATCH} \
	--mem=64GB \
	--partition=$(DSI_PARTITION) \
	--nodes=1 \
	--gres=gpu:1 \
	--output="$(output_file)" \
	--error="$(err_file)" \
	--mail-user=$(SBATCH_MAIL) \
	$(scripts_dir)train-cgfp.slurm


# TODO: could add run-pipeline if we want later

##### HUGGINGFACE #####

# TODO: update this to take model as an arguments
# Usage: make update-dev-model MODEL_DIR=2024-02-05_10-56
update-dev-model:
	$(RUNNING)
	@echo "Moving model files from $(MODEL_DIR)..."
	cp $(CGFP_DIR)model-files/$(MODEL_DIR)/pytorch_model.bin $(CGFP_DIR)huggingface/cgfp-distilbert/
	cp $(CGFP_DIR)model-files/$(MODEL_DIR)/config.json $(CGFP_DIR)huggingface/cgfp-distilbert/
	@echo "Committing changes..."
	cd $(CGFP_DIR)huggingface/cgfp-distilbert/ && git add -u && git commit -m "update dev model" && git push

# TODO: this is probably deprecated...review huggingface saving process
# Usage: This upgrades the dev model to production
update-prod-model:
	$(RUNNING)
	@echo "Moving model files from $(MODEL_DIR)..."
	cp $(CGFP_DIR)huggingface/cgfp-classifier-dev/pytorch_model.bin $(CGFP_DIR)huggingface/cgfp-classifier/
	cp $(CGFP_DIR)huggingface/cgfp-classifier-dev/config.json $(CGFP_DIR)huggingface/cgfp-classifier/
	@echo "Committing changes..."
	cd $(CGFP_DIR)huggingface/cgfp-classifier/ && git add -u && git commit -m "update prod model" && git push
