
TIMESTAMP:=$(shell date '+%Y-%m-%d-%T-%a')
SBATCH_MAIL:=credmond@uchicago.edu
DSI_PARTITION:=general

SCRIPTS_DIR:=scripts
LOG_DIR:=logs

.PHONY: clean
clean:
	find . | grep -E ".*/__pycache__$$" | xargs rm -rf

.PHONY: test
test:
	pytest tests -s

.PHONY: train
train:
	mkdir -p $(LOG_DIR)/$(TIMESTAMP)
	sbatch \
	--partition=$(DSI_PARTITION) \
	--output=$(LOG_DIR)/$(TIMESTAMP)/res.txt \
	--error=$(LOG_DIR)/$(TIMESTAMP)/err.txt \
	--mail-user=$(SBATCH_MAIL) \
	$(SCRIPTS_DIR)/train.slurm

.PHONY: ts
ts:
	echo $(TIMESTAMP)
	sleep 5
	echo $(TIMESTAMP)
