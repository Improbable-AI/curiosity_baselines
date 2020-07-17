
define GetFromJson
$(shell node -p "require('./global.json').$(1)")
endef

CONTAINER_IMAGE := $(call GetFromJson,container_image)
WORK_DIR        := $(call GetFromJson,container_workdir)
RESULTS_DIR     := $(call GetFromJson,container_resultsdir)
TB_PORT         := $(call GetFromJson,tb_port)

start:
	@docker run -it \
		--name trainer \
		-p $(TB_PORT):$(TB_PORT) \
		--mount src=`pwd`,target=$(WORK_DIR),type=bind \
		--mount src=`pwd`/mjkey.txt,target=/root/.mujoco/mjkey.txt,type=bind \
		-w $(WORK_DIR) \
		$(CONTAINER_IMAGE)

stop:
	@docker stop trainer
	@docker rm trainer

clean:
	@find . -name "*.pyc" -delete
	@find . -name "*.pyo" -delete
	@find . -name "*.DS_Store" -delete
	@find . -path "*__pycache__/*"
	@find . -name "__pycache__" -type d -delete
	@find . -name "_vizdoom.ini" -type d -delete
