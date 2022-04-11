define GetFromJson
$(shell node -p "require('./global.json').$(1)")
endef

CONTAINER_IMAGE := $(call GetFromJson,container_image)
WORK_DIR        := $(call GetFromJson,container_workdir)
RESULTS_DIR     := $(call GetFromJson,container_resultsdir)
TB_PORT         := $(call GetFromJson,tb_port)
TB_PORT_LOCAL   := $(call GetFromJson,tb_port_local)

# CPU Version
start_docker:
	@docker run -it \
		--shm-size 2G \
		--name trainer \
		-p $(TB_PORT):$(TB_PORT) \
		--mount src=`pwd`,target=$(WORK_DIR),type=bind \
		-w $(WORK_DIR) \
		$(CONTAINER_IMAGE)
		# --mount src=`pwd`/mjkey.txt,target=/root/.mujoco/mjkey.txt,type=bind \

# GPU Version
start_docker_gpu:
	@docker run -it \
		--shm-size 2G \
		--name trainer \
		--runtime=nvidia \
		--gpus all \
		-p $(TB_PORT):$(TB_PORT) \
		--mount src=`pwd`,target=$(WORK_DIR),type=bind \
		-w $(WORK_DIR) \
		$(CONTAINER_IMAGE)
		# --mount src=`pwd`/mjkey.txt,target=/root/.mujoco/mjkey.txt,type=bind \

stop_docker: clean
	@docker stop trainer
	@docker rm trainer

clean:
	@find . -name "*.pyc" -delete
	@find . -name "*.pyo" -delete
	@find . -name "*.DS_Store" -delete
	@find . -path "*__pycache__/*"
	@find . -name "__pycache__" -type d -delete
