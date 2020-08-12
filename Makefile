
define GetFromJson
$(shell node -p "require('./global.json').$(1)")
endef

CONTAINER_IMAGE := $(call GetFromJson,container_image)
WORK_DIR        := $(call GetFromJson,container_workdir)
RESULTS_DIR     := $(call GetFromJson,container_resultsdir)
TB_PORT         := $(call GetFromJson,tb_port)

AWS_CERT := ~/.ssh/eric-key.pem
gpu_1 := i-0f450034a80158fd1
trainer_1 := i-051d019a04d16e0be
trainer_2 := i-0b0e52f926dae48e7
trainer_3 := i-03ac0ecb80701b20c
trainer_4 := i-0fefcba4a5c464eb1
trainer_5 := i-043bcb928e14bef1a
trainer_6 := i-0f6bf5087225511ed

start_docker:
	@docker run -it \
		--gpus all \
		--name trainer \
		-p $(TB_PORT):$(TB_PORT) \
		--mount src=`pwd`,target=$(WORK_DIR),type=bind \
		--mount src=`pwd`/mjkey.txt,target=/root/.mujoco/mjkey.txt,type=bind \
		--runtime=nvidia \
		-w $(WORK_DIR) \
		$(CONTAINER_IMAGE)

stop_docker: clean
	@docker stop trainer
	@docker rm trainer

clean:
	@find . -name "*.pyc" -delete
	@find . -name "*.pyo" -delete
	@find . -name "*.DS_Store" -delete
	@find . -path "*__pycache__/*"
	@find . -name "__pycache__" -type d -delete
	@find . -name "_vizdoom.ini" -type d -delete

# Args:
# 	logdir
view:
	@tensorboard --port $(TB_PORT) --logdir ./results/$(logdir)

# Args:
# 	instance
start_aws:
	@aws ec2 start-instances --instance-ids $($(instance))

# Args:
# 	instance
stop_aws:
	@aws ec2 stop-instances --instance-ids $($(instance))


# Args:
# 	instance
describe:
	@aws ec2 describe-instances --output table --instance-ids $($(instance))

# Args:
# 	instance
# 	(optional) tb
connect: 
	@if [ "$(tb)" = "true" ]; then\
		ssh -i $(AWS_CERT) -L $(TB_PORT):127.0.0.1:$(TB_PORT) ubuntu@`aws ec2 describe-instances --output text --instance-ids $($(instance)) --query "Reservations[0].Instances[0].PublicDnsName"`;\
	else\
		ssh -i $(AWS_CERT) ubuntu@`aws ec2 describe-instances --output text --instance-ids $($(instance)) --query "Reservations[0].Instances[0].PublicDnsName"`;\
	fi

# Args:
# 	instance
# 	name
#	run
pull:
	scp -r -i $(AWS_CERT) ubuntu@`aws ec2 describe-instances --output text --instance-ids $($(instance)) --query "Reservations[0].Instances[0].PublicDnsName"`:~/curiosity_baselines/results/$(name)/$(run) ./results/$(name)/tmp

# Args:
# 	instance
# 	src
# 	dest
pull_general:
	scp -r -i $(AWS_CERT) ubuntu@`aws ec2 describe-instances --output text --instance-ids $($(instance)) --query "Reservations[0].Instances[0].PublicDnsName"`:$(src) $(dest)

# Args:
# 	instance
# 	name
# 	run
push:
	scp -r -i $(AWS_CERT) ./results/$(name)/$(run) ubuntu@`aws ec2 describe-instances --output text --instance-ids $($(instance)) --query "Reservations[0].Instances[0].PublicDnsName"`:~/curiosity_baselines/results/$(name)
	
# Args:
# 	instance
# 	src
# 	dest
push_general:
	scp -r -i $(AWS_CERT) $(src) ubuntu@`aws ec2 describe-instances --output text --instance-ids $($(instance)) --query "Reservations[0].Instances[0].PublicDnsName"`:$(dest)








