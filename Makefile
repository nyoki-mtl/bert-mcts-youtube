PROJECT ?= bert-mcts
DATADIR ?= ${PWD}/data
WORKSPACE ?= /workspace/$(PROJECT)
DOCKER_IMAGE ?= ${PROJECT}:latest

SHMSIZE ?= 100G
DOCKER_OPTS := \
			--name ${PROJECT} \
			--rm -it \
			--shm-size=${SHMSIZE} \
			-v ${PWD}:${WORKSPACE} \
			-v ${DATADIR}:${WORKSPACE}/data \
			-v ${LOG_DIR}:${WORKSPACE}/work_dirs/logs \
      -w ${WORKSPACE} \
			--ipc=host \
			--network=host \
			--gpus all

docker-build:
	docker build -f docker/Dockerfile -t ${DOCKER_IMAGE} .

docker-start-interactive: docker-build
	docker run ${DOCKER_OPTS} ${DOCKER_IMAGE} bash

docker-start-jupyter: docker-build
	docker run ${DOCKER_OPTS} ${DOCKER_IMAGE} \
		bash -c "jupyter lab --port=8888 --ip=0.0.0.0 --allow-root --no-browser"

docker-run: docker-build
	docker run ${DOCKER_OPTS} ${DOCKER_IMAGE} \
		bash -c "${COMMAND}"
