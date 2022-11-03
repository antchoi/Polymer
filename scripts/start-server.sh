#!/bin/bash
SCRIPT_DIR=$(dirname $(readlink -f $0))
ROOT_DIR="${SCRIPT_DIR}/.."
ENV_FILE="${ROOT_DIR}/.env"

export $(grep -v '^#' ${ENV_FILE} | xargs -d '\n')

export INTERNAL_NETWORK="network-for-test"

docker network create ${INTERNAL_NETWORK}

pushd ${ROOT_DIR} && {
    COMPOSE_DOCKER_CLI_BUILD=1 DOCKER_BUILDKIT=1 docker compose -f docker/docker-compose.yml up -d ${1}
} && popd
