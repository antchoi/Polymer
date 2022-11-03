#!/bin/bash
SCRIPT_DIR=$(dirname $(readlink -f $0))
ROOT_DIR="${SCRIPT_DIR}/.."
ENV_FILE="${ROOT_DIR}/.env"

export $(grep -v '^#' ${ENV_FILE} | xargs -d '\n')

export UNITTEST_TARGET="${1}"
export UNITTEST_CONTAINER="container-for-unittest"
export INTERNAL_NETWORK="network-for-unittest"

function CLEAR() {
    container_id=$(docker ps -f "name=${UNITTEST_CONTAINER}" -aq)
    if [[ -n ${container_id} ]]; then
        echo "Clear orphan unittest container ${container_id}"
        docker rm -f ${container_id}
    fi
}

CLEAR

docker network create ${INTERNAL_NETWORK}

pushd ${ROOT_DIR} && {
    COMPOSE_DOCKER_CLI_BUILD=1 DOCKER_BUILDKIT=1 docker compose -f docker/docker-compose.test.yml up ${2}
} && popd

CLEAR

docker network rm ${INTERNAL_NETWORK}
