#!/bin/bash

ARCH=$(uname -m)

SCRIPT_DIR=$(dirname $(readlink -f $0))
ROOT_DIR="${SCRIPT_DIR}/.."
ENV_FILE="${ROOT_DIR}/.env"
DIST_YAML="${ROOT_DIR}/dist.yml"
OUTPUT_DIR="${ROOT_DIR}/output"
TOOL_DIR="${OUTPUT_DIR}/tools"

YQ=${TOOL_DIR}/yq

IMAGE_TAG=$(${YQ} '.docker.tag' ${DIST_YAML})
DOCKER_REPO=$(${YQ} '.docker.repo' ${DIST_YAML})
VERSION_LEN=$(${YQ} '.docker.version | length' ${DIST_YAML})

if [ ${VERSION_LEN} -gt 0 ]; then
    pushd ${ROOT_DIR} > /dev/null
    BASE_VER=$(${YQ} ".docker.version[0]" ${DIST_YAML})
    BASE_NAME=${DOCKER_REPO}/${IMAGE_TAG}:${BASE_VER}
    DOCKER_BUILDKIT=1 docker build -f docker/Dockerfile -t ${BASE_NAME} .
    echo ${BASE_NAME}
    IMAGES=${BASE_NAME}

    for i in $(seq 1 $((VERSION_LEN-1))); do
        VER=$(${YQ} ".docker.version[${i}]" ${DIST_YAML})
        NAME=${DOCKER_REPO}/${IMAGE_TAG}:${VER}
        docker tag ${BASE_NAME} ${NAME} > /dev/null
        echo ${NAME}
        IMAGES="${IMAGES} ${NAME}"
    done
    popd > /dev/null
fi

if [[ "${1}" == "--push" ]]; then
    for image in ${IMAGES}; do
        docker push ${image}
    done
fi
