#!/bin/bash
ARCH=$(uname -m)
SCRIPT_DIR=$(dirname $(readlink -f $0))
ROOT_DIR="${SCRIPT_DIR}/.."
OUTPUT_DIR="${ROOT_DIR}/output"
TOOL_DIR="${OUTPUT_DIR}/tools"

git config --local core.autocrlf input

# Configure dependencies for development
pip install --no-cache -r ${ROOT_DIR}/requirements-dev.txt

# Configure dependencies for runtime
pip install --no-cache -r ${ROOT_DIR}/requirements.txt

pre-commit autoupdate
pre-commit install

mkdir -p ${TOOL_DIR}
pushd ${TOOL_DIR}

tmp_dir=$(mktemp -d)
case ${ARCH,,} in
    x86_64)

        pushd ${tmp_dir}
            wget https://github.com/mikefarah/yq/releases/download/v4.28.1/yq_linux_amd64.tar.gz ${tmp_dir}
            tar -xvzf yq_linux_amd64.tar.gz
            mv yq_linux_amd64 ${TOOL_DIR}/yq
        popd
        chmod +x yq
        ;;
    *)
        echo "Failed to install yq for arch: ${ARCH,,}"
        ;;
esac
rm -rf ${tmp_dir}

popd
