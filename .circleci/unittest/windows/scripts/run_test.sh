#!/usr/bin/env bash

set -e

export IN_CI=1
mkdir test-reports

eval "$(./conda/Scripts/conda.exe 'shell.bash' 'hook')"
conda activate ./env

this_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
source "$this_dir/set_cuda_envs.sh"

python -m torch.utils.collect_env

git_version=$(python -c "import torch.version; print(torch.version.git_version)")
echo git_version

git clone https://github.com/pytorch/pytorch.git
pushd pytorch
git checkout $git_version
pushd functorch

pytest test
