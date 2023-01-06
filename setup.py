# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import subprocess
from setuptools import setup

cwd = os.path.dirname(os.path.abspath(__file__))
version_txt = os.path.join(cwd, 'version.txt')
with open(version_txt, 'r') as f:
    version = f.readline().strip()

try:
    sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=cwd).decode('ascii').strip()
except Exception:
    sha = 'Unknown'
package_name = 'functorch'

# if os.getenv('BUILD_VERSION'):
#     version = os.getenv('BUILD_VERSION')
# elif sha != 'Unknown':
#     version += '+' + sha[:7]


requirements = [
    "torch>=1.13.1,<1.13.2",
]

extras = {}
extras["aot"] = ["networkx", ]


if __name__ == '__main__':
    try:
        setup(
            # Metadata
            name=package_name,
            version=version,
            author='PyTorch Core Team',
            url="https://github.com/pytorch/functorch",
            description='JAX-like composable function transforms for PyTorch',
            license='BSD',

            # Package info
            packages=[],
            install_requires=requirements,
            extras_require=extras,
        )
    except Exception as e:
        print(e, file=sys.stderr)
        sys.exit(1)
