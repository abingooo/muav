#!/usr/bin/env python3

from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup


setup_args = generate_distutils_setup(
    packages=["mpc_module"],
    package_dir={"": "."},
    package_data={
        "mpc_module": [
            "model_config.yaml",
            "mpc_config.yaml",
        ],
    },
)

setup(**setup_args)
