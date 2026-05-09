#!/usr/bin/env python3

from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup


setup_args = generate_distutils_setup(
    packages=["adv_module"],
    package_dir={"": "."},
    package_data={
        "adv_module": [
            "inference_config.yaml",
            "model_config.yaml",
            "policy_checkpoint.pt",
        ],
    },
)

setup(**setup_args)
