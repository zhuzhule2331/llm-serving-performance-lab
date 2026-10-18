from setuptools import setup

setup(
    name="qwen-int8-lmhead-experiment",
    version="0.1.0",
    py_modules=["int8_vllm_plugin"],
    entry_points={
        "vllm.general_plugins": [
            "qwen_int8_head=int8_vllm_plugin:register"
        ]
    },
)
