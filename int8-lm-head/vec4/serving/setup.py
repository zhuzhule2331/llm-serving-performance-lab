from setuptools import setup
setup(
    name="qwen-int8-lmhead-vec4-followup",
    version="0.1.0",
    py_modules=["vec4_vllm_plugin"],
    entry_points={"vllm.general_plugins":["qwen_int8_vec4_head=vec4_vllm_plugin:register"]},
)
