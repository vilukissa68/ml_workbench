#!/usr/bin/env python3

import tvm
import torch
import os
from tvm import relay
from tvm.micro import export_model_library_format


def get_scripted_model(model, input_shape):
    input_data = torch.randn(input_shape)
    scripted_model = torch.jit.trace(model, input_data).eval()
    return scripted_model


def tvm_import_pytorch_model(model, input_shape, input_name="input0"):
    scripted_model = get_scripted_model(model, input_shape)
    shape_list = [(input_name, input_shape)]
    mod, params = relay.frontend.from_pytorch(scripted_model, shape_list)
    return mod, params


def tvm_compile(mod, params, build_dir, target=None):
    # Create build dir for model files
    if not os.path.isdir(build_dir):
        os.makedirs(build_dir)

    # Default to C export
    if not target:
        target = tvm.target.Target("c")

    runtime = tvm.relay.backend.Runtime("crt", {"system-lib": False})
    executor = tvm.relay.backend.Executor(
        "aot", {"unpacked-api": True, "interface-api": "c", "link-params": True}
    )

    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(
            mod, target=target, runtime=runtime, params=params, executor=executor
        )

    # Save to disk
    file_format_str = "{name}_c.{ext}"
    lib_file_name = os.path.join(
        build_dir, file_format_str.format(name="model", ext="tar")
    )
    export_model_library_format(lib, lib_file_name)
