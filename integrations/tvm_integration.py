import tvm
import torch
import os
from tvm import relay
from tvm.micro import export_model_library_format
from .utils import extract_tar


def tvm_export_model(model, data_shapes, model_type="pytorch"):
    if model_type == "pytorch":
        input_shape = data_shapes[next(iter(data_shapes))]
        mod, params = tvm_import_pytorch_model(model, input_shape)
    elif model_type == "tflite":
        mod, params = tvm_import_tflite_model(model, data_shapes)
    return mod, params


def get_scripted_model(model, input_shape):
    input_data = torch.randn(input_shape)
    scripted_model = torch.jit.trace(model, input_data).eval()
    return scripted_model


def tvm_import_pytorch_model(model, input_shape, input_name="input0"):
    scripted_model = get_scripted_model(model, input_shape)
    shape_list = [(input_name, input_shape)]
    mod, params = relay.frontend.from_pytorch(scripted_model, shape_list)
    return mod, params


def tvm_import_tflite_model(model, shape_dict):
    dtype_dict = {}
    input_type = "int8"
    for input_key in shape_dict:
        dtype_dict[input_key] = input_type
    print(shape_dict)
    print(dtype_dict)

    mod, params = relay.frontend.from_tflite(model, shape_dict, dtype_dict)
    return mod, params


def tvm_compile(mod, params, build_dir, target=None):
    # Create build dir for model files
    if not os.path.isdir(build_dir):
        os.makedirs(build_dir)

    # Default to C export
    target = tvm.target.Target("c")

    runtime = tvm.relay.backend.Runtime("crt", {"system-lib": False})
    executor = tvm.relay.backend.Executor(
        "aot", {"unpacked-api": True, "interface-api": "c", "link-params": True}
    )

    config = {"tir.disable_vectorize": True}

    disable_passes = ["FoldConstant"]
    with tvm.transform.PassContext(
        opt_level=3, config=config, disabled_pass=disable_passes
    ):
        lib = relay.build(
            mod, target=target, runtime=runtime, params=params, executor=executor
        )

    # Save to disk
    file_format_str = "{name}_c.{ext}"
    lib_file_name = os.path.join(
        build_dir, file_format_str.format(name="model", ext="tar")
    )
    export_model_library_format(lib, lib_file_name)
    extract_tar(lib_file_name, build_dir)
