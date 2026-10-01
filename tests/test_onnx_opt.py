import numpy as np
import pytest


def test_asymmetric_convolution_rewrite_preserves_cpu_output(tmp_path):
    onnx = pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    from onnx import TensorProto, helper, numpy_helper

    from deepfake.occlusion.onnx_opt import cuda_xseg_bytes

    weight = np.arange(9, dtype=np.float32).reshape(1, 1, 3, 3) / 9
    graph = helper.make_graph(
        [helper.make_node("ConvTranspose", ["input", "weight"], ["output"], pads=[0, 0, 1, 1], strides=[2, 2])],
        "asymmetric",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 1, 4, 4])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 1, 8, 8])],
        [numpy_helper.from_array(weight, "weight")],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=9)
    path = tmp_path / "model.onnx"
    onnx.save(model, path)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    old = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
    new = ort.InferenceSession(cuda_xseg_bytes(path), options, providers=["CPUExecutionProvider"])
    data = np.arange(16, dtype=np.float32).reshape(1, 1, 4, 4)
    np.testing.assert_allclose(old.run(None, {"input": data})[0], new.run(None, {"input": data})[0], atol=1e-6)
