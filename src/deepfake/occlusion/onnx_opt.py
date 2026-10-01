"""Exact asymmetric ConvTranspose → symmetric ConvTranspose + crop rewrite.

ORT CUDA does not support asymmetric padding on ConvTranspose. XSeg uses
pads=[0,0,1,1], equivalent to zero padding then removing the final row/column.
The model stays in memory; no executable vendor code or unverified cache loads.
"""

from __future__ import annotations


def cuda_xseg_bytes(path) -> bytes:
    import onnx
    from onnx import helper

    model = onnx.load(str(path), load_external_data=False)
    nodes = []
    for node in model.graph.node:
        pads = next((a for a in node.attribute if a.name == "pads"), None)
        if node.op_type != "ConvTranspose" or pads is None or list(pads.ints) != [0, 0, 1, 1]:
            nodes.append(node)
            continue
        pads.ints[:] = [0, 0, 0, 0]
        result = node.output[0]
        node.output[0] = result + "_uncropped"
        nodes.append(node)
        constants = []
        for suffix, values in (("starts", [0, 0]), ("ends", [-1, -1]), ("axes", [2, 3])):
            name = result + "_" + suffix
            constants.append(name)
            model.graph.initializer.append(helper.make_tensor(name, onnx.TensorProto.INT64, [2], values))
        nodes.append(helper.make_node("Slice", [node.output[0], *constants], [result], name=node.name + "_crop"))
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    onnx.checker.check_model(model)
    return model.SerializeToString()
