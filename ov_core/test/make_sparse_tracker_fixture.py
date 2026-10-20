#!/usr/bin/env python3
"""Write a deterministic ONNX contract fixture, NOT a trained tracking network.

Small protobuf wire encoder avoids adding onnx/PyTorch as build dependencies.
Fields follow the official onnx/onnx.proto (IR 8, standard opset 11).
The graph exercises all five real adapter inputs, recurrent flow, and fixed patches.
"""
import struct
import sys
from pathlib import Path


def varint(value):
    data = bytearray()
    while value > 127:
        data.append((value & 127) | 128)
        value >>= 7
    return bytes(data + bytes([value]))


def integer(field, value):
    return varint(field << 3) + varint(value)


def binary(field, value):
    if isinstance(value, str):
        value = value.encode()
    return varint((field << 3) | 2) + varint(len(value)) + value


def value_info(name, dimensions):
    shape = b''.join(binary(1, integer(1, d) if isinstance(d, int) else binary(2, d)) for d in dimensions)
    tensor_type = integer(1, 1) + binary(2, shape)
    return binary(1, name) + binary(2, binary(1, tensor_type))


def constant(name, dimensions, values):
    return (binary(1, b''.join(varint(d) for d in dimensions)) + integer(2, 1) +
            binary(8, name) + binary(9, struct.pack('<' + 'f' * len(values), *values)))


def node(operation, inputs, output, attributes=()):
    return (b''.join(binary(1, name) for name in inputs) + binary(2, output) + binary(3, output) +
            binary(4, operation) + b''.join(binary(5, attr) for attr in attributes))


def mean(input_name, output, axes):
    attributes = [binary(1, 'axes') + integer(20, 7) + binary(8, b''.join(varint(a) for a in axes)),
                  binary(1, 'keepdims') + integer(20, 2) + integer(3, 0)]
    return node('ReduceMean', [input_name], output, attributes)


nodes = [
    node('Mul', ['flow_previous', 'half'], 'momentum'),
    node('Add', ['points_previous', 'momentum'], 'predicted'),
    node('Add', ['predicted', 'translation'], 'points_current'),
    mean('image_previous', 'mean_previous', [2, 3]),
    mean('image_current', 'mean_current', [2, 3]),
    node('Sub', ['mean_current', 'mean_previous'], 'image_difference'),
    node('Mul', ['image_difference', 'zero'], 'image_zero'),
    mean('patch_reference', 'reference_mean', [2, 3]),
    node('Add', ['reference_mean', 'half'], 'patch_confidence'),
    node('Add', ['patch_confidence', 'image_zero'], 'confidence'),
]
inputs = [('image_previous', [1, 1, 80, 96]), ('image_current', [1, 1, 80, 96]),
          ('points_previous', [1, 1, 'N', 2]), ('flow_previous', [1, 1, 'N', 2]),
          ('patch_reference', ['N', 1, 31, 31])]
outputs = [('points_current', [1, 1, 'N', 2]), ('confidence', ['N', 1])]
graph = (b''.join(binary(1, item) for item in nodes) + binary(2, 'sparse_tracker_contract_fixture') +
         b''.join(binary(5, item) for item in [constant('half', [1], [.5]), constant('zero', [1], [0.]),
                                              constant('translation', [1, 1, 1, 2], [2., -1.])]) +
         b''.join(binary(11, value_info(*item)) for item in inputs) +
         b''.join(binary(12, value_info(*item)) for item in outputs))
model = integer(1, 8) + binary(2, 'open_vins_contract_test_only') + binary(7, graph) + binary(8, integer(2, 11))
Path(sys.argv[1]).write_bytes(model)
