# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from maskrcnn_benchmark.utils.amp_compat import amp
from torchvision.ops import nms as tv_nms

try:
    from maskrcnn_benchmark import _C
except ImportError:  # pragma: no cover - runtime fallback path
    _C = None

# Only valid with fp32 inputs - give AMP the hint
if _C is not None:
    nms = amp.float_function(_C.nms)
else:
    nms = amp.float_function(tv_nms)

# nms.__doc__ = """
# This function performs Non-maximum suppresion"""
