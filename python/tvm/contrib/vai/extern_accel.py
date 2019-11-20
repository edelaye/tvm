# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

""" Implementation of external Vitis-AI accel operation """

import warnings
import numpy as np

import tvm

try:
    from dnndk import n2cube
except Exception:
    warnings.warn("Could not import dnndk n2cube")


@tvm.register_func("tvm.accel.accel_fused")
def accel_fused(kernel_name, input_name, output_name,
                layout, out, *ins):
    """
    Registration of external accel.accel_fused operation

    Arguments
    ---------
    kernel_name: str
        the name of the DPU kernel

    input_name: str
        the input_name of the DPU Kernel

    output_name: str
        the output_name of the DPU kernel

    layout: str
        the layout of the accel operation

    out: tvm.ndarray.NDArray
        the output array for the operation

    ins: List[tvm.ndarray.NDArray]
        the operation inputs
    """

    # Attach to DPU driver and prepare for running
    n2cube.dpuOpen()

    # Create DPU Kernels
    kernel = n2cube.dpuLoadKernel(kernel_name)

    # Create DPU Tasks for kernel
    task = n2cube.dpuCreateTask(kernel, 0)

    # Load image to DPU
    X = ins[0].asnumpy()

    # Possibly transpose input if layout is NCHW
    if layout == 'NCHW':
        # NCHW --> NHWC
        X = np.transpose(X, (0, 2, 3, 1)) 

    X = X.reshape((-1))
    n2cube.dpuSetInputTensorInHWCFP32(task, input_name, X, len(X))

    # Model run on DPU
    n2cube.dpuRunTask(task)

    # Get the output tensor size
    size = n2cube.dpuGetOutputTensorSize(task, output_name)
    address = n2cube.dpuGetOutputTensorAddress(task, output_name)

    value = [0 for i in range(size)]

    # Get the output tensor data
    n2cube.dpuGetTensorData(address, value, size)
    scale = n2cube.dpuGetOutputTensorScale(task, output_name, idx=0)

    value = np.array(value).astype(np.float32) * float(scale)

    value_shape = tuple(out.shape) if layout == 'NHWC' else  \
        (out.shape[0], out.shape[2], out.shape[3], out.shape[1])
    value = np.reshape(value, value_shape)

    # DPU output is in NHWC but graph is executed in NCHW
    if output_layout == 'NCHW':
        value = np.transpose(value, (0, 3, 1, 2))

    tvm.nd.array(value).copyto(out)
