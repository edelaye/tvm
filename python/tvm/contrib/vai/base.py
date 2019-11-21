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

"""
Registration of Vitis-AI NNVM and relay 'accel' operations

"""

from nnvm.top import registry as reg

import tvm
from tvm.relay import op as op


@reg.register_schedule("accel", level=15)
def schedule_accel(_, outputs, target):
    del target
    return tvm.create_schedule([x.op for x in outputs])



@reg.register_compute("accel", level=15)
def compute_accel(attrs, inputs, outputs):
    """Compute definition of accel operation for NNVM"""
    name = 'accel0'

    out = tvm.extern(outputs[0].shape, inputs,
                     lambda ins, outs: tvm.call_packed(
                         'tvm.accel.accel_fused', attrs['kernel_name'],
                         attrs['input_name'], attrs['output_name'],
                         attrs['layout'], outs[0], *ins),
                     name=name)

    return out


@op.register_schedule("nn.accel", level=15)
def schedule_nn_accel(_, outputs, target):
    del target
    return tvm.create_schedule([x.op for x in outputs])


@op.register_compute("nn.accel", level=15)
def compute_nn_accel(attrs, inputs, outputs, _):
    """Compute definition of accel operation for Relay"""
    name = 'accel0'

    out = tvm.extern(outputs.shape, inputs,
                     lambda ins, outs: tvm.call_packed(
                         'tvm.accel.accel_fused', attrs.kernel_name,
                         attrs.input_name, attrs.output_name, attrs.layout,
                         outs[0], *ins),
                     name=name)

    return [out]
