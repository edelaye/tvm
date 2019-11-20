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

""" Vitis-AI NNVM partitioning for Xilinx FPGA acceleration """

import os
import json
import nnvm
import nnvm.symbol as sym

import xfgraph
# Register dpu as a xfgraph target device
from xfgraph.contrib import dnndk

from xfgraph.frontend import from_nnvm
from xfgraph.graph.io.xgraph_io import XGraphIO
from xfgraph.generator.tensorflow import XfGraphTfGeneratorOptimizer
from xfgraph.contrib.dnndk.decent_quantizer import DECENTQuantizer
from xfgraph.contrib.dnndk.dnnc_compiler import DNNCCompiler


class NNVMPartitioningPass(object):

    """
    The Vitis-AI NNVM partitioning pass for converting a NNVM graph to a
    Vitis-AI accelerated NNVM graph

    Arguments
    ---------
    target: str
        the target accelerator, only 'dpu' is supported at the moment
    params: dict from str to array
        the relay model parameters
    data_shapes: dict[str:List[int]]
        dictionary containing the data shapes of the inputs
    inputs_func: function
        a python function which takes an iterator number and a layout and
        provides a numpy array of inputs to be used for quantization
        calibration
    layout: str
        the layout of the Relay model, only 'NCHW' and 'NHWC' supported
        at the moment

    """

    def __init__(self, target, params, data_shapes, inputs_func, layout):

        if target not in ['dpu-ultra96', 'dpu-ultra96-skipcompile', \
                'dpu-zcu104', 'dpu-zcu104-skipcompile']: # TODO: remove dpu_nocompile
            raise ValueError("Invalid target: {} for the Vitis-AI"\
                             " partitioning pass, only 'dpu-ultra96' and"\
                             " 'dpu-zcu104' targets are supported at the"\
                             " moment.".format(target))

        if layout not in ['NCHW', 'NHWC']:
            raise ValueError("Invalid layout: {} for Vitis-AI partitioning"\
                             " pass, only 'NCHW' and 'NHWC' are supported at"\
                             " the moment.".format(target))

        self.target = target
        self.params = params
        self.data_shapes = data_shapes
        self.inputs_func = inputs_func
        self.layout = layout

        self.work_dir = '/tmp/vai'
        os.makedirs(self.work_dir, exist_ok=True)

    def __call__(self, graph):
        """
        Partition the provided NNVM graph

        Arguments
        ---------
        graph: NNVM graph
            the NNVM graph to be partitioned

        Returns
        -------
        graph: NNVM graph
            the partitioned NNVM graph
        """

        target = self.target.split("-")[0]
        device = self.target.split("-")[1]
        docompile = len(self.target.split("-")) <= 2

        # TODO params can be found in ctx??
        xfgraph = from_nnvm(graph, self.params,
                            shapes=self.data_shapes,
                            data_layout=self.layout)

        if not docompile:
            pass
        elif target == 'dpu':

            # Optimize xfgraph for Tensorflow generation
            xfgraph.optimize(XfGraphTfGeneratorOptimizer)

            # Internal partitioning
            xfgraph.partition(devices=[target])

            dpu_xgraph = xfgraph.schedule(device=target)
            XGraphIO.save(dpu_xgraph,
                          os.path.join(self.work_dir, 'dpu_xgraph'))

            # Quantization
            quantizer = DECENTQuantizer(xfgraph, self.inputs_func,
                                        self.work_dir)
            netcfgs = quantizer.quantize(subgraphs_only=True)

            # Compilation
            if device.lower() == 'zcu104':
                dcf = "/dnndk/dcf/ZCU104.dcf"
            elif device.lower() == 'ultra96':
                dcf = "/dnndk/dcf/Ultra96.dcf"
            else:
                raise ValueError("Unkwowm device: {}".format(device))

            compiler = DNNCCompiler(xfgraph, netcfgs=netcfgs, dcf=dcf,
                                    work_dir=self.work_dir)
            compiler.compile()

        else:
            raise ValueError("Unsupported target: {}".format(target))

        # NNVM partitioning
        graph = self.graph_reconst(
            nnvm_graph=graph.index.nodes,
            path=self.work_dir,
            layout=self.layout,
            target=target,
            output_layers=[]
        )

        return graph

    def graph_reconst(self, path, nnvm_graph, layout, output_layers=None,
                      target='dpu'):
        """
        Function to pass through NNVM graph, trim nodes and create
        new NNVM graph

        Arguments
        ---------
        path: str
            the path to the compilation and quantization files
        nnvm_graph: List[Node]
            list of NNVM nodes
        layout: str
            the layout of the graph
        output_layers: List[str] (optional)
            list of output layers to be added at the end of the graph
        target: str
            the accelerator target

        Returns
        -------
            a new NNVM graph for acceleration
        """

        node_map = {}
        accel_inputs = []

        if target == 'dpu':
            compiler_json_file = path + "/dpu_xgraph.json"
            dnn_name = path + "/dnnc_comp_xp0.json"
            with open(compiler_json_file) as json_file:
                json_graph = json.load(json_file)
            with open(dnn_name) as json_file:
                dnnc_comp_d = json.load(json_file)

            for node in json_graph['nodes']:
                if node['LayerParameter']['type'][0] == 'DPU':
                    attrs = node['LayerParameter']['attrs']

                    kernel_name = node['name']
                    input_names = attrs['input_names']
                    output_names = attrs['output_names']
                    graph_inputs = attrs['input_layers'][input_names[0]]
                    graph_outputs = [attrs['output_layers']\
                        [output_names[0]][-1]]
                    compiler_shape_output = node['LayerParameter']['shapes']

            input_names = dnnc_comp_d[input_names[0]]
            output_names = dnnc_comp_d[output_names[0]]


        else:
            compiler_json_file = path + "/_compiler.json"
            with open(compiler_json_file) as json_file:
                json_graph = json.load(json_file)

            graph_inputs = json_graph["inputs"]
            graph_outputs = json_graph["outputs"]                                 
            compiler_shape_output = json_graph["network"][-1]["outputshapes"]

            kernel_name = ""
            input_names = ""
            output_names = ""

        xfuse_inputs = []
        fuse_list = []
        queue = []

        if target == 'dpu':
            input_list = graph_inputs
        else:
            input_list = [n['input_name'] for n in graph_inputs]

        for layer in graph_outputs:

            if target == 'dpu':
                layer_name = layer
            else:
                layer_name = layer['previous_layers'][0]

            # Parse the graph and find the matching output node
            layer_nid = 0
            for nid, node in enumerate(nnvm_graph):
                node_name = node["name"]
                if layer_name == node_name:
                    layer_nid = nid
            assert layer_nid != 0, "The output node was not found"
            queue.append(layer_nid)

            self.fuse(nnvm_graph, xfuse_inputs, input_list, queue,
                      fuse_list, 0, target)

        # Reconstruct graph
        for nid, node in enumerate(nnvm_graph):
            inputs = node["inputs"]
            attrs = node.get("attrs", {})
            node_name = node["name"]
            op_name = node["op"]
            get_clone = lambda c, o_n, n_n, a: getattr(nnvm.symbol, o_n)(
                *c, name=n_n, **a)
            new_entry = None

            if nid in fuse_list:
                for layer in graph_outputs:
                    if target == 'dpu':
                        layer_name = layer
                    else:
                        layer_name = layer['previous_layers'][0]
                    if node_name == layer_name:
                        # CREATE ACCEL NODE
                        if target == 'xdnn' and layout == 'NHWC':
                            output_shape = (1,
                                            compiler_shape_output[2],
                                            compiler_shape_output[3],
                                            compiler_shape_output[1])

                        elif target == 'dpu' and layout == 'NCHW':
                            output_shape = (1,
                                            compiler_shape_output[3],
                                            compiler_shape_output[1],
                                            compiler_shape_output[2])
                        else:
                            output_shape = (1,
                                            compiler_shape_output[1],
                                            compiler_shape_output[2],
                                            compiler_shape_output[3])

                        new_entry = sym.accel(*accel_inputs,
                                              kernel_name=kernel_name,
                                              input_name=input_names,
                                              output_name=output_names,
                                              output_shape=output_shape,
                                              layout=layout)

                        node_map[nid] = new_entry
            else:

                if op_name == "null":
                    new_entry = nnvm.symbol.Variable(node_name)
                    accel_inputs.append(new_entry)
                else:
                    children = [node_map[e[0]] for e in node["inputs"]]
                    new_entry = get_clone(children, op_name, node_name, attrs)

                node_map[nid] = new_entry

        # Possibly add output_layers at the end (softmax)
        if output_layers:
            for layer in output_layers:
                if layer == 'Softmax':
                    nodes = list(node_map.keys())
                    node_map[nodes[-1]+1] = sym.softmax(node_map[nodes[-1]])

        node_map_list = list(node_map.items())

        # Assume the last node is always the output
        return nnvm.graph.create(node_map_list[-1][1])

    def fuse(self, graph, xfuse_inputs, input_list, queue, fuse_list,
             count, target):
        """
        Recursive function for parsing graph and finding nodes between
        compiler input and output nodes
        """

        # Return condition
        if not queue:
            return

        nid = queue.pop()

        fuse_list.append(nid)
        count = count + 1
        children = [e[0] for e in graph[nid]["inputs"]]

        # If the nid is not visited then you could queue
        for nid in children:
            node_name = graph[nid]["name"]

            if node_name in input_list:
                if target == 'dpu':
                    fuse_list.append(nid)
                    xfuse_inputs.append(nid-1)
                else:
                    xfuse_inputs.append(nid)
            elif nid in fuse_list:
                continue
            else:
                queue.append(nid)

        self.fuse(graph, xfuse_inputs, input_list, queue, fuse_list,
                  count, target)
