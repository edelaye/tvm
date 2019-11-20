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
Vitis-AI Relay pass to partition Relay graph for Xilinx FPGA acceleration

"""

import os
import json
import tvm
from tvm import relay

import xfgraph
# Register dpu as a xfgraph target device
from xfgraph.contrib import dnndk

from xfgraph.frontend import from_relay
from xfgraph.graph.io.xgraph_io import XGraphIO
from xfgraph.generator.tensorflow import XfGraphTfGeneratorOptimizer
from xfgraph.contrib.dnndk.decent_quantizer import DECENTQuantizer
from xfgraph.contrib.dnndk.dnnc_compiler import DNNCCompiler

@relay.transform.module_pass(opt_level=4)
class PartitioningPass:

    """
    The Vitis-AI partitioning pass

    Arguments
    ---------
    target: str
        the target accelerator, only 'dpu' is supported at the moment

    params: dict from str to array
        the relay model parameters

    inputs_func: function
        a python function which takes an iterator number and a layout and
        provides a numpy array of inputs to be used for quantization
        calibration

    layout: str
        the layout of the Relay model, only 'NCHW' and 'NHWC' supported
        at the moment

    """

    def __init__(self, target, params, inputs_func, layout):

        if target not in ['dpu-ultra96', 'dpu-zcu104']:
            raise ValueError("Invalid target: {} for the Vitis-AI"\
                " partitioning pass, only 'dpu-ultra96' and 'dpu-zcu104'"\
                " targets are supported at the moment.".format(target))

        if layout not in ['NCHW', 'NHWC']:
            raise ValueError("Invalid layout: {} for Vitis-AI partitioning"\
                             " pass, only 'NCHW' and 'NHWC' are supported at:"\
                             " the moment.".format(target))

        self.target = target
        self.params = params
        self.inputs_func = inputs_func
        self.layout = layout

        self.work_dir = '/tmp/vai'
        os.makedirs(self.work_dir, exist_ok=True)

    def transform_module(self, mod, ctx):
        """
        Transformation module method which is called from parent __call__
        """

        target = self.target.split("-")[0]
        device = self.target.split("-")[1]


        if target == 'dpu':
   
            xfgraph = from_relay(mod, self.params, data_layout=self.layout)

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

        # Relay partitioning
        mod = self.reconst_graph(
            mod=mod,
            path=self.work_dir,
            layout=self.layout,
            target=target,
            output_layers=[]
        )

        return mod


    def reconst_graph(self, mod, path, layout, target, output_layers=None):
        """
        Create a partitioned Relay module for acceleration

        Arguments
        ---------
        mod: Relay module
            the Relay module to be partitioned
        path: str
            the path to the compilation and quantization files
        layout: str
            the layout of the graph (NCHW or NHWC)
        target: str
            the acceleration target
        output_layers: List[str]
            the list of output layers to be added at the end of the expression

        Returns
        -------
        mod: A partitioned Relay module
        """
        node_map = {}
        xdnn_inputs = []
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
                    graph_outputs = [attrs['output_layers'][output_names[0]][-1]]
                    compiler_shape_output = node['LayerParameter']['shapes']
          
            input_names = dnnc_comp_d[input_names[0]]
            output_names = dnnc_comp_d[output_names[0]]

        elif target == 'xdnn':
            compiler_json_file = path  + "/_compiler.json"
            with open(compiler_json_file) as json_file:
                json_graph = json.load(json_file)

            graph_inputs = json_graph["inputs"]
            graph_outputs = json_graph["outputs"]

            compiler_shape_output = json_graph["network"][-1]["outputshapes"]

            kernel_name = ""
            input_names = ""
            output_names = ""
        else:
            raise ValueError("Unsupported target: {}".format(target))

        xfuse_inputs = []
        fuse_list = []
        queue = []

        if target == 'dpu':
            input_list = [self.extract_hash(n, 'dpu') for n in graph_inputs]
        else:
            input_list = [self.extract_hash(n, 'input_name') for n in graph_inputs]

        expr = mod.functions[mod.get_global_var('main')]
        expr = expr.body

        for output in graph_outputs:
            if target == 'dpu':
                output_hash = self.extract_hash(output, 'dpu')
            else:
                output_hash = self.extract_hash(output, 'previous_layers')

            expr = self.traverse(expr, path, output_hash, input_list, layout,
                                 compiler_shape_output, kernel_name,
                                 output_names, input_names, target)


        # Possibly add output_layers at the end (softmax)
        if output_layers:
            for layer in output_layers:
                if layer == 'Softmax':
                    expr = relay.nn.softmax(expr)
                else:
                    raise ValueError("Unsupported output layer: {} provided"
                                     .format(layer))

        mod = relay.Module.from_expr(expr)

        return mod

    def extract_hash(self, name, key):
        """
        Extract Relay expression hash from xfgraph layer names
        """
        if key == 'input_name':
            val = name[key].split('-')
        elif key == 'previous_layers':
            val = name[key][0].split('-')
        else:
            val = name.split('-')

        try:
            return int(val[0])
        except ValueError:
            if len(val) == 1:
                return val[0]
            else:
                return int(val[1])

    def recurse(self, expr, input_list):
        """
        Recursively find expression input nodes in provided input list 
        """
        if isinstance(expr, tvm.relay.expr.Function):
            return self.recurse(expr.body, name)

        elif isinstance(expr, tvm.relay.expr.Call):
            if hash(expr) in input_list:
                return expr.args[0]

            for node in expr.args:
                ret = self.recurse(node, input_list)
                if ret is not None:
                    return ret
            return None

        elif isinstance(expr, tvm.relay.expr.TupleGetItem):
            if hash(expr) in input_list:
                return expr
            return self.recurse(expr.tuple_value, input_list)

        elif isinstance(expr, tvm.relay.expr.Var):
            input_name = str(expr.name_hint)

            if (hash(expr) in input_list or input_name in input_list):
                return expr
            else:
                return None
        elif isinstance(expr, tvm.relay.expr.Tuple):
            if hash(expr) in input_list:
                return expr
            for node in expr.fields:
                ret = self.recurse(node, input_list)
                if ret is not None:
                    return ret
                else:
                    return None  
    
        else:
            raise ValueError("Missing condition to handle node type {}".format(type(expr)))


    def traverse(self, expr, path, output_hash, input_list, layout,
                 output_shape, kernel_name, output_names, input_names, target):
        """
        Traverse through Relay expression to find input and output expressions
        and recreate expression with fused acceleration operation

        expr: Relay expression
            the expression to be traversed
        output_hash: int
            the hash of the ouput expression
        input_list: List[int/str]
            the list of input hashes or names
        layout: str
            the expression layout
        output_shape: List[int]
            the shape of the output
        path: str
            the path to the quantization and compilation files

        Returns
        -------
        new_expr: tvm.relay.Expr the new top level Relay expression node
        """
        if hash(expr) == output_hash:
            for node in expr.args:
                input_node = self.recurse(node, input_list)
                if input_node is not None:

                    if target == 'xdnn' and layout == 'NHWC':
                        output_shape = (1,
                                        output_shape[2],
                                        output_shape[3],
                                        output_shape[1])
           
                    elif target == 'dpu'  and layout == 'NCHW':
                        output_shape = (1,
                                        output_shape[3],
                                        output_shape[1],
                                        output_shape[2])
                    else:
                        output_shape = (1,
                                        output_shape[1],
                                        output_shape[2],
                                        output_shape[3])

                    op = relay.nn.accel([input_node],
                                        output_shape=output_shape,
                                        layout=layout,
                                        input_name=input_names,
                                        output_name=output_names,
                                        kernel_name=kernel_name)

                    return op

            return None
    

        else:
            if isinstance(expr, tvm.relay.expr.Constant):
                return None

            elif isinstance(expr, tvm.relay.expr.Call):
                for node in expr.args:

                    output_node = self.traverse(node, path, output_hash,
                                                input_list, layout, 
                                                output_shape, kernel_name, 
                                                output_names, input_names, 
                                                target)
                    if output_node is not None:
                        break
                # Case where the output node is not the chosen branch of the expression
                if output_node is None:
                    return output_node

            elif isinstance(expr, tvm.relay.expr.TupleGetItem):

                return self.traverse(expr.tuple_value, path, output_hash,
                                     input_list, layout,output_shape,
                                     kernel_name, output_names,
                                     input_names, target)
        
            elif isinstance(expr, tvm.relay.expr.Tuple):
                return None

            elif isinstance(expr, tvm.relay.expr.Var):
                return None
            else:
                return None

            # Reconstruct the graph by recreating the nodes outside the subgraph
            #   found by partitioning
            if isinstance(expr, tvm.relay.expr.Call):

                children = []
                for node in expr.args:

                    if isinstance(node, tvm.relay.expr.Call):
                        if node.op == output_node.op or output_node.op.name == 'nn.accel':
                            children.append(output_node)
                        else:
                            children.append(node)
                    else:
                        children.append(node)

                new_node = relay.Call(expr.op, children, expr.attrs, expr.type_args)

            else:
                raise NotImplementedError("Condition to reconstruct node type {}"\
                    " has not been implemented".format(type(expr)))

        return new_node
