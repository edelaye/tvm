#!/bin/bash
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

set -e
set -u
set -o pipefail

# install libraries for python package on ubuntu
pip install --upgrade pip
pip3 install --no-cache-dir \
    onnx==1.5.0 \
    numpy \
    pydot==1.4.1 \
    h5py==2.8.0 \
    opencv-python \
    matplotlib \
    jupyter \
    psutil \
    sklearn \
    scipy \
    progressbar2 \
    dill
