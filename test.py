#!/usr/bin/env python3
# -*- coding:utf-8 -*-
'''
File: /workspace/code/test.py
Project: /workspace/code
Created Date: Friday June 27th 2025
Author: Kaixu Chen
-----
Comment:

Have a good code time :)
-----
Last Modified: Friday June 27th 2025 2:06:13 am
Modified By: the developer formerly known as Kaixu Chen at <chenkaixusan@gmail.com>
-----
Copyright (c) 2025 The University of Tsukuba
-----
HISTORY:
Date      	By	Comments
----------	---	---------------------------------------------------------
'''
import pynvml
pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(0)
info = pynvml.nvmlDeviceGetMemoryInfo(handle)
print("✅ GPU Mem Used:", info.used / 1024**2, "MB")