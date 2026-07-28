#!/usr/bin/env python3
"""Check cuda_graph_setup.py to understand hook registration flow."""
with open('/sgl-workspace/sglang/python/sglang/srt/model_executor/model_runner_components/cuda_graph_setup.py') as f:
    lines = f.readlines()
    for i, line in enumerate(lines[100:150], 101):
        print(f'{i:4d}: {line.rstrip()}')
