# /usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from ot.backend import get_backend_list, jax, tf


def setup_backends():
    if jax:
        from jax.config import config
        config.update("jax_enable_x64", True)

    if tf:
        from tensorflow.python.ops.numpy_ops import np_config
        np_config.enable_numpy_behavior()


def exec_bench(setup, tested_function, param_list, n_runs):
    backend_list = get_backend_list()
    results = dict()
    for param in param_list:
        L = dict()
        inputs = setup(param)
        for nx in backend_list:
            results_nx = nx._bench(
                tested_function,
                *inputs,
                n_runs=n_runs
            )
            L.update(results_nx)
        results[param] = L
    return results


def get_keys(d):
    return sorted(list(d.keys()))


def convert_to_html_table(results, param_name):
    string = "<table>\n"
    keys = get_keys(results)
    print(results[keys[0]].keys())
    subkeys = get_keys(results[keys[0]])
    names, devices, bitsizes = zip(*subkeys)

    names = sorted(list(set(zip(names, devices))))
    length = len(names) + 1

    for bitsize in sorted(list(set(bitsizes))):
        string += f'<tr><th align="center" colspan="{length}">{bitsize} bits</td></tr>\n'
        string += f'<tr><th align="center">{param_name}</td>'
        for name, device in names:
            string += f'<th align="center">{name} {device}</td>'
        string += "</tr>\n"

        for key in keys:
            subdict = results[key]
            subkeys = get_keys(subdict)
            string += f'<tr><td align="center">{key}</td>'
            for subkey in subkeys:
                name, device, size = subkey
                if size == bitsize:
                    string += f'<td align="center">{subdict[subkey]:.4f}</td>'
            string += "</tr>\n"

    string += "</table>"
    return string
