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
            print(param, nx)
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


def convert_to_html_table(results, param_name, main_title=None, comments=None):
    string = "<table>\n"
    keys = get_keys(results)
    subkeys = get_keys(results[keys[0]])
    names, devices, bitsizes = zip(*subkeys)

    devices_names = sorted(list(set(zip(devices, names))))
    length = len(devices_names) + 1
    n_bitsizes = len(set(bitsizes))
    cpus_cols = list(devices).count("CPU") / n_bitsizes
    gpus_cols = list(devices).count("GPU") / n_bitsizes
    assert cpus_cols + gpus_cols == len(devices_names)

    if main_title is not None:
        string += f'<tr><th align="center" colspan="{length}">{str(main_title)}</th></tr>\n'

    for i, bitsize in enumerate(sorted(list(set(bitsizes)))):

        if i != 0:
            string += f'<tr><td colspan="{length}">&nbsp;</td></tr>\n'

        # make bitsize header
        text = f"{bitsize} bits"
        if comments is not None:
            text += " - "
            if isinstance(comments, (tuple, list)) and len(comments) == n_bitsizes:
                text += str(comments[i])
            else:
                text += str(comments)
        string += f'<tr><th align="center">Bitsize</th>'
        string += f'<th align="center" colspan="{length - 1}">{text}</th></tr>\n'

        # make device header
        string += f'<tr><th align="center">Devices</th>'
        string += f'<th align="center" colspan="{cpus_cols}"">CPU</th>'
        string += f'<th align="center" colspan="{gpus_cols}">GPU</tr>\n'

        # make param_name / backend header
        string += f'<tr><th align="center">{param_name}</th>'
        for device, name in devices_names:
            string += f'<th align="center">{name}</th>'
        string += "</tr>\n"

        # make results rows
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
