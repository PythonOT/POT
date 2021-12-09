# /usr/bin/env python3
# -*- coding: utf-8 -*-

from ot.backend import get_backend_list, jax, tf
import gc


def setup_backends():
    if jax:
        from jax.config import config
        config.update("jax_enable_x64", True)

    if tf:
        from tensorflow.python.ops.numpy_ops import np_config
        np_config.enable_numpy_behavior()


def exec_bench(setup, tested_function, param_list, n_runs, warmup_runs):
    backend_list = get_backend_list()
    for i, nx in enumerate(backend_list):
        if nx.__name__ == "tf" and i < len(backend_list) - 1:
            # Tensorflow should be the last one to be benchmarked because
            # as far as I'm aware, there is no way to force it to release
            # GPU memory. Hence, if any other backend is benchmarked after
            # Tensorflow and requires the usage of a GPU, it will not have the
            # full memory available and you may have a GPU Out Of Memory error
            # even though your GPU can technically hold your tensors in memory.
            backend_list.pop(i)
            backend_list.append(nx)
            break

    inputs = [setup(param) for param in param_list]
    results = dict()
    for nx in backend_list:
        for i in range(len(param_list)):
            print(nx, param_list[i])
            args = inputs[i]
            results_nx = nx._bench(
                tested_function,
                *args,
                n_runs=n_runs,
                warmup_runs=warmup_runs
            )
            gc.collect()
            results_nx_with_param_in_key = dict()
            for key in results_nx:
                new_key = (param_list[i], *key)
                results_nx_with_param_in_key[new_key] = results_nx[key]
            results.update(results_nx_with_param_in_key)
    return results


def convert_to_html_table(results, param_name, main_title=None, comments=None):
    string = "<table>\n"
    keys = list(results.keys())
    params, names, devices, bitsizes = zip(*keys)

    devices_names = sorted(list(set(zip(devices, names))))
    params = sorted(list(set(params)))
    bitsizes = sorted(list(set(bitsizes)))
    length = len(devices_names) + 1
    cpus_cols = list(devices).count("CPU") / len(bitsizes) / len(params)
    gpus_cols = list(devices).count("GPU") / len(bitsizes) / len(params)
    assert cpus_cols + gpus_cols == len(devices_names)

    if main_title is not None:
        string += f'<tr><th align="center" colspan="{length}">{str(main_title)}</th></tr>\n'

    for i, bitsize in enumerate(bitsizes):

        if i != 0:
            string += f'<tr><td colspan="{length}">&nbsp;</td></tr>\n'

        # make bitsize header
        text = f"{bitsize} bits"
        if comments is not None:
            text += " - "
            if isinstance(comments, (tuple, list)) and len(comments) == len(bitsizes):
                text += str(comments[i])
            else:
                text += str(comments)
        string += f'<tr><th align="center">Bitsize</th>'
        string += f'<th align="center" colspan="{length - 1}">{text}</th></tr>\n'

        # make device header
        string += f'<tr><th align="center">Device</th>'
        string += f'<th align="center" colspan="{cpus_cols}">CPU</th>'
        string += f'<th align="center" colspan="{gpus_cols}">GPU</th></tr>\n'

        # make param_name / backend header
        string += f'<tr><th align="center">{param_name}</th>'
        for device, name in devices_names:
            string += f'<th align="center">{name}</th>'
        string += "</tr>\n"

        # make results rows
        for param in params:
            string += f'<tr><td align="center">{param}</td>'
            for device, name in devices_names:
                key = (param, name, device, bitsize)
                string += f'<td align="center">{results[key]:.4f}</td>'
            string += "</tr>\n"

    string += "</table>"
    return string
