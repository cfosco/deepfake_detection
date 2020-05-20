import json
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np


def load_all_results(results_dir):
    all_results = []
    for filename in os.listdir(results_dir):
        if not filename.endswith('.json'):
            continue

        path = os.path.join(results_dir, filename)
        with open(path) as f:
            data = json.load(f)
        data.pop('outputs')
        all_results.append(data)
    return all_results


def aggregate_results(all_results, agg_by='dataset'):
    agg_results = defaultdict(dict)
    for res in all_results:
        if agg_by == 'dataset':
            key = os.path.join(res['dataset'], res['part'])
            alt_key = res['checkpoint_file']
        elif agg_by == 'checkpoint_file':
            key = res['checkpoint_file']
            alt_key = os.path.join(res['dataset'], res['part'])

        val = res['acc']
        agg_results[key][alt_key] = val
    return agg_results


def plot_bar(
    labels,
    means,
    title='Model Performance',
    xlabel='Dataset',
    ylabel='Accuracy',
    width=0.35,
    xticks_rotation=90,
):
    x = np.arange(len(labels))
    fig, ax = plt.subplots()
    num_bars = len(means)
    sub_width = width / num_bars
    h_num_bars = num_bars / 2
    for i, (name, mean) in enumerate(means.items()):
        print(i, name, mean)
        if i < h_num_bars:
            ax.bar((x - (1 * sub_width)), mean, width, label=name)
        else:
            ax.bar((x + (i * sub_width)), mean, width, label=name)

        # ax.bar((x - (h_num_bars * sub_width)) + (i * sub_width), mean, width, label=name)
        # ax.bar((x - (h_num_bars * sub_width)) + (i * sub_width), mean, width, label=name)

    ax.set_xlabel(xlabel)
    plt.xticks(rotation=xticks_rotation)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    plt.show()


def print_agg(agg_results):
    for name, data in agg_results.items():
        print(name)
        for k, v in data.items():
            print(f'\t {k}: {v}')
        print()
