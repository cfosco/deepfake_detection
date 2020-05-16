import os
import json


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
