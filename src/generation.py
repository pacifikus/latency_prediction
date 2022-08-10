import pandas as pd
import numpy as np
import torch
import time
import uuid
from .random_cnn import RandomCNN


def run_experiments(model, dim, config):
    results = []
    for _ in range(config['generation']['n_experiments_by_net']):
        start_time = time.time()
        model(torch.rand(1, 3, dim, dim))
        results.append(time.time() - start_time)
    return np.array(results).mean()


def generate_models(config):
    result = []
    dim = config['generation']['dim']
    model_count = 0
    models_hash = set()
    while model_count < config['generation']['n_models']:
        try:
            cnn = RandomCNN(dim, dim, config)
            if cnn not in models_hash:
                mean_running_time = run_experiments(cnn.model, dim, config)
                model_uuid = str(uuid.uuid4())
                torch.save(cnn, config['data']['nn_models_path'] + model_uuid)
                result.append([model_uuid, mean_running_time])
                model_count += 1
                models_hash.add(str(cnn))
        except:
            print(f'An error occurred while generating the RandomCNN object')
    pd.DataFrame(
        result,
        columns=['model_uuid', 'time'],
    ).to_csv(config['data']['model_result_df_path'], index=False)
