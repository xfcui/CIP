import argparse
import os
import sys
sys.path.append('../')
from CIP.models.sbap import *
from CIP import commons, runner
from datetime import datetime
from time import time
import numpy as np
from tqdm import tqdm
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--work_dir', type=str, default='./prove/87')
    args = parser.parse_args()
    config_path = os.path.join(args.work_dir, 'affinity_default.yaml')
    config = commons.get_config_easydict(config_path)
    config.train.device = commons.get_device(config.train.gpus, config.train.gpu_memory_need)
    commons.set_seed(config.seed)

    load_start = time()
    test_data = commons.get_test_dataset(config)
    print('load data time:{:.2f}s'.format(time() - load_start))

    config.model.lig_node_dim, config.model.lig_edge_dim = test_data.lig_node_dim, test_data.lig_edge_dim
    config.model.pro_node_dim, config.model.pro_edge_dim = test_data.pro_node_dim, test_data.pro_edge_dim
    config.model.inter_edge_dim = test_data.inter_edge_dim

    model = globals()[config.model.model_type + '_MTL'](config).to(config.train.device)
    RMSEs, MAEs, Pearsons, Spearmans, SDs = [], [], [], [], []
   
    progress_bar = tqdm(total=config.train.finetune_times, desc="Model Evaluation Progress")

    step_increment = config.train.finetune_times / config.train.finetune_times

    for i in range(config.train.finetune_times):
        checkpoint = os.path.join(args.work_dir, f'checkpointbest_valid_{i}.ckpt')
        state = torch.load(checkpoint, map_location=config.train.device)
        model.load_state_dict(state["model"])
        
        RMSE, MAE, SD, Pearson = runner.reproduce_runner.reproduce_result(config, test_data, model, config.train.device)
        RMSEs.append(RMSE)
        MAEs.append(MAE)
        Pearsons.append(Pearson)
        SDs.append(SD)

        progress_bar.update(step_increment)

    progress_bar.close()

    print(f'- Root Mean Squared Error (RMSE): {np.mean(RMSEs):.2f}')
    print(f'- Mean Absolute Error (MAE): {np.mean(MAEs):.2f}')
    print(f'- Pearson Correlation Coefficient: {np.mean(Pearsons):.2f}')
    print(f'- Standard Deviation (SD) of Errors: {np.mean(SDs):.2f}')
