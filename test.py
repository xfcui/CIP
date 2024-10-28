import argparse
import os
import sys
sys.path.append('../')
from CIP.models.sbap import *
from CIP import commons, runner
from datetime import datetime
from time import time
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--work_dir', type=str, default='./prove/87')
    args = parser.parse_args()
    config_path = os.path.join(args.work_dir, 'affinity_default.yaml')
    config = commons.get_config_easydict(config_path)
    config.train.device = commons.get_device(config.train.gpus, config.train.gpu_memory_need)
    commons.set_seed(config.seed)

    load_start = time()
    test_data,generalize_csar_data = commons.get_test_dataset(config)
    print('load data time:{:.2f}s'.format(time() - load_start))

    config.model.lig_node_dim, config.model.lig_edge_dim = test_data.lig_node_dim, test_data.lig_edge_dim
    config.model.pro_node_dim, config.model.pro_edge_dim = test_data.pro_node_dim, test_data.pro_edge_dim
    config.model.inter_edge_dim = test_data.inter_edge_dim

    model = globals()[config.model.model_type + '_MTL'](config).to(config.train.device)
    RMSEs, MAEs, Pearsons, Spearmans, SDs = [], [], [], [], []
   
    for i in range(config.train.finetune_times):
        checkpoint = os.path.join(args.work_dir, f'checkpointbest_valid_{i}')
        state = torch.load(checkpoint, map_location=config.train.device)
        model.load_state_dict(state["model"])
        RMSE, MAE, SD, Pearson = runner.reproduce_runner.reproduce_result(config, test_data, model, config.train.device)
        RMSEs.append(RMSE)
        MAEs.append(MAE)
        Pearsons.append(Pearson)
        SDs.append(SD)

    print(f'RMSE :{np.mean(RMSEs)}')
    print(f'MAE :{np.mean(MAEs)}')
    print(f'Pearson :{np.mean(Pearsons)}')
    print(f'SD :{np.mean(SDs)}')
