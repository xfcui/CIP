import argparse
import os
import sys
from CIP.models.sbap import *
from CIP import commons, runner
from datetime import datetime
from time import time
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='./config/affinity/affinity_default.yaml')
    args = parser.parse_args()

    # get config
    config = commons.get_config_easydict(args.config_path)

    # get device
    config.train.device = commons.get_device(config.train.gpus, config.train.gpu_memory_need)

    # set random seed
    commons.set_seed(config.seed)

    # set run_dir
    now = str(datetime.now()).replace(" ", "_").split(".")[0]
    run_dir = commons.get_run_dir(config)
    run_dir_now = os.path.join(run_dir, now)
    config.train.save_path = run_dir_now

    if config.train.use_pretrain_model:
        now = config.test.now
        run_dir = commons.get_run_dir(config)
        run_dir = os.path.join(run_dir, now)
        config.train.pretrain_model_save_path = run_dir

    # get logger
    config.logger = commons.get_logger(run_dir_now)

    # load data
    load_start = time()
    train_data, val_data, test_data, generalize_csar_data = commons.get_finetune_dataset(config)
    print('load data time:{:.2f}s'.format(time() - load_start))

    # set feats dim
    config.model.lig_node_dim, config.model.lig_edge_dim = test_data.lig_node_dim, test_data.lig_edge_dim
    config.model.pro_node_dim, config.model.pro_edge_dim = test_data.pro_node_dim, test_data.pro_edge_dim
    config.model.inter_edge_dim = test_data.inter_edge_dim

    RMSEs, MAEs, Pearsons, Spearmans, SDs = [], [], [], [], []
    test_RMSEs, test_MAEs, test_Pearsons, test_Spearmans, test_SDs = [], [], [], [], []

    for i in range(config.train.finetune_times):
        # get model
        model = globals()[config.model.model_type + '_MTL'](config).to(config.train.device)

        # get optimizer
        optimizer = commons.get_optimizer(config.train.optimizer, model)

        # get scheduler
        scheduler = commons.get_scheduler(config.train.scheduler, optimizer)

        # get runner
        solver = runner.finetune_runner.DefaultRunner(train_data, val_data, test_data, generalize_csar_data, model, optimizer, scheduler, config)

        # load pre-trained checkpoint
        if config.train.use_pretrain_model:
            solver.load(config.train.pretrain_model_save_path, epoch=config.test.epoch, load_optimizer=False, load_scheduler=False)

        # get logger
        config.logger = commons.get_logger(run_dir_now)

        # save config file to run dir
        cmd = f'cp {args.config_path} {run_dir_now}'
        os.system(cmd)

        # test before fintune
        RMSE, MAE, SD, Pearson = solver.evaluate('test', verbose=1)

        # train
        RMSE, MAE, SD, Pearson = solver.train(repeat_index=i)

        RMSEs.append(RMSE)
        MAEs.append(MAE)
        Pearsons.append(Pearson)
        SDs.append(SD)

        




