import argparse
import os
from CIP.models.sbap import *
from CIP import commons, runner
from datetime import datetime
from time import time
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='./config/affinity/affinity_default.yaml',help='path of dataset')
    parser.add_argument("--local_rank", default=-1)
    args = parser.parse_args()


    config = commons.get_config_easydict(args.config_path)


    config.train.device = torch.device("cuda:" + str(args.local_rank))

    commons.ddp_setup(int(args.local_rank))

 
    commons.set_seed(config.seed)


    now = str(datetime.now()).replace(" ", "_").split(".")[0]
    run_dir = commons.get_run_dir(config)
    run_dir_now = os.path.join(run_dir,now)
    config.train.save_path = run_dir_now


    if dist.get_rank() == 0:
        config.logger = commons.get_logger(run_dir_now)
        # save config file to run dir
        cmd = f'cp {args.config_path} {run_dir_now}'
        os.system(cmd)
    else:
        config.logger = None


    load_start = time()
    train_data, val_data, test_data = commons.get_dataset(config, ddp=True)
    print('load data time:{:.2f}s'.format(time() - load_start))


    config.model.lig_node_dim, config.model.lig_edge_dim = train_data.lig_node_dim, train_data.lig_edge_dim
    config.model.pro_node_dim, config.model.pro_edge_dim = train_data.pro_node_dim, train_data.pro_edge_dim
    config.model.inter_edge_dim = train_data.inter_edge_dim


    model = globals()[config.model.model_type+'_MTL'](config).to(config.train.device)

    if config.train.resume_train and dist.get_rank() == 0:
        checkpoint = os.path.join(os.path.join(run_dir, config.train.resume_now), 'checkpoint%s' % config.train.resume_epoch)
        state = torch.load(checkpoint, map_location=config.train.device)
        model.load_state_dict(state["model"])


    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(config.train.device)

    model = DDP(model, device_ids=[int(args.local_rank)], find_unused_parameters=True)

    optimizer = commons.get_optimizer(config.train.optimizer, model)


    scheduler = commons.get_scheduler(config.train.scheduler, optimizer)

  
    solver = runner.asrp_runner.DefaultRunner(train_data, val_data, test_data, model, optimizer, scheduler, config)


    if config.train.resume_train and dist.get_rank() == 0:
        solver.best_matric = state['best_matric']
        solver.start_epoch = state['cur_epoch'] + 1


    dist.barrier()
    solver.train(ddp=True)
    dist.destroy_process_group()
