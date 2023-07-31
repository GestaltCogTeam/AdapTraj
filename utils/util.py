import os
import csv
import torch
import random
import numpy as np


def basic_setting(device, gpu_deterministic, seed):
    torch.cuda.set_device(torch.device(device))

    if not gpu_deterministic:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
    else:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def log_writer_add_scalar(log_writer, epoch, train_loss, train_loss_list, lr, eval_loss_items, test_loss_items):
    log_writer.add_scalar("/train/loss_epoch", train_loss, epoch)
    log_writer.add_scalar("/train/lr_epoch", lr, epoch)
    for key, value in train_loss_list.items():
        log_writer.add_scalar(f"/train/{key}_epoch", value, epoch)
    if eval_loss_items:
        log_writer.add_scalar("/eval/ade_epoch", eval_loss_items['ade'], epoch)
        log_writer.add_scalar("/eval/fde_epoch", eval_loss_items['fde'], epoch)
        log_writer.add_scalar("/eval/best_n_ade_epoch", eval_loss_items['best_n_ade'], epoch)
        log_writer.add_scalar("/eval/best_n_fde_epoch", eval_loss_items['best_n_fde'], epoch)
    if test_loss_items:
        log_writer.add_scalar("/test/ade_epoch", test_loss_items['ade'], epoch)
        log_writer.add_scalar("/test/fde_epoch", test_loss_items['fde'], epoch)
        log_writer.add_scalar("/test/best_n_ade_epoch", test_loss_items['best_n_ade'], epoch)
        log_writer.add_scalar("/test/best_n_fde_epoch", test_loss_items['best_n_fde'], epoch)


def print_results(epoch, best_test_ade, best_test_fde, best_top_n_test_ade, best_top_n_test_fde, n, flag='best'):
    print(f"============================= Epoch {epoch} ================================")
    print(f"{flag} top@1 test ade/fde : {best_test_ade:.5f}/{best_test_fde:.5f}")
    print(f"{flag} top@{n} test ade/fde : {best_top_n_test_ade:.5f}/{best_top_n_test_fde:.5f}")
    print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")


def save_model(model_dir, args, model, optimizer):
    save_path = os.path.join(model_dir, 'model_registrar.pt')
    torch.save({'args': args,
                'model_state_dict': model.state_dict(),
                # 'optimizer_state_dict': optimizer.state_dict()
                }, save_path)
    print("Saved model to: {}".format(save_path))
    return save_path


def write_into_results_csv(args, best_test_ade, best_test_fde, best_top_n_test_ade, best_top_n_test_fde, model_dir):
    results_file = os.path.join('results.csv')
    header_flag = True
    if os.path.exists(results_file):
        header_flag = False

    file = open(results_file, 'a+', newline='')
    csv_writer = csv.writer(file)
    if header_flag:
        csv_writer.writerow([
            'model_name', 'train_dataset_name', 'test_dataset_name', 'obs_len', 'pred_len', 'train_epochs',
            'batch_size', 'dropout', 'best_dest', 'best_traj',
            'train_data_rotate', 'train_data_reverse',
            'train_fraction', 'val_fraction',
            'init_lr',
            'invariant_specific_flag',
            'invariant_flag', 'specific_flag', 'similar_dimension_used_flag',
            'former_domain_weight', 'data_scale',
            'rand_domain_name', 'simse_flag',
            'ftraj_dimension_flag', 'fuse_dimension_flag',
            'ebm_flag', 'ftraj_residual', 'fuse_residual',
            'lr_decay_step_size', 'aggregator_epochs', 'aggregator_end_epochs', 'aggregator_ratio',
            'initial_pos_flag', 'plan_flag', 'train_mode', 'aggregator_gan_step', 'seed',
            'low_lr_fraction', 'high_lr_fraction',
            'top@1 ade', 'top@1 fde', 'top@n ade', 'top@n fde',
            'model_dir'])
    csv_writer.writerow([
        args.model_name, args.train_dataset_name, args.test_dataset_name, args.obs_len, args.pred_len, args.train_epochs,
        args.batch_size, args.dropout, args.best_dest, args.best_traj,
        args.train_data_rotate, args.train_data_reverse,
        args.train_fraction, args.val_fraction,
        args.init_learning_rate,
        args.invariant_specific_flag,
        args.invariant_flag, args.specific_flag, args.similar_dimension_used_flag,
        args.former_domain_weight, args.data_scale,
        args.rand_domain_name, args.simse_flag,
        args.ftraj_dimension_flag, args.fuse_dimension_flag,
        args.ebm_flag, args.ftraj_residual, args.fuse_residual,
        args.lr_decay_step_size, args.aggregator_epochs, args.aggregator_end_epochs, args.aggregator_ratio,
        args.initial_pos_flag, args.plan_flag, args.train_mode, args.aggregator_gan_step, args.seed,
        args.low_lr_fraction, args.high_lr_fraction,
        best_test_ade, best_test_fde, best_top_n_test_ade, best_top_n_test_fde,
        model_dir])
    file.close()
    print('save experiment results into csv file.')


def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        torch.nn.init.kaiming_normal_(m.weight)


def interpolate_traj(traj, num_interp=4):
    """
    Add linearly interpolated points of a trajectory
    """
    sz = traj.shape
    dense = np.zeros((sz[0], (sz[1] - 1) * (num_interp + 1) + 1, 2))
    dense[:, :1, :] = traj[:, :1]

    for i in range(num_interp+1):
        ratio = (i + 1) / (num_interp + 1)
        dense[:, i+1::num_interp+1, :] = traj[:, 0:-1] * (1 - ratio) + traj[:, 1:] * ratio

    return dense


def compute_col(predicted_traj, predicted_trajs_all, thres=0.2, num_interp=4):
    """
    Input:
        predicted_trajs: predicted trajectory of the primary agents
        predicted_trajs_all: predicted trajectory of all agents in the scene
    """
    ph = predicted_traj.shape[0]
    dense_all = interpolate_traj(predicted_trajs_all, num_interp)
    dense_ego = interpolate_traj(predicted_traj[None, :], num_interp)
    distances = np.linalg.norm(dense_all - dense_ego, axis=-1)
    mask = distances[:, 0] > 0
    return distances[mask].min(axis=0) < thres


def model_input_data(traj, mask, init_data_pos_method, data_scale, obs_len, sub_goal_indexes=-1):
    # traj format: batch, obs_len + pred_len, input_size
    x = traj[:, :obs_len, :]
    y = traj[:, obs_len:, :]
    initial_pos = (x[:, -1, :]*data_scale).double().cuda()
    if init_data_pos_method == 'last':
        x = x - traj[:, obs_len - 1:obs_len, :]
        y = y - traj[:, obs_len - 1:obs_len, :]
    elif init_data_pos_method == 'first':
        x = x - traj[:, 0:1, :]
        y = y - traj[:, 0:1, :]
    else:
        raise ValueError(f'wrong init_data_pos_method: {init_data_pos_method}')

    x *= data_scale
    y *= data_scale
    x = x.double().cuda()
    y = y.double().cuda()

    mask = mask.double().cuda()

    x = x.reshape(-1, x.shape[1] * x.shape[2])
    plan = y[:, sub_goal_indexes, :].detach().clone().view(y.size(0), -1)

    return x, y, mask, plan, initial_pos
