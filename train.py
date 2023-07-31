import random
import time
import json
import pathlib
import torch
import traceback
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from tensorboardX import SummaryWriter
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import util
from qtw_generalization.preprocess_data import data_preprocess
from qtw_generalization.utils.argument_parser import args


def train(model, model_save_path, optimizer, epoch, phase, train_loaders, best_of_n):
    model.train()
    train_loss = 0
    train_loss_list = {}
    criterion = nn.MSELoss()

    if args.use_lrschd == 'True':
        if (epoch+1) % args.lr_decay_step_size == 0:
            print('load model')
            checkpoint = torch.load(model_save_path)
            model.load_state_dict(checkpoint['model_state_dict'])

    domain_name_flag = True
    if args.model_name in ['AdapTraj']:
        if args.invariant_specific_flag == 'True':

            if epoch == args.aggregator_epochs:
                print('load model')
                checkpoint = torch.load(model_save_path)
                model.load_state_dict(checkpoint['model_state_dict'])

                args.former_domain_weight = 1

                high_lr = args.high_lr_fraction * args.init_learning_rate
                low_lr = args.low_lr_fraction * high_lr

                if args.freeze_mode == 'allButAggregator':
                    params = [
                        {'params': model.encoder_past.parameters(), 'lr': low_lr},
                        {'params': model.encoder_dest.parameters(), 'lr': low_lr},
                        {'params': model.encoder_latent.parameters(), 'lr': low_lr},
                        {'params': model.decoder.parameters(), 'lr': low_lr},
                        {'params': model.non_local_theta.parameters(), 'lr': low_lr},
                        {'params': model.non_local_phi.parameters(), 'lr': low_lr},
                        {'params': model.non_local_g.parameters(), 'lr': low_lr},
                        {'params': model.predictor.parameters(), 'lr': low_lr}
                    ]
                    if args.invariant_flag == 'True':
                        params.extend([
                            {'params': model.invariant_ego.parameters(), 'lr': low_lr},
                            {'params': model.invariant_neighbors.parameters(), 'lr': low_lr},
                            {'params': model.ego_domain_predictor.parameters(), 'lr': low_lr},
                            {'params': model.neighbors_domain_predictor.parameters(), 'lr': low_lr}
                        ])
                    if args.specific_flag == 'True':
                        params.extend([
                            {'params': model.specific_ego_list.parameters(), 'lr': low_lr},
                            {'params': model.specific_neighbors_list.parameters(), 'lr': low_lr},
                            {'params': model.specific_neighbors_aggregator.parameters(), 'lr': high_lr},
                            {'params': model.specific_ego_aggregator.parameters(), 'lr': high_lr}
                        ])
                    if (args.invariant_flag == 'True') and (args.specific_flag == 'True'):
                        params.extend([
                            {'params': model.x_predictor.parameters(), 'lr': low_lr}
                        ])
                    if args.model_name == 'LBEBM-Ours':
                        params.extend([
                            {'params': model.EBM.parameters(), 'lr': low_lr}
                        ])
                elif args.freeze_mode == 'allButAggregatorPredictor':
                    params = [
                        {'params': model.encoder_past.parameters(), 'lr': low_lr},
                        {'params': model.encoder_dest.parameters(), 'lr': low_lr},
                        {'params': model.encoder_latent.parameters(), 'lr': low_lr},
                        {'params': model.decoder.parameters(), 'lr': low_lr},
                        {'params': model.non_local_theta.parameters(), 'lr': low_lr},
                        {'params': model.non_local_phi.parameters(), 'lr': low_lr},
                        {'params': model.non_local_g.parameters(), 'lr': low_lr},
                        {'params': model.predictor.parameters(), 'lr': high_lr}
                    ]
                    if args.invariant_flag == 'True':
                        params.extend([
                            {'params': model.invariant_ego.parameters(), 'lr': low_lr},
                            {'params': model.invariant_neighbors.parameters(), 'lr': low_lr},
                            {'params': model.ego_domain_predictor.parameters(), 'lr': low_lr},
                            {'params': model.neighbors_domain_predictor.parameters(), 'lr': low_lr}
                        ])
                    if args.specific_flag == 'True':
                        params.extend([
                            {'params': model.specific_ego_list.parameters(), 'lr': low_lr},
                            {'params': model.specific_neighbors_list.parameters(), 'lr': low_lr},
                            {'params': model.specific_neighbors_aggregator.parameters(), 'lr': high_lr},
                            {'params': model.specific_ego_aggregator.parameters(), 'lr': high_lr}
                        ])
                    if (args.invariant_flag == 'True') and (args.specific_flag == 'True'):
                        params.extend([
                            {'params': model.x_predictor.parameters(), 'lr': low_lr}
                        ])
                    if args.model_name == 'LBEBM-Ours':
                        params.extend([
                            {'params': model.EBM.parameters(), 'lr': low_lr}
                        ])
                elif args.freeze_mode == 'allButAggregatorPredictorDiffConstruct':
                    params = [
                        {'params': model.encoder_past.parameters(), 'lr': low_lr},
                        {'params': model.encoder_dest.parameters(), 'lr': low_lr},
                        {'params': model.encoder_latent.parameters(), 'lr': low_lr},
                        {'params': model.decoder.parameters(), 'lr': low_lr},
                        {'params': model.non_local_theta.parameters(), 'lr': low_lr},
                        {'params': model.non_local_phi.parameters(), 'lr': low_lr},
                        {'params': model.non_local_g.parameters(), 'lr': low_lr},
                        {'params': model.predictor.parameters(), 'lr': high_lr}
                    ]
                    if args.invariant_flag == 'True':
                        params.extend([
                            {'params': model.invariant_ego.parameters(), 'lr': low_lr},
                            {'params': model.invariant_neighbors.parameters(), 'lr': low_lr},
                            {'params': model.ego_domain_predictor.parameters(), 'lr': high_lr},
                            {'params': model.neighbors_domain_predictor.parameters(), 'lr': high_lr}
                        ])
                    if args.specific_flag == 'True':
                        params.extend([
                            {'params': model.specific_ego_list.parameters(), 'lr': low_lr},
                            {'params': model.specific_neighbors_list.parameters(), 'lr': low_lr},
                            {'params': model.specific_neighbors_aggregator.parameters(), 'lr': high_lr},
                            {'params': model.specific_ego_aggregator.parameters(), 'lr': high_lr}
                        ])
                    if (args.invariant_flag == 'True') and (args.specific_flag == 'True'):
                        params.extend([
                            {'params': model.x_predictor.parameters(), 'lr': high_lr}
                        ])
                    if args.model_name == 'LBEBM-Ours':
                        params.extend([
                            {'params': model.EBM.parameters(), 'lr': low_lr}
                        ])
                elif args.freeze_mode == 'allButOursPredictor':
                    params = [
                        {'params': model.encoder_past.parameters(), 'lr': low_lr},
                        {'params': model.encoder_dest.parameters(), 'lr': low_lr},
                        {'params': model.encoder_latent.parameters(), 'lr': low_lr},
                        {'params': model.decoder.parameters(), 'lr': low_lr},
                        {'params': model.non_local_theta.parameters(), 'lr': low_lr},
                        {'params': model.non_local_phi.parameters(), 'lr': low_lr},
                        {'params': model.non_local_g.parameters(), 'lr': low_lr},
                        {'params': model.predictor.parameters(), 'lr': high_lr}
                    ]
                    if args.invariant_flag == 'True':
                        params.extend([
                            {'params': model.invariant_ego.parameters(), 'lr': high_lr},
                            {'params': model.invariant_neighbors.parameters(), 'lr': high_lr},
                            {'params': model.ego_domain_predictor.parameters(), 'lr': high_lr},
                            {'params': model.neighbors_domain_predictor.parameters(), 'lr': high_lr}
                        ])
                    if args.specific_flag == 'True':
                        params.extend([
                            {'params': model.specific_ego_list.parameters(), 'lr': low_lr},
                            {'params': model.specific_neighbors_list.parameters(), 'lr': low_lr},
                            {'params': model.specific_neighbors_aggregator.parameters(), 'lr': high_lr},
                            {'params': model.specific_ego_aggregator.parameters(), 'lr': high_lr}
                        ])
                    if (args.invariant_flag == 'True') and (args.specific_flag == 'True'):
                        params.extend([
                            {'params': model.x_predictor.parameters(), 'lr': high_lr}
                        ])
                    if args.model_name == 'LBEBM-Ours':
                        params.extend([
                            {'params': model.EBM.parameters(), 'lr': low_lr}
                        ])
                elif args.freeze_mode == 'allButOursPredictor+NonLocal':
                    params = [
                        {'params': model.encoder_past.parameters(), 'lr': low_lr},
                        {'params': model.encoder_dest.parameters(), 'lr': low_lr},
                        {'params': model.encoder_latent.parameters(), 'lr': low_lr},
                        {'params': model.decoder.parameters(), 'lr': low_lr},
                        {'params': model.non_local_theta.parameters(), 'lr': high_lr},
                        {'params': model.non_local_phi.parameters(), 'lr': high_lr},
                        {'params': model.non_local_g.parameters(), 'lr': high_lr},
                        {'params': model.predictor.parameters(), 'lr': high_lr}
                    ]
                    if args.invariant_flag == 'True':
                        params.extend([
                            {'params': model.invariant_ego.parameters(), 'lr': high_lr},
                            {'params': model.invariant_neighbors.parameters(), 'lr': high_lr},
                            {'params': model.ego_domain_predictor.parameters(), 'lr': high_lr},
                            {'params': model.neighbors_domain_predictor.parameters(), 'lr': high_lr}
                        ])
                    if args.specific_flag == 'True':
                        params.extend([
                            {'params': model.specific_ego_list.parameters(), 'lr': low_lr},
                            {'params': model.specific_neighbors_list.parameters(), 'lr': low_lr},
                            {'params': model.specific_neighbors_aggregator.parameters(), 'lr': high_lr},
                            {'params': model.specific_ego_aggregator.parameters(), 'lr': high_lr}
                        ])
                    if (args.invariant_flag == 'True') and (args.specific_flag == 'True'):
                        params.extend([
                            {'params': model.x_predictor.parameters(), 'lr': high_lr}
                        ])
                    if args.model_name == 'LBEBM-Ours':
                        params.extend([
                            {'params': model.EBM.parameters(), 'lr': low_lr}
                        ])
                optimizer = optim.Adam(params, lr=args.init_learning_rate)
            elif epoch == args.aggregator_end_epochs:
                print('load model')
                checkpoint = torch.load(model_save_path)
                model.load_state_dict(checkpoint['model_state_dict'])

                args.former_domain_weight = 1
                optimizer = optim.Adam(model.parameters(), lr=args.high_lr_fraction * args.low_lr_fraction * args.init_learning_rate)


            if epoch >= args.aggregator_epochs:
                if (args.rand_domain_name == 'True') and (random.random() < args.aggregator_ratio):
                    domain_name_flag = False

    if args.batch_hetero == 'het':
        train_loaders_iter = {domain_name: iter(train_loader) for domain_name, train_loader in train_loaders}
        num_batches = min([len(train_loader) for train_loader in train_loaders.values()])
        pbar = tqdm(range(num_batches))
        pbar.set_description(f"Epoch {epoch} {phase} {args.batch_hetero}")
        for _ in pbar:
            for domain_name, train_loader_iter in train_loaders_iter:
                try:
                    batch = next(train_loader_iter)
                except StopIteration:
                    raise RuntimeError()

                if domain_name_flag is False:
                    domain_name = None

                model, train_loss_iter, loss_list = iter_train(batch, model, model_save_path, optimizer, criterion, best_of_n, epoch, domain_name=domain_name)
                train_loss += train_loss_iter
                if loss_list is not None:
                    for key, value in loss_list.items():
                        if key not in train_loss_list:
                            train_loss_list[key] = 0
                        train_loss_list[key] += value
        pbar.close()
    else:
        pbar = tqdm(train_loaders.keys())
        pbar.set_description(f"Epoch {epoch} {phase} {args.batch_hetero}")
        for domain_name in pbar:
            for batch in train_loaders[domain_name]:
                if domain_name_flag is False:
                    domain_name = None

                model, train_loss_iter, loss_list = iter_train(batch, model, model_save_path, optimizer, criterion, best_of_n, epoch, domain_name=domain_name)
                train_loss += train_loss_iter
                if loss_list is not None:
                    for key, value in loss_list.items():
                        if key not in train_loss_list:
                            train_loss_list[key] = 0
                        train_loss_list[key] += value
        pbar.close()

    return model, train_loss, train_loss_list, optimizer


def iter_train(batch, model, model_save_path, optimizer, criterion, best_of_n, epoch, domain_name=None):
    (
        traj, obs_traj, pred_traj, obs_traj_rel, pred_traj_rel,
        non_linear_ped, mask, seq_start_end
    ) = batch
    # traj format: batch, obs_len + pred_len, input_size
    total_loss_list = None

    if args.model_name in ['LBEBM', 'LBEBM-counter']:
        x, y, mask, plan, _ = util.model_input_data(traj, mask,
                                                    args.init_data_pos_method, args.data_scale, args.obs_len,
                                                    args.sub_goal_indexes)
        future = y.reshape(y.size(0), -1)
        dest_recon, mu, var, interpolated_future, cd, en_pos, en_neg, pcd = model.forward(x, dest=plan, mask=mask)
        dest_loss, future_loss, kld, subgoal_reg = model.calculate_loss(plan, dest_recon, mu, var, criterion, future,
                                                                        interpolated_future, args.sub_goal_indexes)
        total_loss = args.dest_loss_coeff * dest_loss + args.future_loss_coeff * future_loss + args.kld_coeff * kld + cd + subgoal_reg
        total_loss_list = {'ade': future_loss.item(), 'rcl': dest_loss.item(), 'kld': kld.item(), 'ebm_cd': cd.item(), 'plan': subgoal_reg.item()}

    elif args.model_name in ['PECNet', 'PECNet-counter']:
        x, y, mask, dest, initial_pos = util.model_input_data(traj, mask, args.init_data_pos_method, args.data_scale, args.obs_len)
        future = y[:, :-1, :].contiguous().view(y.size(0), -1).to(y)
        dest_recon, mu, var, interpolated_future = model.forward(x, initial_pos, mask=mask, dest=dest)
        rcl, kld, adl = model.calculate_loss(dest, dest_recon, mu, var, criterion, future, interpolated_future)
        total_loss_list = {'rcl': rcl.item(), 'ade': adl.item(), 'kld': kld.item()}
        total_loss = rcl + kld + adl

    elif args.model_name == 'AdapTraj':

        x, y, mask, dest, initial_pos = util.model_input_data(traj, mask,
                                                              args.init_data_pos_method, args.data_scale, args.obs_len,
                                                              args.sub_goal_indexes)
        # future = y[:, :-1, :].contiguous().view(y.size(0), -1).to(y)
        future = y.reshape(y.size(0), -1)
        dest_recon, mu, var, interpolated_future, ebm_cd, domain_feature_list = model.forward(x, initial_pos, mask=mask, dest=dest, domain_name=domain_name)
        # if domain_name is None:
        #     mu, var, domain_feature_list = None, None, None
        rcl, kld, adl, domain_loss_list = model.calculate_loss(dest, dest_recon, mu, var, criterion, future, interpolated_future, domain_feature_list)
        total_loss_list = {'rcl': rcl.item(), 'ade': adl.item()}
        total_loss = rcl + adl
        if ebm_cd is not None:
            ebm_cd = args.ebm_loss_coeff * ebm_cd
            total_loss += ebm_cd
            total_loss_list['ebm_cd'] = ebm_cd.item()
        if kld is not None:
            total_loss += kld
            total_loss_list['kld'] = kld.item()
        if domain_loss_list is not None:
            alpha_weight = 0.01
            beta_weight = 0.075
            gamma_weight = 0.25
            # domain_weight = 1
            original_weight = 1
            total_loss = total_loss * original_weight
            ego_diff, neighbors_diff, recon_mse, recon_simse, ego_similar, neighbors_similar = domain_loss_list
            total_loss_list['ego_diff'] = ego_diff.item()
            total_loss_list['neighbors_diff'] = neighbors_diff.item()
            total_loss_list['recon_mse'] = recon_mse.item()
            total_loss_list['recon_simse'] = recon_simse.item()
            if args.simse_flag == 'simse':
                total_loss += (beta_weight * (ego_diff + neighbors_diff) + alpha_weight * recon_simse) * args.former_domain_weight
            elif args.simse_flag == 'mse':
                total_loss += (beta_weight * (ego_diff + neighbors_diff) + alpha_weight * recon_mse) * args.former_domain_weight
            elif args.simse_flag == 'simse+mse':
                total_loss += (beta_weight * (ego_diff + neighbors_diff) + alpha_weight * (recon_mse + recon_simse)) * args.former_domain_weight
            if domain_name is not None:
                total_loss += (ego_similar + neighbors_similar) * gamma_weight * args.former_domain_weight
                total_loss_list['ego_similar'] = ego_similar.item()
                total_loss_list['neighbors_similar'] = neighbors_similar.item()

    else:
        raise ValueError(f'wrong model_name: {args.model_name}')

    if args.model_name not in ['CausalMotion', 'PECNet-CausalMotion']:

        optimizer.zero_grad()
        total_loss.backward()
        if args.clipping_threshold_flag == 'True':
            nn.utils.clip_grad_norm_(model.parameters(), args.clipping_threshold)
        optimizer.step()

    return model, total_loss.item(), total_loss_list


def eval(model, epoch, phase, data_loaders, best_of_n):
    model.eval()
    test_loss, test_loss_items = list(), {'ade': list(), 'fde': list(),
                                          'best_n_ade': list(), 'best_n_fde': list(),
                                          'pred_col': list(), 'gt_col': list(),
                                          'best_n_pred_col': list(), 'best_n_gt_col': list(),
                                          'nll': list()}
    total_best_n_ade, total_best_n_fde = [], []
    total_ade, total_fde = [], []
    total_pred_col, total_gt_col = [], []

    # with torch.no_grad():
    pbar = tqdm(data_loaders.values(), ncols=80)
    pbar.set_description(f"Epoch {epoch} {phase}")
    for data_loader in pbar:
        for batch in data_loader:
            (
                traj, obs_traj, pred_traj, obs_traj_rel, pred_traj_rel,
                non_linear_ped, mask, seq_start_end
            ) = batch
            # traj format: batch, obs_len + pred_len, input_size

            if args.model_name in ['LBEBM', 'LBEBM-counter']:
                x, y, mask, _, _ = util.model_input_data(traj, mask,
                                                            args.init_data_pos_method, args.data_scale, args.obs_len,
                                                            args.sub_goal_indexes)
                y = y.cpu().numpy()
                plan = y[:, args.sub_goal_indexes, :].reshape(y.shape[0], -1)

                all_plan_errs = []
                all_plans = []

                for _ in range(best_of_n):
                    plan_recon = model.forward(x, mask=mask)
                    plan_recon = plan_recon.detach().cpu().numpy()
                    all_plans.append(plan_recon)
                    plan_err = np.linalg.norm(plan_recon - plan, axis=-1)
                    all_plan_errs.append(plan_err)

                all_plan_errs = np.array(all_plan_errs)
                all_plans = np.array(all_plans)
                indices = np.argmin(all_plan_errs, axis=0)
                best_plan = all_plans[indices, np.arange(x.shape[0]), :]
                first_plan = all_plans[0]

                # ADE
                best_plan = torch.DoubleTensor(best_plan).cuda()
                interpolated_future = model.predict(x, best_plan, mask=mask)
                interpolated_future = interpolated_future.detach().cpu().numpy()
                predicted_future = np.reshape(interpolated_future, (-1, args.pred_len, 2))
                overall_err = np.linalg.norm(y - predicted_future, axis=-1).mean(axis=-1).sum()

                first_plan = torch.DoubleTensor(first_plan).cuda()
                first_interpolated_future = model.predict(x, first_plan, mask=mask)
                first_interpolated_future = first_interpolated_future.detach().cpu().numpy()
                first_predicted_future = np.reshape(first_interpolated_future, (-1, args.pred_len, 2))
                first_overall_err = np.linalg.norm(y - first_predicted_future, axis=-1).mean(axis=-1).sum()

                # FDE
                best_dest_err = np.linalg.norm(predicted_future[:, -1, :] - plan[:, -2:], axis=1).sum()
                first_dest_err = np.linalg.norm(first_predicted_future[:, -1, :] - plan[:, -2:], axis=1).sum()

                # collision
                # for neighbor in neighbors:
                #     neighbor_gt = []
                #     neighbor_predicted = []
                # pred_col = util.compute_col(first_predicted_future, neighbor_predicted).astype(float)
                # gt_col = util.compute_col(first_predicted_future, neighbor_gt).astype(float)
                # col_truth = util.compute_col(y, neighbor_gt)
                # pred_col[col_truth] = float('nan')
                # gt_col[col_truth] = float('nan')

                avg_parameter_divided = args.data_scale * traj.shape[0]

                total_best_n_ade.append(overall_err/avg_parameter_divided)
                total_best_n_fde.append(best_dest_err/avg_parameter_divided)
                total_ade.append(first_overall_err/avg_parameter_divided)
                total_fde.append(first_dest_err/avg_parameter_divided)

            elif args.model_name in ['PECNet', 'PECNet-counter']:
                with torch.no_grad():

                    x, y, mask, _, initial_pos = util.model_input_data(traj, mask, args.init_data_pos_method,
                                                                       args.data_scale, args.obs_len)
                    y = y.cpu().numpy()
                    dest = y[:, -1, :]

                    all_plan_errs = []
                    all_plans = []

                    for _ in range(best_of_n):
                        dest_recon = model.forward(x, initial_pos)
                        dest_recon = dest_recon.cpu().numpy()
                        all_plans.append(dest_recon)

                        l2error_sample = np.linalg.norm(dest_recon - dest, axis=1)
                        all_plan_errs.append(l2error_sample)

                    all_plan_errs = np.array(all_plan_errs)
                    all_plans = np.array(all_plans)
                    # choosing the best guess
                    indices = np.argmin(all_plan_errs, axis=0)
                    best_guess_dest = all_plans[indices, np.arange(x.shape[0]), :]
                    first_guess_dest = all_plans[0]

                    # FDE error
                    best_dest_err = np.mean(np.linalg.norm(best_guess_dest - dest, axis=1))
                    first_dest_err = np.mean(np.linalg.norm(first_guess_dest - dest, axis=1))

                    # ADE error
                    # final overall prediction
                    interpolated_future = model.predict(x, torch.DoubleTensor(best_guess_dest).to(x), mask, initial_pos)
                    interpolated_future = interpolated_future.cpu().numpy()
                    predicted_future = np.concatenate((interpolated_future, best_guess_dest), axis=1)
                    predicted_future = np.reshape(predicted_future, (-1, args.pred_len, 2))
                    overall_err = np.mean(np.linalg.norm(y - predicted_future, axis=2))

                    first_interpolated_future = model.predict(x, torch.DoubleTensor(first_guess_dest).to(x), mask, initial_pos)
                    first_interpolated_future = first_interpolated_future.cpu().numpy()
                    first_predicted_future = np.concatenate((first_interpolated_future, first_guess_dest), axis=1)
                    first_predicted_future = np.reshape(first_predicted_future, (-1, args.pred_len, 2))
                    first_overall_err = np.mean(np.linalg.norm(y - first_predicted_future, axis=2))

                    avg_parameter_divided = args.data_scale

                    total_best_n_ade.append(overall_err / avg_parameter_divided)
                    total_best_n_fde.append(best_dest_err / avg_parameter_divided)
                    total_ade.append(first_overall_err / avg_parameter_divided)
                    total_fde.append(first_dest_err / avg_parameter_divided)

            elif args.model_name == 'AdapTraj':
                if args.ebm_flag == 'False':
                    with torch.no_grad():
                        x, y, mask, _, initial_pos = util.model_input_data(traj, mask, args.init_data_pos_method,
                                                                           args.data_scale, args.obs_len,
                                                                           args.sub_goal_indexes)
                        y = y.cpu().numpy()
                        dest = y[:, args.sub_goal_indexes, :].reshape(y.shape[0], -1)

                        all_plan_errs = []
                        all_plans = []

                        for _ in range(best_of_n):
                            dest_recon = model.forward(x, initial_pos, mask=mask)
                            dest_recon = dest_recon.cpu().numpy()
                            all_plans.append(dest_recon)

                            l2error_sample = np.linalg.norm(dest_recon - dest, axis=1)
                            all_plan_errs.append(l2error_sample)

                        all_plan_errs = np.array(all_plan_errs)
                        all_plans = np.array(all_plans)
                        # choosing the best guess
                        indices = np.argmin(all_plan_errs, axis=0)
                        best_guess_dest = all_plans[indices, np.arange(x.shape[0]), :]
                        first_guess_dest = all_plans[0]

                        # ADE error
                        # final overall prediction
                        interpolated_future = model.predict(x, torch.DoubleTensor(best_guess_dest).to(x), mask, initial_pos)
                        predicted_future = interpolated_future.cpu().numpy()
                        predicted_future = np.reshape(predicted_future, (-1, args.pred_len, 2))
                        overall_err = np.mean(np.linalg.norm(y - predicted_future, axis=2))

                        first_interpolated_future = model.predict(x, torch.DoubleTensor(first_guess_dest).to(x), mask, initial_pos)
                        first_predicted_future = first_interpolated_future.cpu().numpy()
                        first_predicted_future = np.reshape(first_predicted_future, (-1, args.pred_len, 2))
                        first_overall_err = np.mean(np.linalg.norm(y - first_predicted_future, axis=2))

                        # FDE error
                        best_dest_err = np.mean(np.linalg.norm(predicted_future[:, -1, :] - dest[:, -2:], axis=1))
                        first_dest_err = np.mean(np.linalg.norm(first_predicted_future[:, -1, :] - dest[:, -2:], axis=1))

                        avg_parameter_divided = args.data_scale

                        total_best_n_ade.append(overall_err / avg_parameter_divided)
                        total_best_n_fde.append(best_dest_err / avg_parameter_divided)
                        total_ade.append(first_overall_err / avg_parameter_divided)
                        total_fde.append(first_dest_err / avg_parameter_divided)
                else:
                    x, y, mask, _, initial_pos = util.model_input_data(traj, mask, args.init_data_pos_method,
                                                                       args.data_scale, args.obs_len,
                                                                       args.sub_goal_indexes)
                    y = y.cpu().numpy()
                    dest = y[:, args.sub_goal_indexes, :].reshape(y.shape[0], -1)

                    all_plan_errs = []
                    all_plans = []

                    for _ in range(best_of_n):
                        dest_recon = model.forward(x, initial_pos, mask=mask)
                        dest_recon = dest_recon.detach().cpu().numpy()
                        all_plans.append(dest_recon)

                        l2error_sample = np.linalg.norm(dest_recon - dest, axis=1)
                        all_plan_errs.append(l2error_sample)

                    all_plan_errs = np.array(all_plan_errs)
                    all_plans = np.array(all_plans)
                    # choosing the best guess
                    indices = np.argmin(all_plan_errs, axis=0)
                    best_guess_dest = all_plans[indices, np.arange(x.shape[0]), :]
                    first_guess_dest = all_plans[0]

                    # ADE error
                    # final overall prediction
                    interpolated_future = model.predict(x, torch.DoubleTensor(best_guess_dest).to(x), mask, initial_pos)
                    predicted_future = interpolated_future.detach().cpu().numpy()
                    predicted_future = np.reshape(predicted_future, (-1, args.pred_len, 2))
                    overall_err = np.mean(np.linalg.norm(y - predicted_future, axis=2))

                    first_interpolated_future = model.predict(x, torch.DoubleTensor(first_guess_dest).to(x), mask,
                                                              initial_pos)
                    first_predicted_future = first_interpolated_future.detach().cpu().numpy()
                    first_predicted_future = np.reshape(first_predicted_future, (-1, args.pred_len, 2))
                    first_overall_err = np.mean(np.linalg.norm(y - first_predicted_future, axis=2))

                    # FDE error
                    best_dest_err = np.mean(np.linalg.norm(predicted_future[:, -1, :] - dest[:, -2:], axis=1))
                    first_dest_err = np.mean(np.linalg.norm(first_predicted_future[:, -1, :] - dest[:, -2:], axis=1))

                    avg_parameter_divided = args.data_scale

                    total_best_n_ade.append(overall_err / avg_parameter_divided)
                    total_best_n_fde.append(best_dest_err / avg_parameter_divided)
                    total_ade.append(first_overall_err / avg_parameter_divided)
                    total_fde.append(first_dest_err / avg_parameter_divided)

            else:
                raise ValueError(f'Wrong model name: {args.model_name}')

    pbar.close()

    test_loss_items['ade'] = sum(total_ade)/len(total_ade)
    test_loss_items['fde'] = sum(total_fde)/len(total_fde)
    test_loss_items['best_n_ade'] = sum(total_best_n_ade)/len(total_best_n_ade)
    test_loss_items['best_n_fde'] = sum(total_best_n_fde)/len(total_best_n_fde)

    assert not (test_loss_items['ade'] != test_loss_items['ade']), "ade@1 is NaN"
    assert not (test_loss_items['fde'] != test_loss_items['fde']), "fde@1 is NaN"
    assert not (test_loss_items['best_n_ade'] != test_loss_items['best_n_ade']), "ade@20 is NaN"
    assert not (test_loss_items['best_n_fde'] != test_loss_items['best_n_fde']), "fde@20 is NaN"

    return test_loss_items


def main():
    model_dir = os.path.join('experiments',
                             f'train_{args.train_dataset_name}',
                             f'test_{args.test_dataset_name}',
                             args.model_name,
                             f'invariant_specific_flag_{args.invariant_specific_flag}',
                             # f'ebm_flag_{args.ebm_flag}',
                             f'invariant_{args.invariant_flag}_specific_{args.specific_flag}_similar_{args.similar_dimension_used_flag}',
                             f'former_domain_weight_{args.former_domain_weight}',
                             f'freezeMode_{args.freeze_mode}',
                             f'low_lr_frac_{args.low_lr_fraction}',
                             f'aggregator_epochs_{args.aggregator_epochs}',
                             f'aggregator_ratio_{args.aggregator_ratio}',
                             f'predict_mode_{args.predict_mode}',
                             f'high_lr_frac_{args.high_lr_fraction}',
                             time.strftime('%d_%b_%Y_%H_%M_%S', time.localtime()))
    pathlib.Path(model_dir).mkdir(parents=True, exist_ok=True)

    # Save config to model directory
    with open(os.path.join(model_dir, 'config.json'), 'w') as conf_json:
        json.dump(args.__dict__, conf_json)

    # basic setting, e.g., random seed, torch.device
    util.basic_setting(args.device, args.gpu_deterministic, args.seed)

    # tensorboard to write
    log_writer = SummaryWriter(log_dir=model_dir)

    # load data
    args.train_dataset_name = args.train_dataset_name.split(',')
    args.test_dataset_name = args.test_dataset_name.split(',')
    train_loaders = data_preprocess.load_data(
        args.train_fraction, args.val_fraction, args.train_dataset_name, 'train',
        args.obs_len, args.pred_len, args.batch_size, args.process_flag,
        args.train_data_rotate, args.train_data_reverse)
    val_loaders = data_preprocess.load_data(
        args.train_fraction, args.val_fraction, args.train_dataset_name, 'val',
        args.obs_len, args.pred_len, args.batch_size, args.process_flag)
    test_loaders = data_preprocess.load_data(
        args.train_fraction, args.val_fraction, args.test_dataset_name, 'test',
        args.obs_len, args.pred_len, args.batch_size, args.process_flag)

    # load model, optimizer
    if args.model_name in ['LBEBM', 'LBEBM-motion']:
        from models.LBEBM import LBEBM

        model = LBEBM(args)
        model.double().to(args.device)
        optimizer = optim.Adam(model.parameters(), lr=args.init_learning_rate)
    elif args.model_name == 'LBEBM-counter':
        from models.LBEBM_counter import LBEBM_counter

        model = LBEBM_counter(args)
        model.double().to(args.device)
        optimizer = optim.Adam(model.parameters(), lr=args.init_learning_rate)
    elif args.model_name == 'PECNet':
        from models.PECNet import PECNet

        model = PECNet(args)
        model.double().to(args.device)
        optimizer = optim.Adam(model.parameters(), lr=args.init_learning_rate)
    elif args.model_name == 'PECNet-counter':
        from models.PECNet_counter import PECNet_counter

        model = PECNet_counter(args)
        model.double().to(args.device)
        optimizer = optim.Adam(model.parameters(), lr=args.init_learning_rate)
    elif args.model_name == 'AdapTraj':
        from models.AdapTraj import AdapTraj

        model = AdapTraj(args, train_loaders.keys())
        model.double().to(args.device)
        optimizer = optim.Adam(model.parameters(), lr=args.init_learning_rate)
    else:
        raise ValueError(f'Wrong model name: {args.model_name}')

    if args.model_init_weight == 'True':
        model.apply(util.init_weights)

    # set scheduler
    if args.use_lrschd == 'True':
        lr_schedule = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_decay_step_size, gamma=args.learning_decay_rate)

    # train
    best_top_n_eval_ade = float("inf")
    best_epoch = -1
    best_test_ade, best_test_fde = float("inf"), float("inf")
    best_top_n_test_ade, best_top_n_test_fde = float("inf"), float("inf")
    best_of_n = args.best_dest * args.best_traj

    model_save_path = None

    for epoch in range(args.train_epochs):

        print(f"============================= Epoch {epoch} ================================")
        model, train_loss, train_loss_list, optimizer = train(model, model_save_path, optimizer, epoch, 'Train', train_loaders, best_of_n)
        # eval_loss_items = eval(model, epoch, 'Eval', val_loaders, best_of_n)
        eval_loss_items = None
        if (epoch % 5 == 0) or (epoch == args.train_epochs - 1):
            test_loss_items = eval(model, epoch, 'Test', test_loaders, best_of_n)
        else:
            test_loss_items = None

        if args.model_name not in ['CausalMotion', 'PECNet-CausalMotion']:
            optim_lr = optimizer.param_groups[0]['lr']
        else:
            optim_lr = args.lrstgat

        util.log_writer_add_scalar(log_writer, epoch, train_loss, train_loss_list, optim_lr, eval_loss_items, test_loss_items)

        if test_loss_items:
            util.print_results(epoch, test_loss_items['ade'], test_loss_items['fde'],
                               test_loss_items['best_n_ade'], test_loss_items['best_n_fde'],
                               best_of_n, flag='current')

            # if eval_loss_items['best_n_ade'] < best_top_n_eval_ade:
            #     best_top_n_eval_ade = eval_loss_items['best_n_ade']
            #     best_test_ade, best_test_fde = test_loss_items['ade'], test_loss_items['fde']
            #     best_top_n_test_ade, best_top_n_test_fde = test_loss_items['best_n_ade'], test_loss_items['best_n_fde']
            #     util.print_results(epoch, best_test_ade, best_test_fde,
            #                        best_top_n_test_ade, best_top_n_test_fde,
            #                        best_of_n)
            #
            #     util.save_model(model_dir, args, model, optimizer)

            if best_test_ade > test_loss_items['ade']:
                best_test_ade = test_loss_items['ade']
                best_test_fde = test_loss_items['fde']

            if best_top_n_test_ade > test_loss_items['best_n_ade']:
                best_top_n_test_ade = test_loss_items['best_n_ade']
                best_top_n_test_fde = test_loss_items['best_n_fde']
                best_epoch = epoch

                model_save_path = util.save_model(model_dir, args, model, optimizer)

            util.print_results(best_epoch, best_test_ade, best_test_fde,
                               best_top_n_test_ade, best_top_n_test_fde,
                               best_of_n, flag='best')

        if args.use_lrschd == 'True':
            lr_schedule.step()

    util.print_results('Final', best_test_ade, best_test_fde, best_top_n_test_ade, best_top_n_test_fde, best_of_n)

    util.write_into_results_csv(args, best_test_ade, best_test_fde, best_top_n_test_ade, best_top_n_test_fde, model_dir)


if __name__ == '__main__':
    main()
    print('Finished.')

