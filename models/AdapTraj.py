import torch
import torch.nn as nn
from torch.autograd import Variable
import random
from torch.nn.utils import weight_norm
import pdb
from torch.nn import functional as F
from torch.distributions.normal import Normal
import math
import numpy as np
import yaml


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_size=(1024, 512), activation='relu', discrim=False, dropout=-1):
        super(MLP, self).__init__()
        dims = []
        dims.append(input_dim)
        dims.extend(hidden_size)
        dims.append(output_dim)
        self.layers = nn.ModuleList()
        for i in range(len(dims)-1):
            self.layers.append(nn.Linear(dims[i], dims[i+1]))

        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()

        self.sigmoid = nn.Sigmoid() if discrim else None
        self.dropout = dropout

    def forward(self, x):
        for i in range(len(self.layers)):
            x = self.layers[i](x)
            if i != len(self.layers)-1:
                x = self.activation(x)
                if self.dropout != -1:
                    x = nn.Dropout(min(0.1, self.dropout/3) if i == 1 else self.dropout)(x)
            elif self.sigmoid:
                x = self.sigmoid(x)
        return x


class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, input_memory):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = input_memory
        self.position = (self.position + 1) % self.capacity

    def sample(self, n=100):
        samples = random.sample(self.memory, n)
        return torch.cat(samples)

    def __len__(self):
        return len(self.memory)


def sample_p_0(e_init_sig, n, nz=16):
    return e_init_sig * torch.randn(*[n, nz]).double().cuda()


class AdapTraj(nn.Module):

    def __init__(self, hyperparams, domain_list):
        super(AdapTraj, self).__init__()

        self.hyperparams = hyperparams

        self.zdim = hyperparams.zdim
        self.nonlocal_pools = hyperparams.nonlocal_pools
        self.sigma = hyperparams.sigma

        self.encoder_past = MLP(input_dim=hyperparams.obs_len * 2, output_dim=hyperparams.fdim, hidden_size=hyperparams.enc_past_size)
        self.encoder_dest = MLP(input_dim=2 * len(hyperparams.sub_goal_indexes),
                                output_dim=hyperparams.fdim,
                                hidden_size=hyperparams.enc_dest_size)
        self.invariant_ego = MLP(input_dim=hyperparams.fdim,
                                 output_dim=hyperparams.fdim,
                                 hidden_size=hyperparams.non_local_theta_size)

        if self.hyperparams.initial_pos_flag == 'True':
            initial_pos_dimension = 2
        else:
            initial_pos_dimension = 0

        if self.hyperparams.ebm_flag == 'True':
            ebm_hidden_dimension = 200
            if (self.hyperparams.ftraj_dimension_flag == 'cat') and (self.hyperparams.invariant_specific_flag == 'True'):
                self.EBM = nn.Sequential(
                    nn.Linear(hyperparams.zdim + 2 * hyperparams.fdim, ebm_hidden_dimension),
                    nn.GELU(),
                    nn.Linear(ebm_hidden_dimension, ebm_hidden_dimension),
                    nn.GELU(),
                    nn.Linear(ebm_hidden_dimension, hyperparams.ny),
                )
            else:
                self.EBM = nn.Sequential(
                    nn.Linear(hyperparams.zdim + hyperparams.fdim, ebm_hidden_dimension),
                    nn.GELU(),
                    nn.Linear(ebm_hidden_dimension, ebm_hidden_dimension),
                    nn.GELU(),
                    nn.Linear(ebm_hidden_dimension, hyperparams.ny),
                )
            self.replay_memory = ReplayMemory(hyperparams.memory_size)

        if self.hyperparams.invariant_specific_flag == 'True':
            if self.hyperparams.ftraj_dimension_flag == 'cat':
                # takes in the past
                self.encoder_latent = MLP(input_dim=3 * hyperparams.fdim, output_dim=2 * hyperparams.zdim,
                                          hidden_size=hyperparams.enc_latent_size)
                self.decoder = MLP(input_dim=2 * hyperparams.fdim + hyperparams.zdim, output_dim=2 * len(hyperparams.sub_goal_indexes),
                                   hidden_size=hyperparams.dec_size)
                self.non_local_theta = MLP(input_dim=2 * hyperparams.fdim, output_dim=hyperparams.non_local_dim,
                                           hidden_size=hyperparams.non_local_theta_size)
                self.non_local_phi = MLP(input_dim=2 * hyperparams.fdim, output_dim=hyperparams.non_local_dim,
                                         hidden_size=hyperparams.non_local_phi_size)
                self.non_local_g = MLP(input_dim=2 * hyperparams.fdim, output_dim=2 * hyperparams.fdim,
                                       hidden_size=hyperparams.non_local_g_size)
                self.invariant_neighbors = MLP(input_dim=3 * hyperparams.fdim + initial_pos_dimension,
                                               output_dim=hyperparams.fdim,
                                               hidden_size=hyperparams.non_local_theta_size)

                self.specific_ego_list = []
                self.specific_neighbors_list = []
                self.domain_name_map_int = {}
                for domain_name in domain_list:
                    self.domain_name_map_int[domain_name.split('-')[0]] = 0
                counter = 0
                for key, value in self.domain_name_map_int.items():
                    self.domain_name_map_int[key] = counter
                    counter += 1
                    self.specific_ego_list.append(MLP(input_dim=hyperparams.fdim,
                                                      output_dim=hyperparams.fdim,
                                                      hidden_size=hyperparams.non_local_theta_size))
                    self.specific_neighbors_list.append(MLP(input_dim=3 * hyperparams.fdim + initial_pos_dimension,
                                                            output_dim=hyperparams.fdim,
                                                            hidden_size=hyperparams.non_local_theta_size))
                self.specific_ego_list = torch.nn.ModuleList(self.specific_ego_list)
                self.specific_neighbors_list = torch.nn.ModuleList(self.specific_neighbors_list)

                self.specific_neighbors_aggregator = MLP(input_dim=hyperparams.fdim * counter,
                                                         output_dim=hyperparams.fdim,
                                                         hidden_size=hyperparams.predictor_hidden_size)
                self.specific_ego_aggregator = MLP(input_dim=hyperparams.fdim * counter,
                                                   output_dim=hyperparams.fdim,
                                                   hidden_size=hyperparams.predictor_hidden_size)
                self.ego_domain_predictor = MLP(input_dim=2 * hyperparams.fdim,
                                                output_dim=counter,
                                                hidden_size=hyperparams.non_local_theta_size)
                self.neighbors_domain_predictor = MLP(input_dim=2 * hyperparams.fdim,
                                                      output_dim=counter,
                                                      hidden_size=hyperparams.non_local_theta_size)

            else:
                if self.hyperparams.ftraj_dimension_flag == 'mlp':
                    self.ftraj_fuse = MLP(input_dim=2 * hyperparams.fdim,
                                          output_dim=hyperparams.fdim,
                                          hidden_size=hyperparams.non_local_theta_size)

                self.default_function(self.hyperparams, domain_list, initial_pos_dimension)

            if self.hyperparams.fuse_dimension_flag == 'cat':
                self.x_predictor = MLP(input_dim=4 * hyperparams.fdim,
                                       output_dim=hyperparams.obs_len * 2,
                                       hidden_size=hyperparams.non_local_theta_size)
                if self.hyperparams.fuse_residual == 'True':
                    if self.hyperparams.ftraj_dimension_flag == 'cat':
                        self.predictor = MLP(input_dim=7 * hyperparams.fdim + initial_pos_dimension,
                                             output_dim=2 * hyperparams.pred_len,
                                             hidden_size=hyperparams.predictor_hidden_size)
                    else:
                        self.predictor = MLP(input_dim=6 * hyperparams.fdim + initial_pos_dimension,
                                             output_dim=2 * hyperparams.pred_len,
                                             hidden_size=hyperparams.predictor_hidden_size)
                else:
                    if self.hyperparams.ftraj_dimension_flag == 'cat':
                        self.predictor = MLP(input_dim=5 * hyperparams.fdim,
                                             output_dim=2 * hyperparams.pred_len,
                                             hidden_size=hyperparams.predictor_hidden_size)
                    else:
                        self.predictor = MLP(input_dim=4 * hyperparams.fdim,
                                             output_dim=2 * hyperparams.pred_len,
                                             hidden_size=hyperparams.predictor_hidden_size)
            else:
                if self.hyperparams.fuse_dimension_flag == 'mlp':
                    self.invariant_fuse = MLP(input_dim=2 * hyperparams.fdim,
                                              output_dim=hyperparams.fdim,
                                              hidden_size=hyperparams.non_local_theta_size)
                    self.specific_fuse = MLP(input_dim=2 * hyperparams.fdim,
                                             output_dim=hyperparams.fdim,
                                             hidden_size=hyperparams.non_local_theta_size)

                self.x_predictor = MLP(input_dim=2 * hyperparams.fdim,
                                       output_dim=hyperparams.obs_len * 2,
                                       hidden_size=hyperparams.non_local_theta_size)
                if self.hyperparams.fuse_residual == 'True':
                    if self.hyperparams.ftraj_dimension_flag == 'cat':
                        self.predictor = MLP(input_dim=5 * hyperparams.fdim + initial_pos_dimension,
                                             output_dim=2 * hyperparams.pred_len,
                                             hidden_size=hyperparams.predictor_hidden_size)
                    else:
                        self.predictor = MLP(input_dim=4 * hyperparams.fdim + initial_pos_dimension,
                                             output_dim=2 * hyperparams.pred_len,
                                             hidden_size=hyperparams.predictor_hidden_size)
                else:
                    if self.hyperparams.ftraj_dimension_flag == 'cat':
                        self.predictor = MLP(input_dim=3 * hyperparams.fdim,
                                             output_dim=2 * hyperparams.pred_len,
                                             hidden_size=hyperparams.predictor_hidden_size)
                    else:
                        self.predictor = MLP(input_dim=2 * hyperparams.fdim,
                                             output_dim=2 * hyperparams.pred_len,
                                             hidden_size=hyperparams.predictor_hidden_size)
        else:
            self.default_function(self.hyperparams, domain_list, initial_pos_dimension)
            self.predictor = MLP(input_dim=2 * hyperparams.fdim + initial_pos_dimension,
                                 output_dim=2 * hyperparams.pred_len,
                                 hidden_size=hyperparams.predictor_hidden_size)

        self.loss_similarity = torch.nn.CrossEntropyLoss()

    def default_function(self, hyperparams, domain_list, initial_pos_dimension):
        self.encoder_latent = MLP(input_dim=2 * hyperparams.fdim, output_dim=2 * hyperparams.zdim,
                                  hidden_size=hyperparams.enc_latent_size)
        self.decoder = MLP(input_dim=hyperparams.fdim + hyperparams.zdim, output_dim=2 * len(hyperparams.sub_goal_indexes),
                           hidden_size=hyperparams.dec_size)
        self.non_local_theta = MLP(input_dim=hyperparams.fdim, output_dim=hyperparams.non_local_dim,
                                   hidden_size=hyperparams.non_local_theta_size)
        self.non_local_phi = MLP(input_dim=hyperparams.fdim, output_dim=hyperparams.non_local_dim,
                                 hidden_size=hyperparams.non_local_phi_size)
        self.non_local_g = MLP(input_dim=hyperparams.fdim, output_dim=hyperparams.fdim,
                               hidden_size=hyperparams.non_local_g_size)
        self.invariant_neighbors = MLP(input_dim=2 * hyperparams.fdim + initial_pos_dimension,
                                       output_dim=hyperparams.fdim,
                                       hidden_size=hyperparams.non_local_theta_size)
        self.specific_ego_list = []
        self.specific_neighbors_list = []
        self.domain_name_map_int = {}
        for domain_name in domain_list:
            self.domain_name_map_int[domain_name.split('-')[0]] = 0
        counter = 0
        for key, value in self.domain_name_map_int.items():
            self.domain_name_map_int[key] = counter
            counter += 1
            self.specific_ego_list.append(MLP(input_dim=hyperparams.fdim,
                                              output_dim=hyperparams.fdim,
                                              hidden_size=hyperparams.non_local_theta_size))
            self.specific_neighbors_list.append(MLP(input_dim=2 * hyperparams.fdim + initial_pos_dimension,
                                                    output_dim=hyperparams.fdim,
                                                    hidden_size=hyperparams.non_local_theta_size))
        self.specific_ego_list = torch.nn.ModuleList(self.specific_ego_list)
        self.specific_neighbors_list = torch.nn.ModuleList(self.specific_neighbors_list)

        self.specific_neighbors_aggregator = MLP(input_dim=hyperparams.fdim * counter,
                                                 output_dim=hyperparams.fdim,
                                                 hidden_size=hyperparams.predictor_hidden_size)
        self.specific_ego_aggregator = MLP(input_dim=hyperparams.fdim * counter,
                                           output_dim=hyperparams.fdim,
                                           hidden_size=hyperparams.predictor_hidden_size)
        self.ego_domain_predictor = MLP(input_dim=2 * hyperparams.fdim,
                                        output_dim=counter,
                                        hidden_size=hyperparams.non_local_theta_size)
        self.neighbors_domain_predictor = MLP(input_dim=2 * hyperparams.fdim,
                                              output_dim=counter,
                                              hidden_size=hyperparams.non_local_theta_size)

    def non_local_social_pooling(self, feat, mask):

        # N,C
        theta_x = self.non_local_theta(feat)

        # C,N
        phi_x = self.non_local_phi(feat).transpose(1,0)

        # f_ij = (theta_i)^T(phi_j), (N,N)
        f = torch.matmul(theta_x, phi_x)

        # f_weights_i =  exp(f_ij)/(\sum_{j=1}^N exp(f_ij))
        f_weights = F.softmax(f, dim = -1)

        # setting weights of non neighbours to zero
        f_weights = f_weights * mask

        # rescaling row weights to 1
        f_weights = F.normalize(f_weights, p=1, dim=1)

        # ith row of all_pooled_f = \sum_{j=1}^N f_weights_i_j * g_row_j
        pooled_f = torch.matmul(f_weights, self.non_local_g(feat))

        return pooled_f + feat

    def forward(self, x, initial_pos, mask=None, dest=None, domain_name=None):

        # provide destination iff training
        # assert model.training
        # assert self.training ^ (dest is None)
        # assert self.training ^ (mask is None)

        # encode
        ftraj = self.encoder_past(x)

        if self.hyperparams.invariant_specific_flag == 'True':
            vanilla_ftraj = ftraj.clone()
            if domain_name is not None:
                domain_name = self.domain_name_map_int[domain_name.split('-')[0]]
                specific_ego_ftraj = self.specific_ego_list[domain_name](ftraj)
            else:
                features = torch.cat([self.specific_ego_list[i](ftraj).detach() for i in self.domain_name_map_int.values()], dim=1)
                specific_ego_ftraj = self.specific_ego_aggregator(features)
            invariant_ego_ftraj = self.invariant_ego(ftraj)

            if self.hyperparams.ftraj_dimension_flag == 'cat':
                ftraj = torch.cat((invariant_ego_ftraj, specific_ego_ftraj), dim=1)
            elif self.hyperparams.ftraj_dimension_flag == 'sum':
                ftraj = invariant_ego_ftraj + specific_ego_ftraj
            elif self.hyperparams.ftraj_dimension_flag == 'mlp':
                ftraj = self.ftraj_fuse(torch.cat((invariant_ego_ftraj, specific_ego_ftraj), dim=1))

            if self.hyperparams.ftraj_residual == 'True':
                if self.hyperparams.ftraj_dimension_flag != 'cat':
                    ftraj += vanilla_ftraj

        for i in range(self.nonlocal_pools):
            # non local social pooling
            ftraj = self.non_local_social_pooling(ftraj, mask)

        if self.hyperparams.ebm_flag == 'True':
            if self.training:
                pcd = True if len(self.replay_memory) == self.hyperparams.memory_size else False
                if pcd:
                    z_e_0 = self.replay_memory.sample(n=ftraj.size(0)).clone().detach().cuda()
                else:
                    z_e_0 = sample_p_0(e_init_sig=self.hyperparams.e_init_sig, n=ftraj.size(0),
                                       nz=self.hyperparams.zdim)
                z_e_k, _ = self.sample_langevin_prior_z(Variable(z_e_0), ftraj, pcd=pcd)
                for _z_e_k in z_e_k.clone().detach().cpu().split(1):
                    self.replay_memory.push(_z_e_k)
            else:
                z_e_0 = sample_p_0(e_init_sig=self.hyperparams.e_init_sig, n=ftraj.size(0), nz=self.hyperparams.zdim)
                z_e_k, _ = self.sample_langevin_prior_z(Variable(z_e_0), ftraj, pcd=False, y=None)
            z_e_k = z_e_k.double().cuda()

            if self.training:
                dest_features = self.encoder_dest(dest)
                features = torch.cat((ftraj, dest_features), dim=1)
                latent = self.encoder_latent(features)
                mu = latent[:, 0:self.hyperparams.zdim]
                logvar = latent[:, self.hyperparams.zdim:]

                var = logvar.mul(0.5).exp_()
                eps = torch.DoubleTensor(var.size()).normal_().cuda()
                z_g_k = eps.mul(var).add_(mu)
                z_g_k = z_g_k.double().cuda()

            if self.training:
                decoder_input = torch.cat((ftraj, z_g_k), dim=1)
            else:
                decoder_input = torch.cat((ftraj, z_e_k), dim=1)
        else:
            if not self.training:
                z = torch.Tensor(x.size(0), self.zdim)
                z.normal_(0, self.sigma)

            else:
                # during training, use the destination to produce generated_dest and use it again to predict final future points

                # CVAE code
                dest_features = self.encoder_dest(dest)
                features = torch.cat((ftraj, dest_features), dim=1)
                latent = self.encoder_latent(features)

                mu = latent[:, 0:self.zdim] # 2-d array
                logvar = latent[:, self.zdim:] # 2-d array

                var = logvar.mul(0.5).exp_()
                eps = torch.DoubleTensor(var.size()).normal_().to(x)
                z = eps.mul(var).add_(mu)

            z = z.double().to(x)
            decoder_input = torch.cat((ftraj, z), dim=1)

        generated_dest = self.decoder(decoder_input)

        if self.training:
            # prediction in training, no best selection
            generated_dest_features = self.encoder_dest(generated_dest)

            if self.hyperparams.initial_pos_flag == 'True':
                prediction_features = torch.cat((ftraj, generated_dest_features, initial_pos), dim=1)
            else:
                prediction_features = torch.cat((ftraj, generated_dest_features), dim=1)

            if self.hyperparams.invariant_specific_flag == 'True':

                if domain_name is not None:
                    specific_neighbors_features = self.specific_neighbors_list[domain_name](prediction_features)
                else:
                    features = torch.cat([self.specific_neighbors_list[i](prediction_features).detach() for i in self.domain_name_map_int.values()], dim=1)
                    specific_neighbors_features = self.specific_neighbors_aggregator(features)
                invariant_neighbors_features = self.invariant_neighbors(prediction_features)
                # invariant_neighbors_features = prediction_features
                # specific_neighbors_features = prediction_features

                if self.hyperparams.fuse_dimension_flag == 'cat':
                    invariant_fuse_features = torch.cat((invariant_ego_ftraj, invariant_neighbors_features), dim=1)
                    specific_fuse_features = torch.cat((specific_ego_ftraj, specific_neighbors_features), dim=1)
                elif self.hyperparams.fuse_dimension_flag == 'sum':
                    invariant_fuse_features = invariant_ego_ftraj + invariant_neighbors_features
                    specific_fuse_features = specific_ego_ftraj + specific_neighbors_features
                elif self.hyperparams.fuse_dimension_flag == 'mlp':
                    invariant_fuse_features = self.invariant_fuse(torch.cat((invariant_ego_ftraj, invariant_neighbors_features), dim=1))
                    specific_fuse_features = self.specific_fuse(torch.cat((specific_ego_ftraj, specific_neighbors_features), dim=1))

                hat_x = self.x_predictor(torch.cat((invariant_fuse_features, specific_fuse_features), dim=1))

                hat_ego_domain = self.ego_domain_predictor(torch.cat((invariant_ego_ftraj, specific_ego_ftraj), dim=1))
                hat_neighbors_domain = self.neighbors_domain_predictor(torch.cat((invariant_neighbors_features, specific_neighbors_features), dim=1))
                # hat_ego_domain = self.ego_domain_predictor(invariant_ego_ftraj)
                # hat_neighbors_domain = self.neighbors_domain_predictor(invariant_neighbors_features)

                if self.hyperparams.fuse_residual == 'True':
                    prediction_features = torch.cat(
                        (prediction_features, invariant_fuse_features, specific_fuse_features), dim=1)
                else:
                    prediction_features = torch.cat((invariant_fuse_features, specific_fuse_features), dim=1)

                domain_feature_list = [invariant_ego_ftraj, specific_ego_ftraj,
                                       invariant_neighbors_features, specific_neighbors_features,
                                       hat_x, x,
                                       hat_ego_domain, hat_neighbors_domain, domain_name]
            else:
                domain_feature_list = None

            pred_future = self.predictor(prediction_features)

            if self.hyperparams.ebm_flag == 'True':
                en_pos = self.ebm(z_g_k, ftraj).mean()
                en_neg = self.ebm(z_e_k.detach().clone(), ftraj).mean()
                ebm_cd = en_pos - en_neg
            else:
                ebm_cd = None

            return generated_dest, mu, logvar, pred_future, ebm_cd, domain_feature_list

        return generated_dest

    # separated for forward to let choose the best destination
    def predict(self, past, generated_dest, mask, initial_pos):
        ftraj = self.encoder_past(past)

        if self.hyperparams.invariant_specific_flag == 'True':
            vanilla_ftraj = ftraj.clone()
            features = torch.cat([self.specific_ego_list[i](ftraj).detach() for i in self.domain_name_map_int.values()], dim=1)
            specific_ego_ftraj = self.specific_ego_aggregator(features)
            invariant_ego_ftraj = self.invariant_ego(ftraj)

            if self.hyperparams.ftraj_dimension_flag == 'cat':
                ftraj = torch.cat((invariant_ego_ftraj, specific_ego_ftraj), dim=1)
            elif self.hyperparams.ftraj_dimension_flag == 'sum':
                ftraj = invariant_ego_ftraj + specific_ego_ftraj
            elif self.hyperparams.ftraj_dimension_flag == 'mlp':
                ftraj = self.ftraj_fuse(torch.cat((invariant_ego_ftraj, specific_ego_ftraj), dim=1))

            if self.hyperparams.ftraj_residual == 'True':
                if self.hyperparams.ftraj_dimension_flag != 'cat':
                    ftraj += vanilla_ftraj

        for i in range(self.nonlocal_pools):
            # non local social pooling
            ftraj = self.non_local_social_pooling(ftraj, mask)

        generated_dest_features = self.encoder_dest(generated_dest)

        if self.hyperparams.initial_pos_flag == 'True':
            prediction_features = torch.cat((ftraj, generated_dest_features, initial_pos), dim=1)
        else:
            prediction_features = torch.cat((ftraj, generated_dest_features), dim=1)

        if self.hyperparams.invariant_specific_flag == 'True':
            features = torch.cat([self.specific_neighbors_list[i](prediction_features).detach() for i in self.domain_name_map_int.values()], dim=1)
            specific_neighbors_features = self.specific_neighbors_aggregator(features)
            invariant_neighbors_features = self.invariant_neighbors(prediction_features)

            if self.hyperparams.fuse_dimension_flag == 'cat':
                invariant_fuse_features = torch.cat((invariant_ego_ftraj, invariant_neighbors_features), dim=1)
                specific_fuse_features = torch.cat((specific_ego_ftraj, specific_neighbors_features), dim=1)
            elif self.hyperparams.fuse_dimension_flag == 'sum':
                invariant_fuse_features = invariant_ego_ftraj + invariant_neighbors_features
                specific_fuse_features = specific_ego_ftraj + specific_neighbors_features
            elif self.hyperparams.fuse_dimension_flag == 'mlp':
                invariant_fuse_features = self.invariant_fuse(
                    torch.cat((invariant_ego_ftraj, invariant_neighbors_features), dim=1))
                specific_fuse_features = self.specific_fuse(
                    torch.cat((specific_ego_ftraj, specific_neighbors_features), dim=1))

            if self.hyperparams.fuse_residual == 'True':
                prediction_features = torch.cat((prediction_features, invariant_fuse_features, specific_fuse_features), dim=1)
            else:
                prediction_features = torch.cat((invariant_fuse_features, specific_fuse_features), dim=1)

        interpolated_future = self.predictor(prediction_features)
        return interpolated_future

    def ebm(self, z, condition, cls_output=False):
        condition_encoding = condition.detach().clone()
        z_c = torch.cat((z, condition_encoding), dim=1)
        conditional_neg_energy = self.EBM(z_c)
        assert conditional_neg_energy.shape == (z.size(0), self.hyperparams.ny)
        if cls_output:
            return - conditional_neg_energy
        else:
            return - conditional_neg_energy.logsumexp(dim=1)

    def sample_langevin_prior_z(self, z, condition, pcd=False, y=None):
        z = z.clone().detach()
        z.requires_grad = True
        _e_l_steps = self.hyperparams.e_l_steps_pcd if pcd else self.hyperparams.e_l_steps
        _e_l_step_size = self.hyperparams.e_l_step_size
        for i in range(_e_l_steps):
            if y is None:
                en = self.ebm(z, condition)
            else:
                en = self.ebm(z, condition, cls_output=True)[range(z.size(0)), y]
            z_grad = torch.autograd.grad(en.sum(), z)[0]

            z.data = z.data - 0.5 * _e_l_step_size * _e_l_step_size * (
                    z_grad + 1.0 / (self.hyperparams.e_prior_sig * self.hyperparams.e_prior_sig) * z.data)
            if self.hyperparams.e_l_with_noise:
                z.data += _e_l_step_size * torch.randn_like(z).data

            z_grad_norm = z_grad.view(z_grad.size(0), -1).norm(dim=1).mean()

        return z.detach(), z_grad_norm

    def calculate_loss(self, dest, reconstructed_dest, mean, log_var, criterion, future, interpolated_future, domain_feature_list):
        # reconstruction loss
        # RCL_dest = self.SIMSE(dest, reconstructed_dest)
        # ADL_traj = self.SIMSE(future, interpolated_future)
        RCL_dest = self.hyperparams.dest_loss_coeff * criterion(dest, reconstructed_dest)
        # RCL_dest += criterion(dest, interpolated_future[:, -2:])
        RCL_dest += criterion(dest, interpolated_future.view(dest.size(0), future.size(1)//2, 2)[:, self.hyperparams.sub_goal_indexes, :].view(dest.size(0), -1))
        ADL_traj = self.hyperparams.future_loss_coeff * criterion(future, interpolated_future)  # better with l2 loss
        # kl divergence loss
        if (mean is None) or (log_var is None):
            KLD = None
        else:
            KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
            KLD = self.hyperparams.kld_coeff * KLD

        #
        if domain_feature_list is None:
            domain_loss_list = None
        else:
            invariant_ego_ftraj, specific_ego_ftraj, invariant_neighbors_features, specific_neighbors_features, \
            hat_x, x, hat_ego_domain, hat_neighbors_domain, domain_name = domain_feature_list
            ego_diff = self.DiffLoss(invariant_ego_ftraj, specific_ego_ftraj)
            neighbors_diff = self.DiffLoss(invariant_neighbors_features, specific_neighbors_features)
            recon_mse = self.MSE(hat_x, x)
            recon_simse = self.SIMSE(hat_x, x)
            if domain_name is not None:
                domain_name = domain_name * torch.ones(x.shape[0]).long().cuda()
                ego_similar = self.loss_similarity(hat_ego_domain, domain_name)
                neighbors_similar = self.loss_similarity(hat_neighbors_domain, domain_name)
                # neighbors_similar = torch.zeros(ego_similar.shape).to(ego_similar)
            else:
                ego_similar, neighbors_similar = None, None

            domain_loss_list = [ego_diff, neighbors_diff, recon_mse, recon_simse, ego_similar, neighbors_similar]

        assert not (RCL_dest != RCL_dest), "RCL_dest is NaN"
        assert not (KLD != KLD), "KLD is NaN"
        assert not (ADL_traj != ADL_traj), "ADL_traj is NaN"

        return RCL_dest, KLD, ADL_traj, domain_loss_list

    def MSE(self, pred, real):
        diffs = torch.add(real, -pred)
        n = torch.numel(diffs.data)
        mse = torch.sum(diffs.pow(2)) / n

        return mse

    def SIMSE(self, pred, real):
        diffs = torch.add(real, - pred)
        n = torch.numel(diffs.data)
        simse = torch.sum(diffs).pow(2) / (n ** 2)

        return simse

    def DiffLoss(self, input1, input2):

        batch_size = input1.size(0)
        input1 = input1.view(batch_size, -1)
        input2 = input2.view(batch_size, -1)

        input1_l2_norm = torch.norm(input1, p=2, dim=1, keepdim=True).detach()
        input1_l2 = input1.div(input1_l2_norm.expand_as(input1) + 1e-6)

        input2_l2_norm = torch.norm(input2, p=2, dim=1, keepdim=True).detach()
        input2_l2 = input2.div(input2_l2_norm.expand_as(input2) + 1e-6)

        diff_loss = torch.mean((input1_l2.t().mm(input2_l2)).pow(2))

        return diff_loss
