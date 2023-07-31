import random
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


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


class LBEBM_counter(nn.Module):
    def __init__(self, hyperparams):
        super(LBEBM_counter, self).__init__()

        self.hyperparams = hyperparams

        self.encoder_past = MLP(input_dim=hyperparams.obs_len*2, output_dim=hyperparams.fdim, hidden_size=hyperparams.enc_past_size)
        self.encoder_dest = MLP(input_dim=len(hyperparams.sub_goal_indexes)*2, output_dim=hyperparams.fdim, hidden_size=hyperparams.enc_dest_size)
        self.encoder_latent = MLP(input_dim=2*hyperparams.fdim, output_dim=2*hyperparams.zdim, hidden_size=hyperparams.enc_latent_size)
        self.decoder = MLP(input_dim=hyperparams.fdim+hyperparams.zdim, output_dim=len(hyperparams.sub_goal_indexes)*2, hidden_size=hyperparams.dec_size)
        self.predictor = MLP(input_dim=2*hyperparams.fdim, output_dim=2*(hyperparams.pred_len), hidden_size=hyperparams.predictor_hidden_size)

        self.non_local_theta = MLP(input_dim=hyperparams.fdim, output_dim = hyperparams.non_local_dim, hidden_size=hyperparams.non_local_theta_size)
        self.non_local_phi = MLP(input_dim=hyperparams.fdim, output_dim = hyperparams.non_local_dim, hidden_size=hyperparams.non_local_phi_size)
        self.non_local_g = MLP(input_dim=hyperparams.fdim, output_dim = hyperparams.fdim, hidden_size=hyperparams.non_local_g_size)

        self.EBM = nn.Sequential(
            nn.Linear(hyperparams.zdim + hyperparams.fdim, 200),
            nn.GELU(),
            nn.Linear(200, 200),
            nn.GELU(),
            nn.Linear(200, hyperparams.ny),
            )
                    
        self.replay_memory = ReplayMemory(hyperparams.memory_size)

    def forward(self, x, dest=None, mask=None, y=None):
        
        ftraj = self.encoder_past(x)
        ftraj_c = torch.zeros_like(ftraj).to(ftraj)

        if mask != None:
            for _ in range(self.hyperparams.nonlocal_pools):
                ftraj = self.non_local_social_pooling(ftraj, mask)
                ftraj_c = self.non_local_social_pooling(ftraj_c, mask)

        if self.training:
            pcd = True if len(self.replay_memory) == self.hyperparams.memory_size else False
            if pcd:
                z_e_0 = self.replay_memory.sample(n=ftraj.size(0)).clone().detach().cuda()
            else:
                z_e_0 = sample_p_0(e_init_sig=self.hyperparams.e_init_sig, n=ftraj.size(0), nz=self.hyperparams.zdim)
            z_e_k, _ = self.sample_langevin_prior_z(Variable(z_e_0), ftraj, pcd=pcd)
            for _z_e_k in z_e_k.clone().detach().cpu().split(1):
                self.replay_memory.push(_z_e_k)
        else:
            z_e_0 = sample_p_0(e_init_sig=self.hyperparams.e_init_sig, n=ftraj.size(0), nz=self.hyperparams.zdim)
            z_e_k, _ = self.sample_langevin_prior_z(Variable(z_e_0), ftraj, pcd=False, y=y)
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
        generated_dest = self.decoder(decoder_input)

        if self.training:
            generated_dest_features = self.encoder_dest(generated_dest)
            prediction_features = torch.cat((ftraj, generated_dest_features), dim=1)
            # counterfactual
            prediction_features_c = torch.cat((ftraj_c, generated_dest_features), dim=1)
            pred_future = self.predictor(prediction_features - prediction_features_c)

            en_pos = self.ebm(z_g_k, ftraj).mean()
            en_neg = self.ebm(z_e_k.detach().clone(), ftraj).mean()
            cd = en_pos - en_neg

            return generated_dest, mu, logvar, pred_future, cd, en_pos, en_neg, pcd

        return generated_dest

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

    def predict(self, past, generated_dest, mask=None):
        ftraj = self.encoder_past(past)
        generated_dest_features = self.encoder_dest(generated_dest)
        prediction_features = torch.cat((ftraj, generated_dest_features), dim=1)
        # counterfactual
        ftraj_c = torch.zeros_like(ftraj).to(ftraj)
        if mask != None:
            for _ in range(self.hyperparams.nonlocal_pools):
                ftraj_c = self.non_local_social_pooling(ftraj_c, mask)
        prediction_features_c = torch.cat((ftraj_c, generated_dest_features), dim=1)
        interpolated_future = self.predictor(prediction_features - prediction_features_c)

        return interpolated_future

    def non_local_social_pooling(self, feat, mask):
        theta_x = self.non_local_theta(feat)
        phi_x = self.non_local_phi(feat).transpose(1,0)
        f = torch.matmul(theta_x, phi_x)
        f_weights = F.softmax(f, dim = -1)
        f_weights = f_weights * mask
        f_weights = F.normalize(f_weights, p=1, dim=1)
        pooled_f = torch.matmul(f_weights, self.non_local_g(feat))

        return pooled_f + feat

    def calculate_loss(self, dest, dest_recon, mean, log_var, criterion, future, interpolated_future, sub_goal_indexes):
        dest_loss = criterion(dest, dest_recon)
        future_loss = criterion(future, interpolated_future)
        subgoal_reg = criterion(dest_recon, interpolated_future.view(dest.size(0), future.size(1)//2, 2)[:, sub_goal_indexes, :].view(dest.size(0), -1))
        kl = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())

        return dest_loss, future_loss, kl, subgoal_reg
