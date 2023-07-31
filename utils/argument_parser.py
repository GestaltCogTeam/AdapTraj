import argparse


def get_basic_setting_parser(parser):
    parser.add_argument('--seed', default=3, type=int)
    parser.add_argument('--gpu_deterministic', type=bool, default=False, help='set cudnn in deterministic mode (slow)')
    parser.add_argument("--num_workers", default=0, type=int)
    return parser


def get_lbebm_model_parser(parser):
    parser.add_argument("--sub_goal_indexes", default=[2, 5, 8, 11], type=list)
    parser.add_argument('--bottleneck_dim', default=32, type=int)
    parser.add_argument('--num_layers', default=1, type=int)

    parser.add_argument("--dec_size", default=[1024, 512, 1024], type=list)
    parser.add_argument("--enc_dest_size", default=[256, 128], type=list)
    parser.add_argument("--enc_latent_size", default=[256, 512], type=list)
    parser.add_argument("--enc_past_size", default=[512, 256], type=list)
    parser.add_argument("--predictor_hidden_size", default=[1024, 512, 256], type=list)
    parser.add_argument("--non_local_theta_size", default=[256, 128, 64], type=list)
    parser.add_argument("--non_local_phi_size", default=[256, 128, 64], type=list)
    parser.add_argument("--non_local_g_size", default=[256, 128, 64], type=list)
    parser.add_argument("--non_local_dim", default=128, type=int)
    parser.add_argument("--fdim", default=16, type=int)
    parser.add_argument("--kld_coeff", default=0.5, type=float)
    parser.add_argument("--future_loss_coeff", default=1, type=float)
    parser.add_argument("--dest_loss_coeff", default=2, type=float)
    parser.add_argument("--mu", default=0, type=float)
    parser.add_argument("--nonlocal_pools", default=3, type=int)
    parser.add_argument("--sigma", default=1.3, type=float)
    parser.add_argument("--zdim", default=16, type=int)
    parser.add_argument('--e_prior_sig', type=float, default=2, help='prior of ebm z')
    parser.add_argument('--e_init_sig', type=float, default=2, help='sigma of initial distribution')
    parser.add_argument('--e_activation', type=str, default='lrelu', choices=['gelu', 'lrelu', 'swish', 'mish'])
    parser.add_argument('--e_activation_leak', type=float, default=0.2)
    parser.add_argument('--e_energy_form', default='identity', choices=['identity', 'tanh', 'sigmoid', 'softplus'])
    parser.add_argument('--e_l_steps', type=int, default=20, help='number of langevin steps')
    parser.add_argument('--e_l_steps_pcd', type=int, default=20, help='number of langevin steps')
    parser.add_argument('--e_l_step_size', type=float, default=0.4, help='stepsize of langevin')
    parser.add_argument('--e_l_with_noise', default=True, type=bool, help='noise term of langevin')
    parser.add_argument('--e_sn', default=False, type=bool, help='spectral regularization')
    parser.add_argument('--e_lr', default=0.00003, type=float)
    parser.add_argument('--e_is_grad_clamp', type=bool, default=False, help='whether doing the gradient clamp')
    parser.add_argument('--e_max_norm', type=float, default=25, help='max norm allowed')
    parser.add_argument('--e_decay', default=1e-4, help='weight decay for ebm')
    parser.add_argument('--e_gamma', default=0.998, help='lr decay for ebm')
    parser.add_argument('--e_beta1', default=0.9, type=float)
    parser.add_argument('--e_beta2', default=0.999, type=float)
    parser.add_argument('--memory_size', default=200000, type=int)
    parser.add_argument('--ny', default=1, type=int)
    parser.add_argument('--embedding_dim', default=16, type=int)
    parser.add_argument('--h_dim', default=32, type=int)
    parser.add_argument('--noise_dim', default=8, type=int)
    parser.add_argument('--mlp_dim', default=64, type=int)
    parser.add_argument("--addNoise", type=int, default=1)
    return parser


def get_stgat_model_parser(parser):
    def int_tuple(s):
        return tuple(int(i) for i in s.split(","))

    parser.add_argument("--n_coordinates", type=int, default=2, help="Number of coordinates")
    parser.add_argument("--traj_lstm_hidden_size", default=32, type=int)
    parser.add_argument("--heads", type=str, default="4,1", help="Heads in each layer, splitted with comma")
    parser.add_argument("--hidden-units", type=str, default="16",
                        help="Hidden units in each hidden layer, splitted with comma")
    parser.add_argument("--graph_network_out_dims", default=32, type=int,
                        help="dims of every node after through GAT module")
    parser.add_argument("--graph_lstm_hidden_size", default=32, type=int)
    parser.add_argument("--alpha", type=float, default=0.2, help="Alpha for the leaky_relu")
    parser.add_argument("--stgat_noise_dim", default=(16,), type=int_tuple)
    parser.add_argument("--noise_type", default="gaussian")
    parser.add_argument("--add_confidence", default=False, type=bool)
    parser.add_argument("--counter", default=False, type=bool, help='counterfactual analysis')
    parser.add_argument("--step_one_epochs", default=150, type=int)
    parser.add_argument("--step_two_epochs", default=100, type=int)
    parser.add_argument("--unbiased", default=False, type=bool, help='Use an Unbiased Estimator for SGD')
    parser.add_argument("--ic_weight", default=0.5, type=float, help='Invariance constraint strenght')
    return parser


def get_causalmotion_model_parser(parser):
    def int_tuple(s, delim=','):
        return tuple(int(i) for i in s.strip().split(delim))

    # general training
    parser.add_argument("--finetune", default="all", type=str)
    parser.add_argument("--num_epochs", default='0-0-2-2-2-2', type=lambda x: int_tuple(x, '-'))

    # learning rates
    parser.add_argument("--lrclass", default=1e-2, type=float, help="initial learning rate for style classifier optimizer")
    parser.add_argument("--lrstgat", default=1e-3, type=float, help="initial learning rate for stgat optimizer")
    parser.add_argument("--lrstyle", default=5e-4, type=float, help="initial learning rate for style encoder optimizer")
    parser.add_argument('--lrinteg', default=0.01, type=float, help="initial learning rate for the integrator optimizer")

    # architecture (Style)
    parser.add_argument('--stylefs', type=str, default='all', choices=['all', 'traj', 'graph'])
    parser.add_argument("--relsocial", action='store_false')  # default value true
    parser.add_argument('--contrastive', default=1, type=float)
    parser.add_argument("--aggrstyle", default='minpol-mean', type=str)
    parser.add_argument("--classification", default=3, type=int)

    # method
    parser.add_argument("--irm", default=1.0, type=float, help='IRM parameter (lambda)')
    parser.add_argument("--vrex", default=0.0, type=float, help='v-REx parameter (beta)')

    return parser


def get_data_preprocess_parser(parser):
    parser.add_argument('--train_fraction', type=float, default=0.6)
    parser.add_argument('--val_fraction', type=float, default=0.2)
    parser.add_argument('--process_flag', default='frame', choices=['frame', 'agent'])
    parser.add_argument('--train_data_rotate', default='False', choices=['True', 'False'])
    parser.add_argument('--train_data_reverse', default='False', choices=['True', 'False'])
    parser.add_argument('--init_data_pos_method', default='last', choices=['first', 'last'])
    parser.add_argument("--data_scale", default=60, type=float)
    parser.add_argument('--delim', default='\t', type=str)
    parser.add_argument('--skip', default=1, type=int)
    parser.add_argument('--min_ped', default=1, type=int)
    parser.add_argument('--threshold', default=0.002, type=float, help='non linear threshold')
    parser.add_argument('--time_thresh', default=0, type=float, help='discard!!!')
    parser.add_argument('--dist_thresh', default=50, type=float, help='discard!!!')
    parser.add_argument('--mask', default=False, type=bool, help='discard!!!')
    return parser


def get_experiment_ablation_parser(parser):
    parser.add_argument('--obs_len', type=int, default=8)
    parser.add_argument('--pred_len', type=int, default=12)
    parser.add_argument('--best_dest', type=int, default=20)
    parser.add_argument('--best_traj', type=int, default=1)
    return parser


def get_optimize_parser(parser):
    parser.add_argument('--init_learning_rate', type=float, default=1e-4)
    parser.add_argument("--lr_decay_step_size", default=150, type=int)
    parser.add_argument('--learning_decay_rate', type=float, default=0.5)
    parser.add_argument('--clipping_threshold', default=2.0, type=float)
    parser.add_argument('--clipping_threshold_flag', default='False', choices=['True', 'False'])
    parser.add_argument('--use_lrschd', default='False', choices=['True', 'False'])
    return parser


def get_invariant_specific_parser(parser):
    parser.add_argument("--invariant_specific_flag", default='True', choices=['True', 'False'])
    parser.add_argument("--invariant_flag", default='True', choices=['True', 'False'])
    parser.add_argument("--specific_flag", default='True', choices=['True', 'False'])
    parser.add_argument("--similar_dimension_used_flag", default='invariant+specific', choices=['invariantOnly', 'invariant+specific'])
    parser.add_argument("--ftraj_residual", default='True', choices=['True', 'False'])
    parser.add_argument("--fuse_residual", default='True', choices=['True', 'False'])

    parser.add_argument("--train_mode", default='randomFreeze', choices=['randomFreeze', 'ganFreeze'])
    parser.add_argument("--aggregator_epochs", type=int, default=100)
    parser.add_argument("--aggregator_end_epochs", type=int, default=250)
    parser.add_argument("--aggregator_ratio", type=float, default=0.5)
    parser.add_argument("--aggregator_gan_step", type=int, default=3)
    parser.add_argument("--former_domain_weight", type=float, default=200)
    parser.add_argument("--freeze_mode", default='allButAggregator', choices=['allButAggregator',
                                                                              'allButAggregatorPredictor',
                                                                              'allButAggregatorPredictorDiffConstruct',
                                                                              'allButOursPredictor',
                                                                              'allButOursPredictor+NonLocal'])
    parser.add_argument("--predict_mode", default='ego_later', choices=['ego_first', 'ego_later'])
    parser.add_argument("--low_lr_fraction", type=float, default=0.1)
    parser.add_argument("--high_lr_fraction", type=float, default=0.9)

    parser.add_argument("--ego_invariant_flag", default='True', choices=['True', 'False'])
    parser.add_argument("--neighbors_invariant_flag", default='True', choices=['True', 'False'])
    parser.add_argument("--fuse_invariant_flag", default='True', choices=['True', 'False'])
    parser.add_argument("--fuse_specific_flag", default='True', choices=['True', 'False'])
    parser.add_argument("--ego_specific_flag", default='True', choices=['True', 'False'])
    parser.add_argument("--neighbors_specific_flag", default='True', choices=['True', 'False'])
    parser.add_argument("--ebm_flag", default='False', choices=['True', 'False'])
    parser.add_argument("--ebm_loss_coeff", type=float, default=0.5)
    parser.add_argument("--rand_domain_name", default='True', choices=['True', 'False'])
    parser.add_argument("--simse_flag", default='simse+mse', choices=['simse', 'mse', 'simse+mse'])
    parser.add_argument("--ftraj_dimension_flag", default='mlp', choices=['cat', 'sum', 'mlp'])
    parser.add_argument("--fuse_dimension_flag", default='sum', choices=['cat', 'sum', 'mlp'])
    parser.add_argument("--initial_pos_flag", default='False', choices=['True', 'False'])
    parser.add_argument("--plan_flag", default='True', choices=['True', 'False'])
    return parser


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:2')
    parser.add_argument("--train_dataset_name", type=str, default='ETH,UCY,L-CAS,SYI')  # choices: ETH,UCY,L-CAS,SYI,SDD,WildTrack,CFF
    parser.add_argument("--test_dataset_name", type=str, default='SDD')  # choices: ETH,UCY,L-CAS,SDD,WildTrack,CFF,SYI
    parser.add_argument("--model_name", default='AdapTraj', choices=['LBEBM', 'LBEBM-counter', 'PECNet', 'PECNet-counter', 'AdapTraj'])
    parser.add_argument("--model_init_weight", default='False', choices=['True', 'False'])
    parser.add_argument("--train_epochs", type=int, default=300)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--batch_hetero', default='hom', choices=['het', 'hom'])
    parser.add_argument('--dropout', default=0, type=float)
    parser = get_basic_setting_parser(parser)
    parser = get_data_preprocess_parser(parser)
    parser = get_optimize_parser(parser)
    parser = get_experiment_ablation_parser(parser)
    parser = get_lbebm_model_parser(parser)
    parser = get_stgat_model_parser(parser)
    parser = get_causalmotion_model_parser(parser)
    parser = get_invariant_specific_parser(parser)
    return parser.parse_args()


args = get_parser()

