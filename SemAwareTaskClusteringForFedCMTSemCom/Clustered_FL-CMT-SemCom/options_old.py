
import argparse

def args_parser():
    parser = argparse.ArgumentParser()

    # Federated arguments
    parser.add_argument('--epochs', type=int, default=10,
                        help='number of rounds of training')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random Seed for reproducibility')
    parser.add_argument('--deterministic', action='store_true',
                        help='Use cuDNN deterministic mode (slower, reproducible)')
    parser.add_argument('--num_users',type=int, default=3,
                        help = 'number of users: K')
    parser.add_argument('--frac',type=float,default=1,
                        help='the fraction of clients: C')
    parser.add_argument('--local_ep',type=int,default=10, # changed from 3 to 10 on 15.05
                        help='the number of local epochs: E')
    parser.add_argument('--local_bs',type=int,default=100,
                        help='local batch size: B')
    parser.add_argument('--lr', type = float, default=0.01,
                        help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.5,
                        help='SGD momentum (default: 0.5)')

    # Model arguments (Updated for auto-encoder)
    parser.add_argument('--model', type=str, default='MLP', choices=['CNN','AutoEncoder','MultiTaskCNN','MLP'],
                        help='Choose between CNN, AutoEncoder, MultiTaskCNN or MLP')
    parser.add_argument('--kernel_num',type=int,default=9,
                        help='Number of each kind of kernel')
    parser.add_argument('--kernel_sizes',type=str,default='3,4,5',
                        help='Comma-separated kernel size to use for convolution')
    parser.add_argument('--num_channels',type=int, default=1,
                        help='Number of channels of images')
    parser.add_argument('--norm',type=str, default='batch_norm',
                        help='batch_norm, layer_norm or None')
    parser.add_argument('--num_filters', type=int, default=32,
                        help="Number of filters for conv nets --32 for \
                        mini-imagenet, 64 for omiglot.")
    parser.add_argument('--max_pool', type=str, default='True',
                        help="Whether use max pooling rather than \
                             strided convolutions")
    parser.add_argument('--latent_dim', type=int, default=64,
                        help='Latent Space dimension for AutoEncoder')

    # Other Arguments
    parser.add_argument('--dataset',type=str, default='cifar10',
                        help='Name of dataset')
    parser.add_argument('--num_classes',type=int, default=10,
                        help='Number of classes')
    parser.add_argument('--gpu', type = int, default=0, help=" GPU id to use (default: 0) ")
    parser.add_argument('--cpu', action='store_true', help="Force CPU even if CUDA is available.")
    parser.add_argument('--optimizer',type=str, default='sgd', choices= ['sgd','Adam'],
                        help='Type of Optimizer')
    # parser.add_argument('--iid',type=int,default=1,
    #                     help='Default set to IID. Set to 0 for non-IID')
    parser.add_argument('--unequal',type=int,default=0,
                        help="Default set to 0 for equal data-splits. \
                             Set to 1 for unequal data-splits")
    parser.add_argument('--partition_type', type=str, default='custom_skewed',
                        choices=['custom_skewed'],
                        help="Data partitioning method: 'iid', 'noniid', or 'custom_skewed'")

    parser.add_argument('--stopping_rounds',type=int, default=10,
                        help='Rounds of early stopping')
    parser.add_argument('--verbose',type=str,default=1,
                        help='Verbose')
    # parser.add_argument('--seed',type=int,default=1,
    #                     help='Random Seed')

    # Argument for input noise
    # parser.add_argument('--sigma', type=float, default=0.0,
    #                     help='Standard deviation for AWGN noise; 0 disables the noise')

    # Arguments for no normalization
    parser.add_argument('--snr_db_signal', type=float, default=None,
                        help='Signal to Noise ratio desired in an AWGN in the channel between the decoder \n'
                             'and encoder during training; None implies no noise.\n'
                             'sigma_signal can be used instead of this')

    parser.add_argument('--sigma_signal', type=float, default=None,
                        help='Noise standard deviation of the AWGN in the channel between the encoder and decoder')

    parser.add_argument('--signal_normalize', action='store_true',
                        help= 'Controls whether normalization is applied to encoder outputs or not; \n'
                              'Passing the flag applies normalization and not passing it does not')


    parser.add_argument('--downlink', action='store_true', help='Controls if noise is to be\n'
                        ' applied in the channel from clients to Global Server')
    parser.add_argument('--uplink', action='store_true', help='Controls if noise is to be applied\n'
                        'in the channel from Global Server to clients')

    parser.add_argument('--sigma_agg', type=float, default=None,
                        help='Noise standard deviation of the AWGN channel between between clients and GS')

    parser.add_argument('--weight_normalize', action='store_true', help='Controls whether normalization is \n'
                        'applied to weights or not. Passing the flag applies normalization and not passing it does not')

    parser.add_argument('--snr_db_agg', type=float, default= None,
                        help = 'Signal to Noise ratio desired in an AWGN in the channel between clients and GS \n'
                               'This is to be used when aggregating model parameters.\n'
                               'None implies no noise')
    parser.add_argument('--use_noise_regularizer', action='store_true',
                        help='Use FedSem-style noise-aware gradient norm regularization')


    args = parser.parse_args()

    return args


