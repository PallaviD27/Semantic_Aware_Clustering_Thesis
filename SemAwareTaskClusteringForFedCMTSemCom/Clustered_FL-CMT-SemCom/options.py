
import argparse

def args_parser():
    parser = argparse.ArgumentParser()

    # Federated arguments
    parser.add_argument('--epochs', type=int, default=30,
                        help='number of rounds of training')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random Seed for reproducibility')
    parser.add_argument('--deterministic', action='store_true',
                        help='Use cuDNN deterministic mode (slower, reproducible)')
    parser.add_argument('--num_users',type=int, default=3,
                        help = 'number of users: K')
    parser.add_argument('--frac',type=float,default=1,
                        help='the fraction of clients: C')
    parser.add_argument('--local_ep',type=int,default=3, # Changed so that it is consistent with Clustered code  07.06
                        help='the number of local epochs: E')
    parser.add_argument('--local_bs',type=int,default=16, # was 16
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
    parser.add_argument('--num_channels',type=int, default=12,
                        help='Number of channels of images, BigEarthNet has 12 channels from Sentinel')
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
    parser.add_argument('--dataset',type=str, default='EuroSAT', choices=['mnist','BENet','EuroSAT'],
                        help='Name of dataset')
    parser.add_argument('--num_classes',type=int, default=19,
                        help='Number of classes')
    parser.add_argument('--gpu', type=int, default=0, help=" GPU id to use (default: 0) ")
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
    # Arguments for new dataset
    parser.add_argument('--bigearthnet_s2_dir', type=str, default='../BigEarthNet-S2/BigEarthNet-S2',
                        help='Path to extracted BigEarthNet-S2 image folder')
    parser.add_argument(
        '--eurosat_dir',
        type=str,
        default='../EuroSAT_MS/EuroSAT_MS',
        help='Path to the extracted multispectral EuroSAT dataset'
    )
    parser.add_argument('--bigearthnet_s2_metadata', type=str, default='../BigEarthNet-S2/metadata_light.csv',
                        help='Path to BigEarthNet metadata.parquet')
    parser.add_argument('--image_size', type=int, default=120,
                        help = 'Image size of resize bigearthnet bands to')
    parser.add_argument('--max_train_samples_per_client', type=int, default=3000, # was 1000 for bigearth need to be changed for EuroSat
                        help='Maximum training samples per client')
    parser.add_argument('--max_test_samples_per_client', type=int, default=700, # was 300 for bigearth need to be changed for EuroSat
                        help='Maximum test samples per client')

    parser.add_argument(
        '--cluster_warmup_rounds',
        type=int,
        default=3,
        help='Number of common-encoder rounds before forming clusters'
    )

    parser.add_argument(
        '--similarity_threshold',
        type=float,
        default=-0.005, # Changing from 0.8 to -0.005 just to have fedavg
        help='Minimum average cosine similarity for clustering clients'
    )

    parser.add_argument('--use_noise_regularizer', action='store_true',
                        help='Use FedSem-style noise-aware gradient norm regularization')

    cluster_mode = parser.add_mutually_exclusive_group()
    cluster_mode.add_argument('--single_cluster', action='store_true',
                              help='Keep every client in cluster 0 for every round; requires frac=1')
    cluster_mode.add_argument('--fixed_clusters', type=str, default=None,
                              help='Fixed groups from round 1, e.g. "0,1;2"; requires frac=1')
    parser.add_argument('--encoder_norm', choices=['none', 'group'], default='none',
                        help='Optional per-sample GroupNorm inside the encoder')
    parser.add_argument('--noise_reg_strength', type=float, default=8.0,
                        help='Multiplier of sigma_agg squared (legacy value: 8)')
    parser.add_argument('--noise_reg_norm', choices=['l2', 'squared'], default='l2',
                        help='Encoder gradient penalty; l2 is the legacy unsquared norm')
    parser.add_argument('--noise_reg_warmup_rounds', type=int, default=0,
                        help='Linearly ramp regularizer strength over this many rounds')
    parser.add_argument('--max_grad_norm', type=float, default=0.0,
                        help='Clip combined encoder/head gradient norm; 0 disables clipping')
    parser.add_argument('--results_dir', type=str, default=None,
                        help='Output directory; default is a new timestamped Results/run_* folder')
    parser.add_argument('--Gr', type=float, default=10.0,
                        help='linear antenna power gain for the receiver')
    parser.add_argument('--Gt', type=float, default=10.0,
                        help='linear antenna power gain for the transmitter')
    parser.add_argument('--wavelength_m', type=float, default=300000000.0/8.2e9,
                        help='Wavelength for X-band in metres')
    parser.add_argument('--d_m',type=float,default=786000,
                        help='Satellite-to-ground slant range in metres')
    parser.add_argument('--use_ste', type=bool, default=True,
                        help='To control the use of the STE trick')

    args = parser.parse_args()
    if (args.single_cluster or args.fixed_clusters is not None) and args.frac != 1.0:
        parser.error('--single_cluster and --fixed_clusters require --frac=1')
    if args.fixed_clusters is not None:
        try:
            groups = [
                [int(client_id.strip()) for client_id in group.split(',')]
                for group in args.fixed_clusters.split(';')
            ]
        except ValueError:
            parser.error('--fixed_clusters must contain nonempty integer groups, e.g. "0,1;2"')
        members = [client_id for group in groups for client_id in group]
        if sorted(members) != list(range(args.num_users)):
            parser.error('--fixed_clusters must include every client exactly once, using IDs 0 to num_users-1')
        args.fixed_clusters = groups
    if args.lr <= 0 or args.noise_reg_strength < 0 or args.max_grad_norm < 0:
        parser.error('lr must be positive; noise_reg_strength and max_grad_norm nonnegative')
    if args.noise_reg_warmup_rounds < 0:
        parser.error('noise_reg_warmup_rounds must be nonnegative')
    if args.use_noise_regularizer:
        if args.sigma_agg is None or args.sigma_agg < 0 or args.snr_db_agg is not None:
            parser.error('The gradient regularizer requires a fixed nonnegative sigma_agg and no snr_db_agg')

    return args


