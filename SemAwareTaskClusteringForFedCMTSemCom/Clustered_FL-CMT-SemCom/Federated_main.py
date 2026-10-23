# Note - This is still to be checked if this includes everything for auto-encoder
import os
import copy
import time
import pickle
import json
import numpy as np
from tqdm import tqdm
import tikzplotlib

import torch
import sys
from tensorboardX import SummaryWriter

from options import args_parser
# from update import test_inference
from update import LocalUpdate
from model import CNNMNIST,AutoEncoder
from utils import (
    get_dataset,
    average_weights,
    exp_details,
    print_mapped_label_distribution,
    calculate_encoder_update,
    cluster_clients_by_similarity,
)
from model import SharedEncoder, TaskHead
from update import test_inference_multitask
from collections import defaultdict
from channel import awgn_agg_normalize, awgn_agg_no_normalize
from channel import rms_power_normalize_weight, snr_db_to_sigma_weight_normalize, snr_db_to_sigma_weight_no_normalize
import pandas as pd
from update import collect_class_features, plot_class_gaussianity

print("Python:", sys.version)
print("Torch version:", torch.__version__)
print("Torch CUDA version:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("Device count:", torch.cuda.device_count())

os.makedirs('../../save', exist_ok=True)

print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)
print("Device count:", torch.cuda.device_count())

if __name__ == '__main__':
    start_time = time.time()

    # Define paths
    path_project = os.path.abspath('...')

    args = args_parser()
    automatic_clustering = not args.single_cluster and args.fixed_clusters is None

    print("[Args check]", args)
    print("[Args check] has cpu flag?", hasattr(args, "cpu"))

    if args.dataset != 'EuroSAT' or args.model != 'MLP':
        raise ValueError(
            "This program implements clustered EuroSAT training and requires "
            "--dataset=EuroSAT --model=MLP."
        )

    if automatic_clustering and not 1 <= args.cluster_warmup_rounds <= args.epochs:
        raise ValueError(
            "cluster_warmup_rounds must be between 1 and args.epochs."
        )

    if not -1.0 <= args.similarity_threshold <= 1.0:
        raise ValueError(
            "similarity_threshold must be between -1 and 1."
        )

    import random, numpy as np, torch


    def set_seed(seed: int, deterministic: bool = False):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        # Full determinism
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


    set_seed(args.seed, args.deterministic)

    if args.partition_type != 'iid':
        args.iid = 0
    else:
        args.iid = 1

    exp_details(args)


    # if args.gpu is not None:
    #     torch.cuda.set_device(int(args.gpu))
    # device = 'cuda' if torch.cuda.is_available() and args.gpu is not None else 'cpu'

    use_cuda = torch.cuda.is_available() and (not args.cpu)
    device = 'cuda' if use_cuda else 'cpu'

    if device == 'cuda':
        torch.cuda.set_device(args.gpu)

    print(
        f"[Server] device={device} | cuda_available={torch.cuda.is_available()} | gpu={args.gpu} | cpu_flag={args.cpu}")

    # Local dataset and user groups

    train_dataset, test_dataset, user_groups, test_user_groups, label_mappings = get_dataset(args)

    # Keep a separate plain-text clustering log so cluster decisions are not
    # lost when the terminal history is truncated.
    run_timestamp = time.strftime('%Y%m%d_%H%M%S')
    save_dir = args.results_dir or os.path.join('Results', f'run_{run_timestamp}')
    args.quant_stats_path = os.path.join(
        save_dir, "quantization_batches.csv"
    )
    # Refuse to mix a new run with previous results.
    os.makedirs(save_dir, exist_ok=False)
    cluster_log_dir = save_dir
    with open(os.path.join(save_dir, 'run_config.json'), 'w', encoding='utf-8') as config_file:
        json.dump(vars(args), config_file, indent=2)
    cluster_log_path = os.path.join(
        cluster_log_dir,
        f'clustering_log_{run_timestamp}.txt',
    )

    def write_cluster_log(message):
        with open(cluster_log_path, 'a', encoding='utf-8') as log_file:
            log_file.write(f'{message}\n')

    write_cluster_log('Clustered FL EuroSAT run')
    write_cluster_log(f'Start time: {time.strftime("%Y-%m-%d %H:%M:%S")}')
    write_cluster_log(f'Seed: {args.seed}')
    write_cluster_log(f'Communication rounds: {args.epochs}')
    write_cluster_log(f'Local epochs per round: {args.local_ep}')
    write_cluster_log(f'Warm-up rounds: {args.cluster_warmup_rounds}')
    write_cluster_log(f'Cosine-similarity threshold: {args.similarity_threshold}')
    write_cluster_log(f'Client fraction: {args.frac}')
    write_cluster_log(f'Latent noise sigma: {args.sigma_signal}')
    write_cluster_log(f'Aggregation noise sigma: {args.sigma_agg}')
    write_cluster_log(f'Single cluster forced: {args.single_cluster}')
    write_cluster_log(f'Fixed client groups from round 1: {args.fixed_clusters}')
    write_cluster_log(f'Automatic clustering enabled: {automatic_clustering}')
    write_cluster_log(f'Encoder normalization: {args.encoder_norm}')
    write_cluster_log(f'Regularizer: enabled={args.use_noise_regularizer}, '
                      f'norm={args.noise_reg_norm}, strength={args.noise_reg_strength}, '
                      f'warmup={args.noise_reg_warmup_rounds}')
    write_cluster_log(f'Channel flags: uplink={args.uplink}, downlink={args.downlink}, '
                      f'signal_normalize={args.signal_normalize}, weight_normalize={args.weight_normalize}')
    print(f'[Clustering] Detailed log: {os.path.abspath(cluster_log_path)}')

    logger = SummaryWriter('../logs')

    try:

        from sampling import print_raw_label_distribution

        print_raw_label_distribution(user_groups, train_dataset, title="TRAIN raw label distribution")
        print_mapped_label_distribution(user_groups, train_dataset, label_mappings, title="TRAIN mapped task-label distribution")
        print_raw_label_distribution(test_user_groups, test_dataset, title="TEST raw-label distribution")
        print_mapped_label_distribution(test_user_groups,test_dataset, label_mappings, title="TEST mapped task-label distribution")

        # Build model
        # If convolutional neural network
        if args.model == 'CNN':
            if args.dataset =='mnist':
                global_model = CNNMNIST(args= args).to(device)
            else:
                exit('Invalid dataset selected')

        elif args.model == 'AutoEncoder':
            if args.dataset == 'mnist':
                # global_model = AutoEncoder(args=args).to(device) #Commented coz not using auto-encoder arguments
                global_model = AutoEncoder().to(device)


        elif args.model == 'MLP':

            if args.dataset != 'EuroSAT':
                raise ValueError(

                    "This clustered shared-encoder implementation currently expects EuroSAT."

                )

            # Initially all clients share one common encoder.

            common_encoder = SharedEncoder(args).to(device)

            # Created dynamically as {0: 3, 1: 5, 2: 3}.

            client_output_dims = args.client_output_dims

            client_heads = {

                client_id: TaskHead(

                    input_dim=common_encoder.output_dim,

                    output_dim=output_dim,

                ).to(device)

                for client_id, output_dim in client_output_dims.items()

            }

            # All groups start from identical encoder weights, with independent
            # storage. Deep copies do not consume extra random initialization.
            clusters = (
                {cluster_id: members for cluster_id, members in enumerate(args.fixed_clusters)}
                if args.fixed_clusters is not None
                else {0: sorted(client_heads.keys())}
            )
            cluster_encoders = {
                cluster_id: common_encoder if cluster_id == 0 else copy.deepcopy(common_encoder)
                for cluster_id in clusters
            }
            client_to_cluster = {
                client_id: cluster_id
                for cluster_id, members in clusters.items()
                for client_id in members
            }
            write_cluster_log(f'Initial clusters: {clusters}')

            for client_id, head in client_heads.items():
                print(

                    f"Client {client_id} - "

                    f"Head Output Dim: {head.fc[-1].out_features}"

                )

        # Set the model to train and set it to device
        # Set models to train
        if args.model != 'MLP':
            global_model.to(device)
            global_model.train()
            print(global_model)
        else:
            for encoder in cluster_encoders.values():
                encoder.train()
            for head in client_heads.values():
                head.train()


        print(f"\nTotal users: {args.num_users}")
        for uid in user_groups:
            print(f"User {uid} has {len(user_groups[uid])} samples")

        # Added to track loss per client
        client_losses = {i: [] for i in range(args.num_users)}
        val_acc_list, net_list = [], []
        cv_loss, cv_acc = [], []
        print_every = 2
        val_loss_pre, counter = 0, 0

        # Simulation of distributed users and multiple communication rounds

        # Initialization of histories and other dictionaries (outside the loop)
        # Per-client TRAIN histories (length will become == args.epochs)
        client_train_acc = defaultdict(list)  # values in %
        client_train_loss = defaultdict(list)

        # Per-client TEST histories (filled by test_inference_multitask each round)
        client_test_accuracies = defaultdict(list)  # %
        client_test_losses = defaultdict(list)

        # Overall histories (one scalar per round)
        overall_test_acc_macro_history = []
        overall_test_loss_macro_history = []

        # Server averages per round (optional overall training curves)
        server_avg_train_loss = []
        server_avg_train_acc = []

        # Also keep a simple per-round training loss list if you like
        train_loss = []  # one value per round (same as server_avg_train_loss)

        test_acc, test_loss = None, None

        for epoch in tqdm(range(args.epochs)):
            print(f'\n | Global Training Round: {epoch+1} |\n')

            if args.model == 'MLP':
                for encoder in cluster_encoders.values():
                    encoder.train()
                for head in client_heads.values():
                    head.train()
            else:
                global_model.train()

            m = max(int(args.frac * args.num_users),1)
            idxs_users = np.random.choice(range(args.num_users), m, replace=False)

            participating_client_ids = [int(client_id) for client_id in idxs_users]
            write_cluster_log('')
            write_cluster_log(f'===== Communication round {epoch + 1} =====')
            write_cluster_log(
                f'Participating clients: {participating_client_ids}'
            )
            write_cluster_log(f'Clusters before training: {clusters}')
            write_cluster_log(
                f'Client-to-cluster mapping before training: {client_to_cluster}'
            )


            participated = set()

            # Pre-append accuracy and loss so every client has exactly args.epochs elements
            for cid in range(args.num_users):
                client_train_acc[cid].append(np.nan)
                client_train_loss[cid].append(np.nan)

            # Per-round client encoder payloads, clean updates, and FedAvg weights.
            local_weights_by_client = {}
            client_updates = {}
            local_sample_counts_by_client = {}


            # Client-wise local training.
            for idx in idxs_users:
                print(f"Training Client {idx} with task head output size: {client_heads[idx].fc[-1].out_features}")

                idx = int(idx)

                cluster_id = client_to_cluster[idx]
                source_encoder = cluster_encoders[cluster_id]

                # Save the exact reference before local training.
                reference_weights = copy.deepcopy(
                    source_encoder.state_dict()
                )

                local_encoder = copy.deepcopy(source_encoder).to(device)

                if args.downlink:
                    noisy_state = awgn_agg_no_normalize(
                        local_encoder.state_dict(),
                        sigma_agg=args.sigma_agg,
                        snr_db_agg=args.snr_db_agg,
                    )
                    local_encoder.load_state_dict(noisy_state)

                local_model = LocalUpdate(
                    args=args,
                    dataset=train_dataset,
                    idxs=user_groups[idx],
                    logger=logger,
                    encoder=local_encoder, # DUpdated from encoder = shared_encoder coz this isnt copying the model locally
                    head=client_heads[idx],  # Use client-specific head
                    client_id=idx,
                    label_mappings=label_mappings
                )
                # Step 2 FL - Local weights per-client found/updated
                w, loss = local_model.update_weights(global_round=epoch+1)

                # Preserve the clean encoder weights before applying uplink noise.
                clean_local_weights = copy.deepcopy(w)

                # Calculate updates only while all clients still use the common encoder.


                aggregation_weights = copy.deepcopy(clean_local_weights)

                if args.uplink:
                    if args.weight_normalize:
                        aggregation_weights = rms_power_normalize_weight(
                            aggregation_weights
                        )
                        aggregation_weights = awgn_agg_normalize(
                            aggregation_weights,
                            sigma_agg=args.sigma_agg,
                            snr_db_agg=args.snr_db_agg,
                        )
                    else:
                        aggregation_weights = awgn_agg_no_normalize(
                            aggregation_weights,
                            sigma_agg=args.sigma_agg,
                            snr_db_agg=args.snr_db_agg,
                        )
                if automatic_clustering and epoch < args.cluster_warmup_rounds:
                    client_updates[idx] = calculate_encoder_update(
                        local_weights=aggregation_weights,
                        reference_weights=reference_weights,
                    )

                local_weights_by_client[idx] = aggregation_weights
                local_sample_counts_by_client[idx] = len(user_groups[idx])

                client_train_loss[idx][-1] = float(loss)

                acc_train_frac, _ = local_model.inference()  # accuracy on this client's TRAIN data
                client_train_acc[idx][-1] = float(acc_train_frac) * 100.0  # store as %

                participated.add(idx)

            if args.model == 'MLP':
                if automatic_clustering and epoch < args.cluster_warmup_rounds:
                    expected_clients = set(range(args.num_users))
                    participating_clients = set(local_weights_by_client.keys())

                    if participating_clients != expected_clients:
                        raise ValueError(
                            "All clients must participate during clustering warm-up. "
                            "Run with --frac=1."
                        )

                    # Before the final warm-up round, all clients continue to
                    # update one common encoder using sample-weighted FedAvg.
                    if epoch < args.cluster_warmup_rounds - 1:
                        active_clients = sorted(local_weights_by_client.keys())

                        common_weights = average_weights(
                            [
                                local_weights_by_client[client_id]
                                for client_id in active_clients
                            ],
                            [
                                local_sample_counts_by_client[client_id]
                                for client_id in active_clients
                            ],
                        )

                        cluster_encoders[0].load_state_dict(common_weights)

                    # At the end of warm-up, calculate similarities and create
                    # one independently aggregated encoder per detected cluster.
                    else:
                        clusters, similarity_matrix = cluster_clients_by_similarity(
                            client_updates,
                            similarity_threshold=args.similarity_threshold,
                        )

                        similarity_client_ids = sorted(client_updates.keys())
                        print(
                            "\nCosine-similarity client order:",
                            similarity_client_ids,
                        )
                        print("Cosine-similarity matrix:")
                        print(similarity_matrix.numpy())
                        print("Detected clusters:")
                        print(clusters)

                        write_cluster_log(
                            f'Cosine-similarity client order: '
                            f'{similarity_client_ids}'
                        )
                        write_cluster_log('Cosine-similarity matrix:')
                        write_cluster_log(
                            np.array2string(
                                similarity_matrix.numpy(),
                                precision=6,
                                suppress_small=False,
                            )
                        )
                        write_cluster_log(f'Detected clusters: {clusters}')

                        client_to_cluster = {
                            client_id: cluster_id
                            for cluster_id, members in clusters.items()
                            for client_id in members
                        }
                        write_cluster_log(
                            f'Client-to-cluster mapping: {client_to_cluster}'
                        )

                        new_cluster_encoders = {}

                        for cluster_id, members in clusters.items():
                            averaged_weights = average_weights(
                                [
                                    local_weights_by_client[client_id]
                                    for client_id in members
                                ],
                                [
                                    local_sample_counts_by_client[client_id]
                                    for client_id in members
                                ],
                            )

                            cluster_encoder = SharedEncoder(args).to(device)
                            cluster_encoder.load_state_dict(averaged_weights)
                            new_cluster_encoders[cluster_id] = cluster_encoder

                        cluster_encoders = new_cluster_encoders

                else:
                    # Once clusters have been formed, keep their membership
                    # fixed and aggregate encoder weights only within clusters.
                    for cluster_id, members in clusters.items():
                        active_members = [
                            client_id
                            for client_id in members
                            if client_id in local_weights_by_client
                        ]

                        # With partial participation, a cluster may have no
                        # active client in this round. Its encoder is then kept.
                        if not active_members:
                            continue

                        averaged_weights = average_weights(
                            [
                                local_weights_by_client[client_id]
                                for client_id in active_members
                            ],
                            [
                                local_sample_counts_by_client[client_id]
                                for client_id in active_members
                            ],
                        )

                        cluster_encoders[cluster_id].load_state_dict(
                            averaged_weights
                        )

                write_cluster_log(f'Clusters after round {epoch + 1}: {clusters}')
                write_cluster_log(
                    f'Client-to-cluster mapping after round {epoch + 1}: '
                    f'{client_to_cluster}'
                )

            # per-round server averages (ignore NaNs for non-participants)
            server_avg_train_loss.append(float(np.nanmean([client_train_loss[c][-1] for c in client_train_loss])))
            server_avg_train_acc.append(float(np.nanmean([client_train_acc[c][-1] for c in client_train_acc])))

            # (optional) keep the same value in train_loss for plotting
            train_loss.append(server_avg_train_loss[-1])

            # Calculate avg training accuracy over all users at every epoch
            # list_acc, list_loss = [], []
            if args.model == 'MLP':
                for encoder in cluster_encoders.values():
                    encoder.eval()
                for head in client_heads.values():
                    head.eval()

                # Test each client with its own independently noisy downloaded encoder
                for cid in range(args.num_users):
                    cluster_id = client_to_cluster[cid]

                    test_encoder = copy.deepcopy(
                        cluster_encoders[cluster_id]
                    ).to(device)

                    # Downlink parameter noise during testing/inference:
                    # GS aggregated encoder -> noisy client-side received encoder
                    if args.downlink:
                        noisy_state = awgn_agg_no_normalize(
                            test_encoder.state_dict(),
                            sigma_agg=args.sigma_agg,
                            snr_db_agg=args.snr_db_agg
                        )
                        test_encoder.load_state_dict(noisy_state)

                    if epoch == args.epochs - 1:
                        collected = collect_class_features(
                            args=args,
                            encoder=test_encoder,
                            dataset=train_dataset,
                            idxs=user_groups[cid],
                            client_id=cid,
                            label_mappings=label_mappings,
                            num_classes=client_heads[cid].fc[-1].out_features,
                        )

                        torch.save(
                            collected,
                            os.path.join(save_dir, f"class_features_client_{cid}.pt"),
                        )

                        plot_class_gaussianity(
                            collected,
                            os.path.join(save_dir, f"gaussianity_client_{cid}.pdf"),
                        )

                    single_client_head = {cid: client_heads[cid]}

                    _, _, client_test_accuracies, client_test_losses = test_inference_multitask(
                        args,
                        test_encoder,
                        single_client_head,
                        test_dataset,
                        test_user_groups,
                        client_test_accuracies,
                        client_test_losses,
                        label_mappings=label_mappings
                    )

                macro_acc = float(np.mean([
                    client_test_accuracies[cid][-1]
                    for cid in range(args.num_users)
                ]))
                macro_loss = float(np.mean([
                    client_test_losses[cid][-1]
                    for cid in range(args.num_users)
                ]))

                overall_test_acc_macro_history.append(float(macro_acc))
                overall_test_loss_macro_history.append(float(macro_loss))

            else:
                global_model.eval()


            # Print global training loss after every 'i' rounds
            if (epoch+1) % print_every == 0:
                print(f'\nAvg Training Stats after {epoch+1} global rounds:')
                print(f'Training Loss: {np.mean(np.array(train_loss))}')
                print('Train Accuracy: {:.2f}% \n'.format(server_avg_train_acc[-1])) # No *100

        stats = pd.read_csv(args.quant_stats_path)

        summary = stats.groupby(
            ["round", "client", "bits", "upper_bound"],
            as_index=False,
        ).agg(
            batches=("batch", "size"),
            elements=("elements", "sum"),
            minimum=("minimum", "min"),
            maximum=("maximum", "max"),
            clipped_elements=("clipped_elements", "sum"),
            squared_error_sum=("squared_error_sum", "sum"),
            power_before_sum=("power_before_sum", "sum"),
            power_after_sum=("power_after_sum", "sum"),
        )

        summary["clipped_percent"] = (
                100 * summary["clipped_elements"] / summary["elements"]
        )
        summary["quantization_mse"] = (
                summary["squared_error_sum"] / summary["elements"]
        )
        summary["power_before"] = (
                summary["power_before_sum"] / summary["elements"]
        )
        summary["power_after"] = (
                summary["power_after_sum"] / summary["elements"]
        )

        summary.to_csv(
            os.path.join(save_dir, "quantization_summary.csv"),
            index=False,
        )

        print(f'\n Results after {args.epochs} global rounds of training:')
        print('|---- Avg Train Accuracy: {:.2f}%'.format(server_avg_train_acc[-1])) # No *100
        if client_test_accuracies is not None:
            print('|---- Test Accuracy: {:.2f}%'.format(overall_test_acc_macro_history[-1]))
            print('|---- Test Loss: {:.4f}'.format(overall_test_loss_macro_history[-1]))
        else:
            print('|---- Test Accuracy: Not Applicable (e.g., in case of an AutoEncoder')

        # Saving the objects train_loss and train_accuracy
        file_name = (
            f'{save_dir}/objects/'
            '{}_{}_{}_C[{}]_partition[{}]_E[{}]_B[{}]'
            '_sigma_signal[{}]_sigma_agg[{}].pkl'
        ).format(
            args.dataset,
            args.model,
            args.epochs,
            args.frac,
            args.partition_type,
            args.local_ep,
            args.local_bs,
            args.sigma_signal,
            args.sigma_agg,
        )

        os.makedirs(save_dir, exist_ok=True)
        tikz_dir = save_dir
        os.makedirs(tikz_dir, exist_ok=True)

        os.makedirs(os.path.dirname(file_name), exist_ok=True)
        with open(file_name, 'wb') as f:
            pickle.dump([server_avg_train_loss, server_avg_train_acc], f)

        print('\n Total Run Time: {0:0.4f}'.format(time.time()-start_time))
        write_cluster_log('')
        write_cluster_log(
            f'Run completed at {time.strftime("%Y-%m-%d %H:%M:%S")}'
        )
        write_cluster_log(
            f'Total runtime seconds: {time.time() - start_time:.4f}'
        )
        print(f'Clustering log saved to: {os.path.abspath(cluster_log_path)}')

        # Plotting
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        # Plotting curves

        rounds = list(range(1, args.epochs + 1))

        # 1) Training loss (server-average per round)
        plt.figure()
        plt.title('Training Loss vs Communication Rounds Clustered_FL-CMT-SemCom')
        plt.plot(rounds, server_avg_train_loss)
        plt.xlabel('Communication Rounds')
        plt.ylabel('Training Loss')
        plt.xticks(rounds)
        plt.savefig(f'{save_dir}/server_avg_train_loss_per_round_Clustered_FL-CMT-SemCom.png')
        tikzplotlib.save(f'{save_dir}/server_avg_train_loss_per_round_Clustered_FL-CMT-SemCom.tikz')

        # 2) Training accuracy (server-average per round)
        plt.figure()
        plt.title('Average Train Accuracy vs Communication Rounds Clustered_FL-CMT-SemCom')
        plt.plot(rounds, server_avg_train_acc)
        plt.xlabel('Communication Rounds')
        plt.ylabel('Average Train Accuracy (%)')
        plt.xticks(rounds)
        plt.savefig(f'{save_dir}/server_avg_train_accuracy_per_round_Clustered_FL-CMT-SemCom.png')
        tikzplotlib.save(
            f'{save_dir}/server_avg_train_accuracy_per_round_Clustered_FL-CMT-SemCom.tikz')

        # 3) Test accuracy (overall system)
        plt.figure()
        plt.title('Overall TEST Accuracy vs Communication Rounds Clustered_FL-CMT-SemCom')
        plt.plot(rounds, overall_test_acc_macro_history)
        plt.xlabel('Communication Rounds')
        plt.ylabel('Test Accuracy (%)')
        plt.xticks(rounds)
        plt.savefig(f'{save_dir}/overall_test_accuracy_macro_per_round_Clustered_FL-CMT-SemCom.png')
        tikzplotlib.save(
            f'{save_dir}/overall_test_accuracy_macro_per_round_Clustered_FL-CMT-SemCom.tikz')


        # 4) Per-client TRAIN accuracy
        def dict_to_df(d, rounds):
            import pandas as pd
            cols = sorted(d.keys())
            data = {f'client_{cid}': d[cid] for cid in cols}
            df = pd.DataFrame(data, index=rounds)
            df.index.name = 'round'
            return df


        import pandas as pd


        df_train_acc = dict_to_df(client_train_acc, rounds)
        df_train_acc.to_csv(f'{save_dir}/client_train_accuracy_per_round_Clustered_FL-CMT-SemCom.csv')

        plt.figure()
        plt.title('Client-wise TRAIN Accuracy per Round Clustered_FL-CMT-SemCom')
        for cid in sorted(client_train_acc.keys()):
            plt.plot(rounds, client_train_acc[cid], label=f'Client {cid}')
        plt.xlabel('Communication Rounds');
        plt.ylabel('Train Accuracy (%)')
        plt.xticks(rounds);
        plt.legend(loc='lower right');
        plt.tight_layout()
        plt.savefig(f'{save_dir}/client_train_accuracy_per_round_Clustered_FL-CMT-SemCom.png')
        tikzplotlib.save(f'{save_dir}/client_train_accuracy_per_round_Clustered_FL-CMT-SemCom.tikz')

        # 5) Per-client TRAIN loss
        df_train_loss = dict_to_df(client_train_loss, rounds)
        df_train_loss.to_csv(f'{save_dir}/client_train_loss_per_round_Clustered_FL-CMT-SemCom.csv')

        plt.figure()
        plt.title('Client-wise TRAIN Loss per Round Clustered_FL-CMT-SemCom')
        for cid in sorted(client_train_loss.keys()):
            plt.plot(rounds, client_train_loss[cid], label=f'Client {cid}')
        plt.xlabel('Communication Rounds');
        plt.ylabel('Train Loss')
        plt.xticks(rounds);
        plt.legend(loc='lower right');
        plt.tight_layout()
        plt.savefig(f'{save_dir}/client_train_loss_per_round_Clustered_FL-CMT-SemCom.png')
        tikzplotlib.save(f'{save_dir}/client_train_loss_per_round_Clustered_FL-CMT-SemCom.tikz')

        # 6) Per-client TEST accuracy
        df_test_acc = dict_to_df(client_test_accuracies, rounds)
        df_test_acc.to_csv(f'{save_dir}/client_test_accuracy_per_round_Clustered_FL-CMT-SemCom.csv')
        plt.figure()
        plt.title('Client-wise TEST Accuracy per Round Clustered_FL-CMT-SemCom')
        for cid in sorted(client_test_accuracies.keys()):
            plt.plot(rounds, client_test_accuracies[cid], label=f'Client {cid}')
        plt.xlabel('Communication Rounds');
        plt.ylabel('Test Accuracy (%)')
        plt.xticks(rounds);
        plt.legend(loc='lower right');
        plt.tight_layout()
        plt.savefig(f'{save_dir}/client_test_accuracy_per_round_Clustered_FL-CMT-SemCom.png')
        tikzplotlib.save(f'{save_dir}/client_test_accuracy_per_round_Clustered_FL-CMT-SemCom.tikz')

        # 7) Per-client TEST loss
        df_test_loss = dict_to_df(client_test_losses, rounds)
        df_test_loss.to_csv(f'{save_dir}/client_test_loss_per_round_Clustered_FL-CMT-SemCom.csv')
        plt.figure()
        plt.title('Client-wise TEST Loss per Round Clustered_FL-CMT-SemCom')
        for cid in sorted(client_test_losses.keys()):
            plt.plot(rounds, client_test_losses[cid], label=f'Client {cid}')
        plt.xlabel('Communication Rounds');
        plt.ylabel('Test Loss')
        plt.xticks(rounds);
        plt.legend(loc='lower right');
        plt.tight_layout()
        plt.savefig(f'{save_dir}/client_test_loss_per_round_Clustered_FL-CMT-SemCom.png')
        tikzplotlib.save(f'{save_dir}/client_test_loss_per_round_Clustered_FL-CMT-SemCom.tikz')

        # 8) Overall test loss
        plt.figure()
        plt.title('Overall TEST (MACRO) Loss vs Communication Rounds Clustered_FL-CMT-SemCom')
        plt.plot(rounds, overall_test_loss_macro_history)
        plt.xlabel('Communication Rounds');
        plt.ylabel('Test Loss')
        plt.xticks(rounds)
        plt.savefig(f'{save_dir}/overall_test_loss_macro_per_round_Clustered_FL-CMT-SemCom.png')
        tikzplotlib.save(f'{save_dir}/overall_test_loss_macro_per_round_Clustered_FL-CMT-SemCom.tikz')

    except Exception as error:
        write_cluster_log('')
        write_cluster_log(
            f'Run failed with {type(error).__name__}: {error}'
        )
        raise

    finally:
        logger.close()