import copy # Import Python's in-built copy function which can be used to create
            # deep copy (completely independent copy) or shallow-copy
            # (copies outer object, maintains reference for inner object)

import torch
from torchvision import datasets, transforms # Loads standard datasets like MNIST, CIFAR10, etc.
                                             # And loads the module transforms used for pre-processing image data

from sampling import custom_skewed_partition
import numpy as np
from collections import Counter
from bigearthnet_dataset import BigEarthNetS2Dataset,CLIENT_TASKS,build_client_partitions
from eurosat_dataset import (
    EUROSAT_CLASSES,
    EUROSAT_CLIENT_TASKS,
    build_eurosat_client_partitions,
    build_eurosat_label_mappings,
    get_eurosat_datasets,
    print_eurosat_client_distribution,
)
import torch.nn.functional as F
from sklearn.cluster import AgglomerativeClustering

def get_dataset(args):
    '''
    Returns train and test datasets and a user group which is a dict
    where the keys are the user index and the values are the corresponding
    data for each of those users.
    :param args:
    :return: train and test datsets
    '''

    if args.dataset == 'mnist':
        data_dir = '../../data/mnist/'

        apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])

        train_dataset = datasets.MNIST(data_dir,train=True, download=True,
                                       transform=apply_transform)

        test_dataset = datasets.MNIST(data_dir,train=False,download=True,
                                      transform=apply_transform)

        # Define task-specific label mapping only if MLP is selected
        label_mappings = {}

        # if args.model == 'MLP':
        #     label_mappings = {
        #         0: lambda y: 0 if y == 2 else 1,  # Client 0: [2, not-2]
        #         1: lambda y: 0 if y == 1 else 1 if y == 2 else 2 if y == 3 else 3,  # Client 1: [1, 2, 3, none]
        #         2: lambda y: 0 if y == 6 else 1  # Client 2: [6, not-6]
        #     }

        if args.model == 'MLP':
            def map_client_0(y):  # 2 vs not-2
                return 1 if y == 2 else 0

            def map_client_1(y):  # 1, 2, 3, none
                if y == 1:
                    return 1
                elif y == 2:
                    return 2
                # elif y == 3:
                #     return 3
                else:
                    return 0

            def map_client_2(y):  # 6 vs not-6
                return 1 if y == 6 else 0

            label_mappings = {
                0: map_client_0,
                1: map_client_1,
                2: map_client_2
            }

            # 🔍 Debug: check that all labels map correctly for each client
            for client_id, map_func in label_mappings.items():
                for y in range(10):
                    print(f"Client {client_id}, Label {y} -> Mapped: {map_func(y)}")

        # Mode of data sampling amongst users. This part controls
        # the data heterogeneity

        if args.partition_type=='iid':
            # Sample iid user data from MNIST
            user_groups = mnist_iid(train_dataset,args.num_users)
            test_user_groups = mnist_iid(test_dataset, args.num_users)

        elif args.partition_type=='noniid':
            # Sample non-IID user data from MNIST
            if args.unequal:
                # Choose unequal splits for every user
                user_groups = mnist_noniid_unequal(train_dataset,args.num_users)
                test_user_groups = mnist_noniid_unequal(test_dataset, args.num_users)

            else:
                user_groups = mnist_non_iid(train_dataset,args.num_users)
                test_user_groups = mnist_non_iid(test_dataset, args.num_users)

        elif args.partition_type=='custom_skewed':
            try:
                user_groups = custom_skewed_partition(train_dataset, args.num_users, seed=args.seed) # Training data partition
                test_user_groups = {u: np.arange(len(test_dataset)) for u in range(args.num_users)}
                # test_user_groups = custom_skewed_partition(test_dataset, args.num_users, seed = args.seed + 1 ) # Testing data partition
            except Exception as e:
                print("[Partition Error]", repr(e))
                raise


        else:
            raise ValueError("Invalid partition_type. Choose from: 'iid', 'noniid', 'custom_skewed'")

        return (
            train_dataset,
            test_dataset,
            user_groups,
            test_user_groups,
            label_mappings,
        )

    elif args.dataset == 'BENet':
        print("\n [Dataset] Loading BigEarthNet-S2")
        label_mappings = None

        # Loading BigEarthNet-S2's training dataset
        train_dataset = BigEarthNetS2Dataset(s2_dir=args.bigearthnet_s2_dir, metadata_path=args.bigearthnet_s2_metadata, split = "train",
                                             image_size=args.image_size,)

        # Loading BigEarthNet-S2's test dataset
        test_dataset = BigEarthNetS2Dataset(s2_dir=args.bigearthnet_s2_dir, metadata_path=args.bigearthnet_s2_metadata,
                                            split="test", image_size=args.image_size,)

        # Creating client partitions of train dataset based on client tasks
        user_groups = build_client_partitions(dataset=train_dataset, client_tasks=CLIENT_TASKS,
                                              max_samples_per_client=args.max_train_samples_per_client, seed=args.seed)
        test_user_groups = build_client_partitions(dataset=test_dataset, client_tasks=CLIENT_TASKS,
                                                   max_samples_per_client=args.max_test_samples_per_client, seed=args.seed+100)

        # Store BigEarthNet information in args so other files can use it later.
        args.num_classes = 19
        args.multilabel = True
        args.client_task_indices = CLIENT_TASKS

        # Since we are currently all but can use RGB bands B04, B03, B02.
        args.num_channels = 12

        # Each client's head output size should match the number of task labels.
        args.client_output_dims = {
            client_id: len(task_indices)
            for client_id, task_indices in CLIENT_TASKS.items()
        }

        print("\n[BigEarthNet info]")
        print("Task type: multi-label classification")
        print(f"Number of global classes: {args.num_classes}")
        print(f"Number of image channels: {args.num_channels}")
        print(f"Image size: {args.image_size} x {args.image_size}")
        print("Label format: float32 multi-hot vector [19]")
        print("Training loss needed later: BCEWithLogitsLoss")

        return train_dataset, test_dataset, user_groups, test_user_groups, label_mappings

    elif args.dataset == "EuroSAT":
        print("\n[Dataset] Loading EuroSAT multispectral dataset")

        if args.num_users != len(EUROSAT_CLIENT_TASKS):
            raise ValueError(
                f"EuroSAT requires {len(EUROSAT_CLIENT_TASKS)} clients, "
                f"but num_users={args.num_users}."
            )

        # Convert EuroSAT global labels into each client's local label space.
        label_mappings = build_eurosat_label_mappings(
            EUROSAT_CLIENT_TASKS
        )

        # Store EuroSAT information for model creation and training.
        args.num_classes = len(EUROSAT_CLASSES)  # 10 global classes
        args.multilabel = False
        args.num_channels = 12
        args.image_size = 64

        args.client_output_dims = {
            client_id: len(task_classes)
            for client_id, task_classes in EUROSAT_CLIENT_TASKS.items()
        }
        # Produces {0: 3, 1: 5, 2: 2}

        train_dataset, test_dataset = get_eurosat_datasets(
            root=args.eurosat_dir,
            train_fraction=0.8,
            seed=args.seed,
        )

        user_groups = build_eurosat_client_partitions(
            dataset=train_dataset,
            client_tasks=EUROSAT_CLIENT_TASKS,
            max_samples_per_client=args.max_train_samples_per_client,
            seed=args.seed,
        )

        test_user_groups = build_eurosat_client_partitions(
            dataset=test_dataset,
            client_tasks=EUROSAT_CLIENT_TASKS,
            max_samples_per_client=args.max_test_samples_per_client,
            seed=args.seed + 1,
        )

        print_eurosat_client_distribution(
            train_dataset,
            user_groups,
            EUROSAT_CLIENT_TASKS,
            "EuroSAT training distribution",
        )

        print_eurosat_client_distribution(
            test_dataset,
            test_user_groups,
            EUROSAT_CLIENT_TASKS,
            "EuroSAT test distribution",
        )

        return (
            train_dataset,
            test_dataset,
            user_groups,
            test_user_groups,
            label_mappings,
        )

    else:
        raise ValueError('Invalid dataset')



def print_mapped_label_distribution(dict_users, dataset, label_mappings, title ="Mapped label distribution"):
    '''
    Print mapped task-label counts for each client subset.
    :param dict_users:
    :param dataset:
    :param label_mappings:
    :param title:
    :return:
    '''
    print(f"\n ======={title}=======")
    for client_id, indices in dict_users.items():
        raw_labels = dataset.targets[indices].numpy().tolist()

        if label_mappings and client_id in label_mappings:
            mapping = label_mappings[client_id]

            if isinstance(mapping, dict):
                # EuroSAT: global numeric label → local numeric label
                mapped_labels = [
                    mapping[int(y)]
                    for y in raw_labels
                ]
            else:
                # MNIST: callable mapping function
                mapped_labels = [
                    mapping(int(y))
                    for y in raw_labels
                ]
        else:
            mapped_labels = raw_labels

        counts = Counter(mapped_labels)
        total = len(mapped_labels)

        print(f"\nClient {client_id} mapped labels (total={total}):")
        for cls in sorted(counts.keys()):
            print(f" class{cls}: {counts[cls]}")

# The function below is FedAvg equivalent
def average_weights(w, weights=None):
    '''
    Return the average of the weights.
    :return: average_weights
    '''

    if len(w) ==0:
        raise ValueError("No clients weight available")

    if weights is None:
        weights = [1.0] * len(w)

    if len(weights) != len(w):
        raise ValueError(
            f"Number of weights ({len(weights)}) must match number of client models ({len(w)})."
        )

    total_weight = float(sum(weights))

    if total_weight <= 0:
        raise ValueError("Total averaging weight must be positive.")

    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        if torch.is_floating_point(w_avg[key]):
            w_avg[key] = w_avg[key] * (weights[0] / total_weight)

            for i in range(1,len(w)):
                w_avg[key] += w[i][key] * (weights[i]/total_weight)

        else:
            w_avg[key] = copy.deepcopy(w[0][key])
    return w_avg

def calculate_encoder_update(local_weights, reference_weights):
    """
    Calculate the client's encoder parameter update:

        update = local encoder - reference encoder
    """
    if local_weights.keys() != reference_weights.keys():
        raise ValueError("Local and reference encoder state_dict keys differ.")

    update = {}

    for key in reference_weights:
        # Skip non-floating-point state such as num_batches_tracked.
        if not torch.is_floating_point(local_weights[key]):
            continue

        update[key] = (
            local_weights[key].detach().cpu().float()
            - reference_weights[key].detach().cpu().float()
        )

    return update


def flatten_encoder_update(update):
    """Flatten an encoder update into one vector."""

    vectors = [
        update[key].reshape(-1)
        for key in sorted(update.keys())
    ]

    if not vectors:
        raise ValueError("Encoder update contains no floating-point parameters.")

    return torch.cat(vectors)


def calculate_cosine_similarity_matrix(client_updates, eps=1e-12):
    """
    client_updates:
        {client_id: encoder_update_state_dict}
    """
    client_ids = sorted(client_updates.keys())

    vectors = []

    for client_id in client_ids:
        vector = flatten_encoder_update(client_updates[client_id])

        if torch.linalg.vector_norm(vector) < eps:
            raise ValueError(
                f"Client {client_id} has an approximately zero update."
            )

        vectors.append(vector)

    update_matrix = torch.stack(vectors)

    normalized_updates = F.normalize(
        update_matrix,
        p=2,
        dim=1,
        eps=eps,
    )

    similarity_matrix = normalized_updates @ normalized_updates.T
    similarity_matrix = similarity_matrix.clamp(-1.0, 1.0)

    return client_ids, similarity_matrix


def cluster_clients_by_similarity(
    client_updates,
    similarity_threshold=0.80,
):
    """
    Form clusters using average-linkage agglomerative clustering.
    """
    client_ids, similarity_matrix = (
        calculate_cosine_similarity_matrix(client_updates)
    )

    distance_matrix = 1.0 - similarity_matrix
    distance_matrix = distance_matrix.clamp(min=0.0).numpy()
    np.fill_diagonal(distance_matrix, 0.0)

    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=1.0 - similarity_threshold,
        metric="precomputed",
        linkage="average",
    )

    cluster_labels = clustering.fit_predict(distance_matrix)

    clusters = {}

    for client_id, cluster_label in zip(client_ids, cluster_labels):
        clusters.setdefault(int(cluster_label), []).append(client_id)

    return clusters, similarity_matrix

def exp_details(args):
    print('\nExperimental Details:')
    print(f'    Model   : {args.model}')
    print(f'    Optimizer   : {args.optimizer}')
    print(f'    Learning    : {args.lr}')
    print(f'    Global Rounds   : {args.epochs}\n')
    print('     Federated parameters:')
    # if args.iid:
    #     print('     IID')
    #
    # else:
    #     print('     Non-IID')

    print(f'     Partition Type     : {args.partition_type}')
    print(f'    Fraction of users  : {args.frac}')
    print(f'    Local Batch Size    : {args.local_bs}')
    print(f'    Local Epochs    : {args.local_ep}\n')

    return

