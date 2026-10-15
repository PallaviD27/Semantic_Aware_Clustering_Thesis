# Local training of the clients happen in this

import torch
from torch import nn  # nn is for neural neural network modules like loss functions and layers.
# Import Dataloader and Dataset to handle the data batching and custom dataset structures.
from torch.utils.data import DataLoader, Dataset
from channel import awgn_channel_normalize, awgn_channel_no_normalize
from channel import rms_power_normalize_signal
# Removed DatasetSplit class
from channel import awgn_agg_normalize, awgn_agg_no_normalize
from channel import rms_power_normalize_weight
from channel import gradient_norm_regularizer
from channel import gradient_norm_regularizersq
import math

# Local training on each client; each client train privately
class LocalUpdate(object):
    def __init__(self, args, dataset, idxs, logger, encoder=None, head=None, client_id=None, label_mappings=None):
        self.args = args
        self.logger = logger
        self.device = 'cuda' if torch.cuda.is_available() and (not self.args.cpu) else 'cpu'
        if self.device == 'cuda':
            torch.cuda.set_device(self.args.gpu)

        self.client_id = client_id
        base_seed = int(getattr(self.args, "seed", 0) or 0)
        cid = int(self.client_id) if self.client_id is not None else 0
        g = torch.Generator()  # CPU generator (what DataLoader expects)
        g.manual_seed(base_seed + cid)  # unique but repeatable per client
        # self.trainloader, self.validloader, self.testloader = self.train_val_test(dataset, list(idxs))
        self.trainloader = DataLoader(Subset(dataset, list(idxs)), batch_size=self.args.local_bs, shuffle=True, generator=g)



        if args.model == 'CNN':
            # Default criterion set to NLL loss function
            self.criterion = nn.CrossEntropyLoss().to(self.device)

        elif args.model == 'AutoEncoder':
            # When AutoEncoder structure in models, MSE loss to be used
            self.criterion = nn.CrossEntropyLoss().to(self.device)

        elif args.model == 'MLP':
            self.encoder = encoder.to(self.device)
            self.head = head.to(self.device)
            self.label_mappings = label_mappings

            # Combine encoder and head into one model for optimization
            self.model = nn.Sequential(self.encoder, self.head)
            self.criterion = nn.CrossEntropyLoss().to(self.device)  # or CrossEntropyLoss() if output isn't log_softmax
            self.optimizer = None # Optimizer not being built here but where it is needed i.e. update_weights

    #  Function to change learning rate
    def get_lr_for_round(self, global_round):
        """
        Learning-rate schedule based on communication round.

        Assumption:
        global_round starts from 1.

        CR 1  - 15  : 0.01
        CR 16 - 25  : 0.001
        CR 26 - 30  : 0.0001
        """
        # Try 1
        # if global_round <= 10:
        #     return 0.01
        # elif global_round <= 20:
        #     return 0.001
        # else:
        #     return 0.0001
        # Try 2
        # if global_round <= 20:
        #     return 0.01
        # elif global_round <= 25:
        #     return 0.001
        # else:
        #     return 0.0001

        # Try 0
        if global_round <= 20:
            return 0.01
        elif global_round <= 25:
            return 0.01
        else:
            return 0.01

        # Try3

        # if global_round <= 15:
        #     return 0.01
        # elif global_round <= 24:
        #     return 0.005
        # else:
        #     return 0.001

        # # Try 4
        # if global_round <= 10:
        #     return 0.01
        # elif global_round <= 20:
        #     return 0.005
        # else:
        #     return 0.0025

        # # Try 5
        # if global_round <= 5:
        #     return 0.005
        # elif global_round <= 15:
        #     return 0.0025
        # else:
        #     return 0.001
        # # Try 6
        # if global_round <= 5:
        #     return 0.01
        # elif global_round <= 10:
        #     return 0.005
        # elif global_round <= 15:
        #     return 0.0025
        # elif global_round <=20:
        #     return 0.002
        # elif global_round <=25:
        #     return 0.001
        # else:
        #     return 0.0001

    # def get_lr_for_round(self, global_round, total_rounds=30):
    #     initial_lr = 0.01
    #     min_lr = 0.002
    #
    #     return min_lr + 0.5 * (initial_lr - min_lr) * (
    #             1 + math.cos(math.pi * (global_round - 1) / (total_rounds - 1))
    #     )


# Removed def train_val_test

    def update_weights(self, global_round):
        self.encoder.to(self.device)
        self.head.to(self.device)
        self.encoder.train()
        self.head.train()
        # self.model.train()
        # Commented to introduce lr for different rounds
        # if self.optimizer is None:
        #     self.optimizer = torch.optim.SGD(list(self.encoder.parameters()) + list(self.head.parameters()),
        #                                  lr=self.args.lr, momentum=self.args.momentum)

        current_lr = self.get_lr_for_round(global_round)

        if self.optimizer is None:
            self.optimizer = torch.optim.SGD(
                list(self.encoder.parameters()) + list(self.head.parameters()),
                lr=current_lr,
                momentum=self.args.momentum
            )
            # One-time sanity check; prints once per training call (Felt cute, might delete later)
            ids_opt = {id(p) for g in self.optimizer.param_groups for p in g['params']}
            ids_mod = {id(p) for p in self.encoder.parameters()} | {id(p) for p in self.head.parameters()}
            print(f"[Client {self.client_id}] Optimizer has all params:", ids_mod <= ids_opt)
            # One-time sanity check (Felt cute, might delete later)

        else:
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = current_lr

        print(f"[Client {self.client_id}] CR {global_round} | Learning rate = {current_lr}")



        loss_sum, count = 0.0, 0
        # epoch_loss = []

        print(f"--> Starting training for Client {self.client_id}")

        for epoch in range(self.args.local_ep):
            print(f"    Epoch {epoch + 1}/{self.args.local_ep}")
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                print(f"        Batch {batch_idx + 1}")
                images, labels = images.to(self.device), labels.to(self.device)
                labels = labels.long() #casting dtype

                # Hard device consistency checks (fail fast)
                assert images.device == next(self.encoder.parameters()).device, (
                    f"[Client {self.client_id}] Device mismatch: images={images.device} "
                    f"encoder={next(self.encoder.parameters()).device}"
                )
                assert images.device == next(self.head.parameters()).device, (
                    f"[Client {self.client_id}] Device mismatch: images={images.device} "
                    f"head={next(self.head.parameters()).device}"
                )

                # Sanity check for GPU training
                if batch_idx == 0 and epoch == 0:
                    print(
                        f"[Client {self.client_id}] "
                        f"images={images.device}, "
                        f"encoder={next(self.encoder.parameters()).device}, "
                        f"head={next(self.head.parameters()).device}"
                    )

                # Apply label mapping if applicable i.e. if the client has one
                if self.label_mappings is not None and self.client_id in self.label_mappings:
                    map_fn = self.label_mappings[self.client_id]
                    labels = torch.tensor([map_fn(y.item()) for y in labels],
                        device=self.device, dtype=torch.long)

                # Checking bounds before loss; assumes the head's last Linear is in a lit/Sequential like head.fc[-1]
                if hasattr(self.head, "fc"):
                    out_dim = self.head.fc[-1].out_features
                elif isinstance(self.head, torch.nn.Sequential):
                    out_dim = list(self.head.children())[-1].out_features
                else:
                    # generic fallback: last Linear in the head
                    last_linear = [m for m in self.head.modules() if isinstance(m, torch.nn.Linear)][-1]
                    out_dim = last_linear.out_features

                if labels.min() < 0 or labels.max() >= out_dim:
                    raise ValueError(
                        f"[Train][Client {self.client_id}] Mapped labels out of range: "
                        f"min={labels.min().item()} max={labels.max().item()} vs head_out={out_dim}"
                    )

                self.optimizer.zero_grad()
                # outputs = self.head(self.encoder(images))
                # With AWGN noise
                x = self.encoder(images)
                power_before = x.pow(2).mean(dim=1).mean().item()
                print(power_before)
                if self.args.signal_normalize:
                    x = rms_power_normalize_signal(x) # Normalization of latent vector to unit power
                    power_after = x.pow(2).mean(dim=1).mean().item()
                    print(power_after)
                    x_hat=awgn_channel_normalize(x,sigma_signal=self.args.sigma_signal, snr_db_signal=self.args.snr_db_signal)

                else:
                    x_hat = awgn_channel_no_normalize(x, sigma_signal=self.args.sigma_signal, snr_db_signal=self.args.snr_db_signal)
                if batch_idx == 0 and epoch == 0:
                    print(
                        f"[Client {self.client_id}] x std={x.std().item():.4f} | noise std={(x_hat - x).std().item():.4f}")

                outputs = self.head(x_hat)
                task_loss = self.criterion(outputs, labels)  # normal CrossEntropy task loss
                # Added for noise regularization
                if getattr(self.args, "use_noise_regularizer", False):

                    # FedSem-style coefficient:
                    # uplink model/update noise variance + downlink model/update noise variance
                    noise_reg_coeff = 4 * 2 * (self.args.sigma_agg**2)

                    # In the setup, the transmitted/aggregated semantic transmitter is the encoder
                    grad_norm = gradient_norm_regularizer(
                        task_loss,
                        self.encoder.parameters()
                    )

                    loss = task_loss + noise_reg_coeff * grad_norm

                    print(f"task_loss: {task_loss.item():.6f}")
                    print(f"grad_norm: {grad_norm.item():.6f}")
                    print(f"reg_term: {(noise_reg_coeff * grad_norm).item():.6f}")
                    print(f"total_loss: {loss.item():.6f}")

                else:
                    loss = task_loss
                # loss = self.criterion(outputs, labels) # FL Loss - Client's batch loss; Cross-Entropy
                loss.backward()
                self.optimizer.step()

                # Convert batch mean loss to sum loss and then accumulate
                bs = labels.size(0)
                loss_sum += loss.item() * bs
                count += bs

        # (this is optional but we are doing it here) switch back to eval when done
        self.encoder.eval()
        self.head.eval()

        # per sample avg loss across all local updates this round
        avg_loss = loss_sum/count if count > 0 else 0.0


        #  Step xx FL - Return only encoder weights if MLP, else full model ( used in Fed Avg)
        if self.args.model == 'MLP':
            return self.encoder.state_dict(), avg_loss
        else:
            return self.model.state_dict(), avg_loss

    # Train Accuracy on client's local training data
    def inference(self):
        self.encoder.eval()
        self.head.eval()

        loss_sum, count = 0.0, 0
        total, correct = 0, 0

        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels.to(self.device)
                labels = labels.long() # Specifiying dtype explicitly

                # Apply label mapping once
                if self.label_mappings is not None and self.client_id in self.label_mappings:
                    map_func = self.label_mappings[self.client_id]
                    labels = torch.tensor([map_func(label.item()) for label in labels],
                                          device=self.device, dtype=torch.long)


                # Optional sanity check (same logic as in update_weights)
                if hasattr(self.head, "fc"):
                    out_dim = self.head.fc[-1].out_features
                elif isinstance(self.head, torch.nn.Sequential):
                    out_dim = list(self.head.children())[-1].out_features
                else:
                    last_linear = [m for m in self.head.modules() if isinstance(m, torch.nn.Linear)][-1]
                    out_dim = last_linear.out_features

                if labels.min() < 0 or labels.max() >= out_dim:
                    raise ValueError(f"[Infer][Client {self.client_id}] Mapped labels out of range "
                                     f"(min={labels.min().item()}, max={labels.max().item()}, out_dim={out_dim})")


                x = self.encoder(images)
                power_before = x.pow(2).mean(dim=1).mean().item()
                if self.args.signal_normalize:
                    x = rms_power_normalize_signal(x)  # Normalization of latent vector to unit power
                    power_after = x.pow(2).mean(dim=1).mean().item()
                    print(power_after)
                    x_hat = awgn_channel_normalize(x, sigma_signal=self.args.sigma_signal, snr_db_signal=self.args.snr_db_signal)
                else:
                    x_hat = awgn_channel_no_normalize(x, sigma_signal=self.args.sigma_signal, snr_db_signal=self.args.snr_db_signal)

                outputs = self.head(x_hat)
                batch_loss = self.criterion(outputs, labels)  # mean over batch
                loss_sum += batch_loss.item() * labels.size(0)  # accumulate by sample
                count += labels.size(0)


                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        accuracy = correct / total if total > 0 else 0.0 # Client accuracy based on correct predictions from total predictions
        avg_loss = loss_sum /count if count > 0 else 0.0
        return accuracy, avg_loss


from torch.utils.data import DataLoader, Subset
import torch.nn as nn

def test_inference_multitask(
    args, encoder, heads, test_dataset, user_groups,
    client_test_accuracies=None, client_test_losses=None,
    label_mappings=None
):
    """
    Returns:
        avg_acc (overall %), avg_loss (overall per-sample),
        client_test_accuracies: dict[int, list[float]]  # % per client per round
        client_test_losses:     dict[int, list[float]]  # per-sample loss per client per round
    """
    # device = 'cuda' if (torch.cuda.is_available() and args.gpu) else 'cpu'
    # device = 'cuda' if torch.cuda.is_available() and args.gpu is not None else 'cpu'
    device = 'cuda' if torch.cuda.is_available() and (not args.cpu) else 'cpu'
    if device == 'cuda':
        torch.cuda.set_device(args.gpu)

    # init dicts if needed
    if client_test_accuracies is None:
        client_test_accuracies = {}
    if client_test_losses is None:
        client_test_losses = {}

    # convenience: last Linear out_features of a head
    def head_out_dim(h):
        if hasattr(h, "fc"):
            return h.fc[-1].out_features
        elif isinstance(h, torch.nn.Sequential):
            return list(h.children())[-1].out_features
        else:
            last_linear = [m for m in h.modules() if isinstance(m, torch.nn.Linear)][-1]
            return last_linear.out_features

    total_correct, total_samples = 0, 0
    total_loss_sum = 0.0

    # ensure encoder is on correct device (heads are moved when wrapped into model)
    encoder = encoder.to(device)

    # Specific to scenario 2
    for client_id in heads.keys():
        head = heads[client_id]
        model = nn.Sequential(encoder, head).to(device)
        model.eval()

        test_loader = DataLoader(
            Subset(test_dataset, list(user_groups[client_id])),
            batch_size=64, shuffle=False
        )
        criterion = nn.CrossEntropyLoss().to(device)

        correct, samples = 0, 0
        loss_sum = 0.0  # accumulate **per-sample** loss
        out_dim = head_out_dim(head)

        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device).long()   # CE needs Long target

                # client-specific label remap (if provided)
                if label_mappings and client_id in label_mappings:
                    map_fn = label_mappings[client_id]
                    mapped = [map_fn(int(y.item())) for y in labels]
                    labels = torch.tensor(mapped, dtype=torch.long, device=device)

                # bound check once (avoids IndexError in CE)
                if labels.numel() > 0 and (labels.min() < 0 or labels.max() >= out_dim):
                    raise ValueError(
                        f"[Client {client_id}] Mapped label(s) out of range for head with {out_dim} outputs. "
                        f"min={labels.min().item()}, max={labels.max().item()}"
                    )

                # With noise (toggle)
                x = encoder(images)
                power_before = x.pow(2).mean(dim=1).mean().item()
                if args.signal_normalize:
                    x = rms_power_normalize_signal(x)  # Normalization of latent vector to unit power
                    power_after = x.pow(2).mean(dim=1).mean().item()
                    print(power_after)
                    x_hat=awgn_channel_normalize(x, sigma_signal=args.sigma_signal, snr_db_signal=args.snr_db_signal)

                else:
                    x_hat = awgn_channel_no_normalize(x, sigma_signal=args.sigma_signal,snr_db_signal=args.snr_db_signal)
                outputs = head(x_hat)
                head = head.to(device)

                # loss returned is mean over the batch -> convert to sum then to per-sample
                bs = labels.size(0)
                batch_loss = criterion(outputs, labels)   # mean over batch
                loss_sum += batch_loss.item() * bs
                samples += bs

                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()

        # per-client stats for this round
        acc_client = 100.0 * correct / samples if samples > 0 else 0.0
        loss_client = loss_sum / samples if samples > 0 else 0.0

        # stash into dicts
        if client_id not in client_test_accuracies:
            client_test_accuracies[client_id] = []
            client_test_losses[client_id] = []
        client_test_accuracies[client_id].append(acc_client)
        client_test_losses[client_id].append(loss_client)

        # accumulate overall
        total_correct += correct
        total_loss_sum += loss_sum
        total_samples += samples

    avg_acc = 100.0 * total_correct / total_samples if total_samples > 0 else 0.0
    avg_loss = total_loss_sum / total_samples if total_samples > 0 else 0.0
    return avg_acc, avg_loss, client_test_accuracies, client_test_losses
