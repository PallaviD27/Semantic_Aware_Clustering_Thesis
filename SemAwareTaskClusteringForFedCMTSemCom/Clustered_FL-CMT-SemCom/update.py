# Local training of the clients happen in this

import torch
from torch import nn  # nn is for neural neural network modules like loss functions and layers.
# Import Dataloader and Dataset to handle the data batching and custom dataset structures.
from torch.utils.data import DataLoader, Dataset, Subset
import torch.optim as optim
from channel import awgn_channel_normalize, awgn_channel_no_normalize, rms_power_normalize_signal
from channel import awgn_agg_normalize, awgn_agg_no_normalize, rms_power_normalize_weight
from channel import gradient_norm_regularizersq, gradient_norm_regularizer
from quantization import quantize_features, _quantizer_step, dequantize_features, index_to_bits, bits_to_index
from channel import los_channel_model, los_channel_model_sq, los_feature_channel, los_symbol_channel
from modulation import modulation, demodulation
# Removed DatasetSplit Class

import csv
from pathlib import Path


@torch.no_grad()
def log_quantization_stats(
    path, x, x_quantized, *,
    client_id, global_round, epoch, batch_idx,
    bits=8,lower_bound=-4.0, upper_bound=4.0,
):
    """x is the feature tensor immediately before quantization."""
    x = x.detach().float()
    x_quantized = x_quantized.detach().float()

    # clipped = (x < 0.0) | (x > upper_bound)
    clipped = (x < lower_bound) | (x > upper_bound)
    error = x_quantized - x

    row = {
        "round": global_round,
        "client": client_id,
        "epoch": epoch + 1,
        "batch": batch_idx + 1,
        "bits": bits,
        "upper_bound": upper_bound,
        "elements": x.numel(),
        "minimum": x.min().item(),
        "maximum": x.max().item(),
        "clipped_elements": clipped.sum().item(),
        "squared_error_sum": error.square().sum().item(),
        "power_before_sum": x.square().sum().item(),
        "power_after_sum": x_quantized.square().sum().item(),
    }

    path = Path(path)
    write_header = not path.exists() or path.stat().st_size == 0

    with path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=row.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(row)


@torch.no_grad()
def collect_class_features(
    args, encoder, dataset, idxs, client_id,
    label_mappings, num_classes,
):
    if args.dataset != "EuroSAT":
        raise ValueError("This function currently supports EuroSAT.")

    device = next(encoder.parameters()).device
    mapping = label_mappings[client_id]

    loader = DataLoader(
        Subset(dataset, sorted(int(i) for i in idxs)),
        batch_size=args.local_bs,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        generator=torch.Generator().manual_seed(
            int(args.seed) + int(client_id)
        ),
    )

    feature_batches = []
    label_batches = []
    raw_feature_batches = []
    mu_batches = []
    ln_var_batches = []
    was_training = encoder.training
    encoder.eval()

    try:
        for images, labels in loader:
            # x = encoder(images.to(device))
            # Reparameterization
            mu, ln_var = encoder(images.to(device))
            x = reparameterize(mu, ln_var)

            raw_feature_batches.append(x.detach().cpu())
            mu_batches.append(mu.detach().cpu())
            ln_var_batches.append(ln_var.detach().cpu())

            if args.signal_normalize:
                x = rms_power_normalize_signal(x)

            # Stop BEFORE quantization/modulation/channel.
            if x.ndim != 2 or not torch.isfinite(x).all():
                raise ValueError("Expected finite [batch, features] output.")

            mapped = [
                mapping(int(y)) if callable(mapping) else mapping[int(y)]
                for y in labels.detach().cpu().tolist()
            ]
            mapped = torch.tensor(mapped, dtype=torch.long)

            if (mapped < 0).any() or (mapped >= num_classes).any():
                raise ValueError("Mapped labels outside the task-class range.")

            feature_batches.append(x.detach().cpu())
            label_batches.append(mapped)

    finally:
        encoder.train(was_training)

    if not feature_batches:
        raise ValueError(f"Client {client_id} has no calibration samples.")

    return {
        "features": torch.cat(feature_batches, dim=0),
        "features_before_normalization": torch.cat(
            raw_feature_batches, dim=0
        ),
        "mu": torch.cat(mu_batches, dim=0),
        "ln_var": torch.cat(ln_var_batches, dim=0),
        "labels": torch.cat(label_batches, dim=0),
        "client_id": int(client_id),
        "num_classes": int(num_classes),
        "signal_normalize": bool(args.signal_normalize),
        "stage": "before_quantization",
        "sampling": "one sample per image",
    }
def reparameterize(mu, ln_var):
    std = torch.exp(0.5 * ln_var)
    eps = torch.randn_like(std)
    return mu + std * eps

# Local training on each client; each client train privately
class LocalUpdate(object):
    def __init__(self, args, dataset, idxs, logger, encoder=None, head=None, client_id=None, label_mappings=None):
        self.args = args
        self.logger = logger
        # self.device = 'cuda' if args.gpu else 'cpu'

        self.device = 'cuda' if torch.cuda.is_available() and (not self.args.cpu) else 'cpu'
        if self.device == 'cuda':
            torch.cuda.set_device(self.args.gpu)

        self.client_id = client_id
        self.label_mappings=label_mappings
        base_seed = int(getattr(self.args, "seed", 0) or 0)
        cid = int(self.client_id) if self.client_id is not None else 0
        g = torch.Generator()  # CPU generator (what DataLoader expects)
        g.manual_seed(base_seed + cid)  # unique but repeatable per client
        # self.trainloader, self.validloader, self.testloader = self.train_val_test(dataset, list(idxs))
        self.trainloader = DataLoader(Subset(dataset, list(idxs)), batch_size=self.args.local_bs, shuffle=True, generator=g)



        if args.model == 'CNN':
            # Default criter    ion set to NLL loss function
            self.criterion = nn.CrossEntropyLoss().to(self.device)

        elif args.model == 'AutoEncoder':
            # When AutoEncoder structure in models, MSE loss to be used
            self.criterion = nn.CrossEntropyLoss().to(self.device)

        elif args.model == 'MLP':
            self.encoder = encoder.to(self.device)
            self.head = head.to(self.device)
            self.label_mappings = label_mappings

            # Combine encoder and head into one model for optimization
            # Below line has been uncommented coz it is uncommented in Clustered code too
            self.model = nn.Sequential(self.encoder, self.head)
            if self.args.dataset == 'BENet':
                self.criterion = nn.BCEWithLogitsLoss().to(self.device)
                self.task_indices = self.args.client_task_indices[self.client_id]
                self.task_indices =torch.tensor(
                    self.task_indices,
                    dtype=torch.long,
                    device=self.device
                )

            else:
                self.criterion = nn.CrossEntropyLoss().to(self.device)  # or CrossEntropyLoss() if output isn't log_softmax
                self.task_indices = None
            self.optimizer = None # Optimizer not being built here but where it is needed i.e. update_weights

    def get_lr_for_round(self, global_round):
        """Use the configured constant learning rate for controlled comparisons."""
        return float(self.args.lr)

    def noise_regularizer_coefficient(self, global_round):
        """Tune the penalty independently of the actual channel noise."""
        if not getattr(self.args, "use_noise_regularizer", False):
            return 0.0
        sigma = self.args.sigma_agg
        if sigma is None or sigma < 0 or self.args.snr_db_agg is not None:
            raise ValueError("Noise regularization requires fixed nonnegative sigma_agg.")
        active_links = int(self.args.uplink) + int(self.args.downlink)
        # Preserve the legacy coefficient with both links enabled.
        coefficient = getattr(self.args, "noise_reg_strength", 8.0) * sigma**2 * active_links / 2
        warmup = getattr(self.args, "noise_reg_warmup_rounds", 0)
        if warmup > 0:
            coefficient *= min(max(global_round, 0) / warmup, 1.0)
        return coefficient

    def update_weights(self, global_round):
        self.encoder.to(self.device)
        self.head.to(self.device)
        self.encoder.train()
        self.head.train()
        # self.model.train()
        # Commented to introduce different LR
        # if self.optimizer is None:
        #     self.optimizer = torch.optim.SGD(list(self.encoder.parameters()) + list(self.head.parameters()),
        #                                  lr=self.args.lr, momentum=self.args.momentum)

        current_lr = self.get_lr_for_round(global_round)

        if self.optimizer is None:
            parameters = list(self.encoder.parameters()) + list(self.head.parameters())
            if self.args.optimizer == "sgd":
                self.optimizer = torch.optim.SGD(
                    parameters, lr=current_lr, momentum=self.args.momentum
                )
            elif self.args.optimizer == "Adam":
                self.optimizer = torch.optim.Adam(parameters, lr=current_lr)
            else:
                raise ValueError(f"Unsupported optimizer: {self.args.optimizer}")

            # One-time sanity check; prints once per training call (Felt cute, might delete later)
            ids_opt = {id(p) for g in self.optimizer.param_groups for p in g['params']}
            ids_mod = {id(p) for p in self.encoder.parameters()} | {id(p) for p in self.head.parameters()}
            print(f"[Client {self.client_id}] Optimizer has all params:", ids_mod <= ids_opt)
            # One-time sanity check (Felt cute, might delete later)

        else:
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = current_lr

        print(f"[Client {self.client_id}] CR {global_round} | Learning rate = {current_lr}")


        noise_reg_coeff = self.noise_regularizer_coefficient(global_round)
        epoch_loss = []

        loss_sum, count = 0.0, 0

        print(f"--> Starting training for Client {self.client_id}")

        for epoch in range(self.args.local_ep):
            print(f"    Epoch {epoch + 1}/{self.args.local_ep}")
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                if batch_idx == 0:
                    print(f"        Batch {batch_idx + 1}")
                # images, labels = images.to(self.device), labels.to(self.device)
                # labels = labels.long() #casting dtype
                #
                # # Apply label mapping if applicable i.e. if the client has one
                # if self.label_mappings is not None and self.client_id in self.label_mappings:
                #     map_fn = self.label_mappings[self.client_id]
                #     labels = torch.tensor([map_fn(y.item()) for y in labels],
                #         device=self.device, dtype=torch.long)

                images =images.to(self.device)
                labels = labels.to(self.device)

                if self.args.dataset == "BENet":
                    labels = labels.float()
                    labels = labels[:,self.task_indices]

                elif self.args.dataset == "EuroSAT":
                    mapping = self.label_mappings[self.client_id]

                    labels = torch.tensor(
                        [
                            mapping[int(label)]
                            for label in labels.tolist()
                        ],
                        dtype=torch.long,
                        device=self.device,
                    )

                else:
                    labels = labels.long()

                    if self.label_mappings is not None and self.client_id in self.label_mappings:
                        map_fn = self.label_mappings[self.client_id]
                        labels = torch.tensor([map_fn(int(label)) for label in labels],
                                              device=self.device,
                                              dtype=torch.long)


                # Checking bounds before loss; assumes the head's last Linear is in a lit/Sequential like head.fc[-1]

                if self.args.dataset == "BENet":
                    expected_out_dim = self.head.fc[-1].out_features

                    if labels.shape[-1] != expected_out_dim:
                        raise ValueError(
            f"[Train][Client {self.client_id}] Label shape mismatch: "
            f"labels.shape={labels.shape}, head_out={expected_out_dim}"
        )
                else:
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
                # Without noise
                # outputs = self.head(self.encoder(images))
                # loss = self.criterion(outputs, labels) # FL Loss - Client's batch loss; Cross-Entropy

                # With AWGN noise
                # x = self.encoder(images)

                # With Reparameterization
                mu, ln_var = self.encoder(images)
                x = reparameterize(mu, ln_var)

# -----------Inspecting features for quantization ------------
                if epoch ==0 and batch_idx == 0:
                    z = x.detach()

                    print(f"\n[Client {self.client_id}] Raw encoder features")
                    print(f"Shape: {tuple(z.shape)}")
                    print(f"Minimum: {z.min().item():.6f}")
                    print(f"Maximum: {z.max().item():.6f}")
                    print(f"Mean: {z.mean().item():.6f}")
                    print(f"Std: {z.std(unbiased=False).item():.6f}")

# ----------------------------------------------------------------

                power_before = x.pow(2).mean(dim=1).mean().item()
                if batch_idx == 0:
                    print(f"Latent power before channel: {power_before:.6f}")
# ------------------------First normalizing--------------------------------
                if self.args.signal_normalize:
                    x = rms_power_normalize_signal(x)  # Normalization of latent vector to unit power

# --------------------------------Then quantizing----------------------------------
                #x_quantized, indices = quantize_features(x, bits=8, upper_bound=4.0, use_ste=True)
                indices = quantize_features(x, bits=8,lower_bound=-4.0,upper_bound=4)
                # Reconstruction only for logs
                x_quantized = dequantize_features(indices, bits=8,lower_bound=-4.0,upper_bound=4,dtype=x.dtype)
                log_quantization_stats(
                    self.args.quant_stats_path,
                    x,
                    x_quantized,
                    client_id=self.client_id,
                    global_round=global_round,
                    epoch=epoch,
                    batch_idx=batch_idx,
                    bits=8,
                    upper_bound=4.0,
                )
# ---------------------Then converting quantized indices to Binary-----------------
                x_bin = index_to_bits(indices, bits=8)

# ---------------------Then converting binary to BPSK symbols--------------------
                x_sym = modulation(x_bin)

# -------------------------Channel---------------------------------
                h = float(los_channel_model(
                    Gt=self.args.Gt,
                    Gr=self.args.Gr,
                    wavelength_m=self.args.wavelength_m,
                    d_m=self.args.d_m,
                    L=1.0,
                ))

                sigma_rx = self.args.sigma_signal
                if sigma_rx is None or sigma_rx <= 0:
                    raise ValueError("Set --sigma_signal to a positive value.")

                # Target SNR = 0 dB = 1 in linear units.
                tx_power = sigma_rx ** 2 / h ** 2

                x_hat, y = los_symbol_channel(
                    x_sym,
                    h=h,
                    tx_power=tx_power,
                    noise_std_rx=sigma_rx,
                )

                # if batch_idx == 0:
                with torch.no_grad():
                    # clipped_fraction = (
                    # (x < 0.0) | (x > 4.0)
                    # ).float().mean().item()
                    clipped_fraction = (
                            (x < -4.0) | (x > 4.0)
                    ).float().mean().item()

                    quantization_mse = (
                        x_quantized - x
                    ).square().mean().item()

                    print(
                        f"[Client {self.client_id}] "
                        f"Before quantization: min={x.min().item():.4f}, "
                        f"max={x.max().item():.4f}, "
                        f"power={x.square().mean().item():.6f} | "
                        f"After quantization: "
                        f"power={x_quantized.square().mean().item():.6f} | "
                        f"clipped={100 * clipped_fraction:.3f}% | "
                        f"quantization MSE={quantization_mse:.8f}"
                    )
                # if self.args.signal_normalize:
                #     x_hat = awgn_channel_normalize(x_quantized, sigma_signal=self.args.sigma_signal,snr_db_signal=self.args.snr_db_signal,)
                #
                # else:
                #     x_hat = awgn_channel_no_normalize(x_quantized, sigma_signal=self.args.sigma_signal,snr_db_signal=self.args.snr_db_signal,)

                #     if batch_idx == 0:
                #         print(f"Normalized latent power: {power_after:.6f}")
                #     x_hat = awgn_channel_normalize(x, sigma_signal=self.args.sigma_signal, snr_db_signal=self.args.snr_db_signal)
                # else:
                #     x_hat = awgn_channel_no_normalize(x, sigma_signal=self.args.sigma_signal, snr_db_signal=self.args.snr_db_signal)



                # if batch_idx == 0 and epoch == 0: # Old noise diagnostic
                #     print(
                #         f"[Client {self.client_id}] x std={x.std().item():.4f} | noise std={(x_hat - x_quantized).std().item():.4f}")

# ---------------------Demodulation after channel-----------------
                bits_received = demodulation(x_hat)
# ----------------------- Bits to index conversion------------------
                received_indices = bits_to_index(bits_received, bits=8)
# ----------------------Dequantize for reconstruction-------------------
                y_reconstructed = dequantize_features(received_indices, bits=8,lower_bound=-4.0, upper_bound=4, dtype=x.dtype,
                                                      ste_reference=x)

#-------------------------------------Diagnostics------------------------------------
                if batch_idx == 0 and epoch == 0:
                    with torch.no_grad():
                        ber = (bits_received != x_bin).float().mean().item()

                        communication_mse = (
                                y_reconstructed - x_quantized
                        ).square().mean().item()

                        print(
                            f"[Client {self.client_id}] "
                            f"BER={ber:.4f} | "
                            f"Communication feature MSE={communication_mse:.8f}"
                        )
#------------------------------------------------------------------------------------

                outputs = self.head(y_reconstructed)
                task_loss = self.criterion(outputs, labels)
                if noise_reg_coeff > 0:
                    regularizer = (
                        gradient_norm_regularizersq
                        if getattr(self.args, "noise_reg_norm", "l2") == "squared"
                        else gradient_norm_regularizer
                    )
                    grad_norm = regularizer(task_loss, self.encoder.parameters())
                    loss = task_loss + noise_reg_coeff * grad_norm
                    if batch_idx == 0:
                        print(f"task_loss={task_loss.item():.6f} "
                              f"reg_term={(noise_reg_coeff * grad_norm).item():.6f} "
                              f"coefficient={noise_reg_coeff:.6f}")
                else:
                    loss = task_loss

                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        f"Non-finite loss: client={self.client_id}, "
                        f"round={global_round}, epoch={epoch + 1}, batch={batch_idx + 1}"
                    )
                loss.backward()
                max_grad_norm = getattr(self.args, "max_grad_norm", 0.0)
                if max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        list(self.encoder.parameters()) + list(self.head.parameters()),
                        max_grad_norm, error_if_nonfinite=True,
                    )
                self.optimizer.step()

                # Corrected Loss calculation

                bs = labels.size(0)  # batch size
                loss_sum += loss.item() * bs  # accumulate total loss
                count += bs  # accumulate total samples


                # batch_loss.append(loss.item()) #  FL Loss - Storing it for "this" batch

            # epoch_loss.append(sum(batch_loss) / len(batch_loss)) #  FL Loss - epoch loss is average of the batch losses in that epoch

        # avg_loss = sum(epoch_loss) / len(epoch_loss) # FL Loss - Average loss over local epochs = loss for a communication round

        # (this is optional but we are doing it here) switch back to eval when done
        self.encoder.eval()
        self.head.eval()

        # Corrected Loss calculation
        avg_loss = loss_sum / count if count > 0 else 0.0

        #  Step xx FL - Return only encoder weights if MLP, else full model ( used in Fed Avg)
        if self.args.model == 'MLP':
            return self.encoder.state_dict(), avg_loss
        else:
            return self.model.state_dict(), avg_loss

    def inference(self):
        self.encoder.eval()
        self.head.eval()
        # self.model.eval()

        loss_sum, count = 0.0, 0
        total, correct = 0, 0

        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                # images, labels = images.to(self.device), labels.to(self.device)
                # labels = labels.long() # Specifiying dtype explicitly
                images = images.to(self.device)
                labels = labels.to(self.device)

                if self.args.dataset == "BENet":
                    labels = labels.float()
                    labels = labels[:, self.task_indices]

                elif self.args.dataset == "EuroSAT":
                    mapping = self.label_mappings[self.client_id]

                    labels = torch.tensor(
                        [
                            mapping[int(label)]
                            for label in labels.detach().tolist()
                        ],
                        dtype=torch.long,
                        device=self.device,
                    )

                else:
                    labels = labels.long()

                    if self.label_mappings is not None and self.client_id in self.label_mappings:
                        map_func = self.label_mappings[self.client_id]
                        labels = torch.tensor(
                            [map_func(label.item()) for label in labels],
                            device=self.device,
                            dtype=torch.long
                        )

                # # Apply label mapping once
                # if self.label_mappings is not None and self.client_id in self.label_mappings:
                #     map_func = self.label_mappings[self.client_id]
                #     labels = torch.tensor([map_func(label.item()) for label in labels],
                #                           device=self.device, dtype=torch.long)

                if self.args.dataset == "BENet":
                    expected_out_dim = self.head.fc[-1].out_features

                    if labels.shape[-1] != expected_out_dim:
                        raise ValueError(
                            f"[Infer][Client {self.client_id}] Label shape mismatch: "
                            f"labels.shape={labels.shape}, head_out={expected_out_dim}"
                        )
                else:
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
                            f"[Infer][Client {self.client_id}] Mapped labels out of range: "
                            f"min={labels.min().item()} max={labels.max().item()} vs head_out={out_dim}"
                        )


                # x = self.encoder(images)
                # Reparameterization
                mu, ln_var = self.encoder(images)
                x = reparameterize(mu, ln_var)
# --------------------------------------Normalization-------------------------------------
                power_before = x.pow(2).mean(dim=1).mean().item()
                if batch_idx == 0:
                    print(f"Latent power before channel: {power_before:.6f}")
                if self.args.signal_normalize:
                    x = rms_power_normalize_signal(x)  # Normalization of latent vector to unit power
# ----------------------------------------------------------------------------------------
# --------------------------------------Quantization-------------------------------------
                indices = quantize_features(x, bits=8,lower_bound=-4.0, upper_bound=4.0)
# -----------------------------------------------------------------------------------------
# ------------------------Converting index to bits-----------------------------------------
                x_bin = index_to_bits(indices)
# -----------------------------------------------------------------------------------------
# -----------------------------Modulation--------------------------------------------------
                x_sym = modulation(x_bin)
# -----------------------------------------------------------------------------------------
# ----------------------------------Channel------------------------------------------------

                h = float(los_channel_model(
                    Gt=self.args.Gt,
                    Gr=self.args.Gr,
                    wavelength_m=self.args.wavelength_m,
                    d_m=self.args.d_m,
                    L=1.0,
                ))

                sigma_rx = self.args.sigma_signal
                if sigma_rx is None or sigma_rx <= 0:
                    raise ValueError("Set --sigma_signal to a positive value.")

                # Target SNR = 0 dB = 1 in linear units.
                tx_power = sigma_rx ** 2 / h ** 2

                x_hat, y = los_symbol_channel(
                    x_sym,
                    h=h,
                    tx_power=tx_power,
                    noise_std_rx=sigma_rx,
                )
# ---------------------------------------------------------------------------------
# -----------------------------Demodulation------------------------------------------
                bits_received = demodulation(x_hat)
# ------------------------------------------------------------------------------------
# ------------------------------Convert bits to indices---------------------------------
                received_indices = bits_to_index(bits_received, bits=8)
# ----------------------Dequantize for reconstruction-------------------
                y_reconstructed = dequantize_features(received_indices, bits=8,lower_bound=-4.0, upper_bound=4, dtype=x.dtype)

                # if self.args.signal_normalize:
                #
                #     x_hat = awgn_channel_normalize(x_quantized, sigma_signal= self.args.sigma_signal,snr_db_signal=self.args.snr_db_signal)
                # else:
                #     x_hat =awgn_channel_no_normalize(x_quantized, sigma_signal=self.args.sigma_signal, snr_db_signal=self.args.snr_db_signal)

                outputs = self.head(y_reconstructed)
                batch_loss = self.criterion(outputs, labels)  # mean over batch
                loss_sum += batch_loss.item() * labels.size(0)  # accumulate by sample
                count += labels.size(0)


                # _, predicted = torch.max(outputs, 1)
                # correct += (predicted == labels).sum().item()
                # total += labels.size(0)
                if self.args.dataset == "BENet":
                    probs = torch.sigmoid(outputs)
                    predicted = (probs >=0.5 ).float()
                    correct += (predicted==labels).sum().item()
                    total += labels.numel()

                else:
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

    total_correct, total_acc_count = 0, 0
    total_loss_count = 0.0
    total_loss_sum = 0.0

    # ensure encoder is on correct device (heads are moved when wrapped into model)
    encoder = encoder.to(device)

    # for client_id in range(args.num_users):
    for client_id in sorted(heads.keys()):
        head = heads[client_id]
        model = nn.Sequential(encoder, head).to(device)
        model.eval()

        test_loader = DataLoader(
            Subset(test_dataset, list(user_groups[client_id])),
            batch_size=64, shuffle=False
        )
        if args.dataset == "BENet":
            criterion = nn.BCEWithLogitsLoss().to(device)

        elif args.dataset == "EuroSAT":
            criterion = torch.nn.CrossEntropyLoss().to(device)
        else:
            criterion = nn.CrossEntropyLoss().to(device)

        correct, acc_count = 0, 0
        loss_sum, loss_count = 0.0,0  # accumulate **per-sample** loss
        out_dim = head_out_dim(head)

        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)

                labels = labels.to(device)
                if args.dataset == "BENet":
                    labels = labels.float()
                    task_indices = torch.tensor(
                        args.client_task_indices[client_id],
                        dtype=torch.long,
                        device=device
                    )
                    labels = labels[:, task_indices]

                elif args.dataset == "EuroSAT":
                    mapping = label_mappings[client_id]

                    labels = torch.tensor(
                        [
                            mapping[int(label)]
                            for label in labels.detach().tolist()
                        ],
                        dtype=torch.long,
                        device=device,
                    )
                else:
                    labels = labels.long()   # CE needs Long target

                # client-specific label remap (if provided)
                    if label_mappings and client_id in label_mappings:
                        map_fn = label_mappings[client_id]
                        mapped = [map_fn(int(y.item())) for y in labels]
                        labels = torch.tensor(mapped, dtype=torch.long, device=device)

                # bound check once (avoids IndexError in CE)
                if args.dataset != "BENet":
                    if labels.numel() > 0 and (labels.min() < 0 or labels.max() >= out_dim):
                        raise ValueError(
                            f"[Client {client_id}] Mapped label(s) out of range for head with {out_dim} outputs. "
                            f"min={labels.min().item()}, max={labels.max().item()}"
                        )
                else:
                    if labels.shape[1] != out_dim:
                        raise ValueError(
                            f"[Test][Client{client_id}] label shape mismatch:"
                            f"labels.shape={labels.shape}, head_out={out_dim}"
                        )

                #  Without noise
                # outputs = model(images)

                # With noise
                # x = encoder(images)
                # Reparameterization
                mu, ln_var = encoder(images)
                x = reparameterize(mu, ln_var)
                power_before = x.pow(2).mean(dim=1).mean().item()
                print(power_before)
                if args.signal_normalize:
                    x = rms_power_normalize_signal(x)  # Normalization of latent vector to unit power

# ----------------------------------------------------------------------------------------
# --------------------------------------Quantization-------------------------------------
                indices = quantize_features(x, bits=8,lower_bound=-4.0,upper_bound=4.0)
# -----------------------------------------------------------------------------------------
# ------------------------Converting index to bits-----------------------------------------
                x_bin = index_to_bits(indices)
# -----------------------------------------------------------------------------------------
# -----------------------------Modulation--------------------------------------------------
                x_sym = modulation(x_bin)
# -----------------------------------------------------------------------------------------
# ----------------------------------Channel------------------------------------------------

                # x_quantized, _ = quantize_features(x, bits=8, upper_bound=4.0, use_ste=False,)

                h = float(los_channel_model(
                    Gt=args.Gt,
                    Gr=args.Gr,
                    wavelength_m=args.wavelength_m,
                    d_m=args.d_m,
                    L=1.0,
                ))

                sigma_rx = args.sigma_signal
                if sigma_rx is None or sigma_rx <= 0:
                    raise ValueError("Set --sigma_signal to a positive value.")

                # Target SNR = 0 dB = 1 in linear units.
                tx_power = sigma_rx ** 2 / h ** 2

                x_hat, y = los_symbol_channel(
                    x_sym,
                    h=h,
                    tx_power=tx_power,
                    noise_std_rx=sigma_rx,
                )
# ---------------------------------------------------------------------------------
# -----------------------------Demodulation------------------------------------------
                bits_received = demodulation(x_hat)
# ------------------------------------------------------------------------------------
# ------------------------------Convert bits to indices---------------------------------
                received_indices = bits_to_index(bits_received, bits=8)
# ----------------------Dequantize for reconstruction------------------------------
                y_reconstructed = dequantize_features(received_indices, bits=8,lower_bound=-4.0, upper_bound=4, dtype=x.dtype)

                # if args.signal_normalize:
                #     x_hat = awgn_channel_normalize(x_quantized, sigma_signal=args.sigma_signal,snr_db_signal=args.snr_db_signal)
                #
                # else:
                #     x_hat=awgn_channel_no_normalize(x_quantized, sigma_signal=args.sigma_signal, snr_db_signal=args.snr_db_signal)

                outputs = head(y_reconstructed)
                head = head.to(device)

                # loss returned is mean over the batch -> convert to sum then to per-sample
                bs = labels.size(0)
                batch_loss = criterion(outputs, labels)   # mean over batch
                loss_sum   += batch_loss.item() * bs
                loss_count  += bs

                if args.dataset == "BENet":
                    probs = torch.sigmoid(outputs)
                    predicted = (probs >= 0.5).float()
                    correct += (predicted == labels).sum().item()
                    acc_count += labels.numel()
                else:
                    _, predicted = torch.max(outputs, 1)
                    correct += (predicted == labels).sum().item()
                    acc_count += bs

        # per-client stats for this round
        acc_client = 100.0 * correct / acc_count if acc_count > 0 else 0.0
        loss_client = loss_sum / loss_count if loss_count > 0 else 0.0

        # stash into dicts
        if client_id not in client_test_accuracies:
            client_test_accuracies[client_id] = []
            client_test_losses[client_id]     = []
        client_test_accuracies[client_id].append(acc_client)
        client_test_losses[client_id].append(loss_client)

        # accumulate overall
        total_correct  += correct
        total_acc_count += acc_count
        total_loss_sum += loss_sum
        total_loss_count  += loss_count

    avg_acc  = 100.0 * total_correct / total_acc_count if total_acc_count > 0 else 0.0
    avg_loss = total_loss_sum / total_loss_count if total_loss_count > 0 else 0.0
    return avg_acc, avg_loss, client_test_accuracies, client_test_losses

def plot_class_gaussianity(collected, output_path):
    import numpy as np
    from statistics import NormalDist
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_pdf import PdfPages

    features = collected["features"].numpy().astype(np.float64)
    labels = collected["labels"].numpy()
    client_id = collected["client_id"]
    normal = NormalDist()

    with PdfPages(output_path) as pdf:
        for c in range(collected["num_classes"]):
            class_features = features[labels == c]
            n_samples, n_features = class_features.shape

            if n_samples < 3:
                raise ValueError(
                    f"Client {client_id}, class {c}: "
                    "need at least 3 samples; many more are preferable."
                )

            # Gaussian reference quantiles.
            probabilities = (
                np.arange(1, n_samples + 1) - 0.5
            ) / n_samples
            theoretical = np.array([
                normal.inv_cdf(float(p)) for p in probabilities
            ])

            for start in range(0, n_features, 8):
                fig = Figure(figsize=(12, 14))
                axes = fig.subplots(4, 4)
                axes = axes.ravel()

                for slot in range(8):
                    feature_id = start + slot
                    hist_ax = axes[2 * slot]
                    qq_ax = axes[2 * slot + 1]

                    if feature_id >= n_features:
                        hist_ax.set_visible(False)
                        qq_ax.set_visible(False)
                        continue

                    values = class_features[:, feature_id]
                    mean = values.mean()
                    std = values.std(ddof=1)

                    hist_ax.hist(
                        values, bins=30, density=True,
                        alpha=0.65, color="steelblue",
                    )
                    hist_ax.set_title(f"Feature {feature_id}: histogram")

                    if std <= 1e-12:
                        qq_ax.text(
                            0.1, 0.5, "Constant / near-constant feature",
                            transform=qq_ax.transAxes,
                        )
                        continue

                    grid = np.linspace(
                        min(values.min(), mean - 3 * std),
                        max(values.max(), mean + 3 * std),
                        200,
                    )
                    gaussian_pdf = np.exp(
                        -0.5 * ((grid - mean) / std) ** 2
                    ) / (std * np.sqrt(2 * np.pi))
                    hist_ax.plot(grid, gaussian_pdf, color="darkorange")

                    observed = np.sort((values - mean) / std)
                    qq_ax.plot(theoretical, observed, ".", markersize=2)
                    qq_ax.plot(
                        theoretical, theoretical, "--", color="darkorange"
                    )
                    qq_ax.set_title(f"Feature {feature_id}: Gaussian Q–Q")
                    qq_ax.set_xlabel("Gaussian quantiles")
                    qq_ax.set_ylabel("Standardized feature quantiles")

                fig.suptitle(
                    f"Client {client_id} | Task class {c} | "
                    f"{n_samples} images | Before quantization"
                )
                fig.tight_layout(rect=(0, 0, 1, 0.97))
                pdf.savefig(fig)
                fig.clear()