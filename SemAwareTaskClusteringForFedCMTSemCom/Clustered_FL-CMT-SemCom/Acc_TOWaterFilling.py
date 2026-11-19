

def accuracy_evaluation(allocation_df, run_dir, mode, repeats, seed):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from model import TaskHead
    from quantization import quantize_features, dequantize_features
    if mode in ("bpsk", "both"):
        from channel import los_channel_model, los_symbol_channel
        from modulation import modulation, demodulation
        from quantization import index_to_bits, bits_to_index
        config_file = run_dir / "run_config.json"
        with config_file.open(encoding="utf8") as f:
            cfg = json.load(f)
        h = float(los_channel_model(Gt=cfg["Gt"], Gr=cfg["Gr"],
                  wavelength_m=cfg["wavelength_m"], d_m=cfg["d_m"], L=1.0))
    else:
        h = None
    cache = {}
    results = []
    for cid, client_rows in allocation_df.groupby("client_id"):
        saved = torch.load(run_dir / f"test_features_client_{cid}.pt",
                           map_location="cpu", weights_only=True)
        ckpt = torch.load(run_dir / f"client_{cid}_checkpoint.pt",
                          map_location="cpu", weights_only=True)
        if (saved["split"] != "test" or saved["stage"] != "before_quantization"
                or int(saved["client_id"]) != cid or int(ckpt["client_id"]) != cid):
            raise ValueError(f"Unexpected test metadata for client {cid}")
        features, labels = saved["features"].float(), saved["labels"].long()
        head = TaskHead(input_dim=features.shape[1], output_dim=int(ckpt["num_classes"]))
        head.load_state_dict(ckpt["head_state_dict"]); head.eval()
        for _, row in client_rows.iterrows():
            bits = int(row["bits"])
            method = "ideal" if mode == "ideal" else "bpsk"
            selected_modes = ["ideal", "bpsk"] if mode == "both" else [method]
            for selection in selected_modes:
                key = (cid, bits, selection, float(row["power_w"]) if selection == "bpsk" else None,
                       float(row["gamma_per_w"]) if selection == "bpsk" else None)
                if key not in cache:
                    torch.manual_seed(seed + cid)  # Common random numbers across conditions
                    correct = 0
                    for x, target in zip(features.split(128), labels.split(128)):
                        indices = quantize_features(x,bits=bits,lower_bound=-4.0,upper_bound=4.0)
                        if selection == "ideal":
                            reconstructed = dequantize_features(indices,bits=bits,lower_bound=-4.0,
                                                                 upper_bound=4.0,dtype=x.dtype)
                            pred = head(reconstructed).argmax(dim=1)
                            correct += int((pred==target).sum())
                        else:
                            symbols = modulation(index_to_bits(indices,bits=bits))
                            # gamma=h^2/sigma_rx^2, so physical sigma=h/sqrt(gamma).
                            sigma = h / math.sqrt(float(row["gamma_per_w"]))
                            # Match user's existing test-accuracy channel model.
                            for _ in range(repeats):
                                equalized, _ = los_symbol_channel(symbols, h=h,
                                    tx_power=float(row["power_w"]),noise_std_rx=sigma)
                                decoded = bits_to_index(demodulation(equalized),bits=bits)
                                reconstructed = dequantize_features(decoded,bits=bits,
                                     lower_bound=-4.0,upper_bound=4.0,dtype=x.dtype)
                                pred = head(reconstructed).argmax(dim=1)
                                correct += int((pred==target).sum())
                    denom = len(labels) * (1 if selection == "ideal" else repeats)
                    cache[key] = 100.0 * correct / denom
                results.append({"scheme":row["scheme"], "reference_snr_db":float(row["reference_snr_db"]),
                    "client_id":int(cid),"mode":selection,"bits":bits,
                    "accuracy_pct":cache[key],"test_images":len(labels),
                    "noise_repeats":repeats if selection=="bpsk" else 1,
                    "actual_snr_db":float(row["actual_snr_db"])})
    return pd.DataFrame(results)


def accuracy_plots(adf, out_dir):
    if adf.empty:
        return
    for mode, group in adf.groupby("mode"):
        for cid, clients in group.groupby("client_id"):
            fig,ax=plt.subplots(figsize=(7,4.6))
            for scheme, sub in clients.groupby("scheme"):
                sub=sub.sort_values("reference_snr_db")
                ax.plot(sub.reference_snr_db,sub.accuracy_pct,"o-",label=scheme)
            ax.set(xlabel="Reference SNR (dB)",ylabel="Test accuracy (%)",ylim=(0,101),
                   title=f"Client {cid} accuracy ({mode})")
            ax.grid(alpha=0.3);ax.legend();fig.tight_layout()
            fig.savefig(out_dir/f"client_{cid}_accuracy_{mode}_vs_snr.png",dpi=180)
            plt.close(fig)
        mean=group.groupby(["scheme","reference_snr_db"],as_index=False)["accuracy_pct"].mean()
        fig,ax=plt.subplots(figsize=(7,4.6))
        for scheme, sub in mean.groupby("scheme"):
            sub=sub.sort_values("reference_snr_db")
            ax.plot(sub.reference_snr_db,sub.accuracy_pct,"o-",label=scheme)
        ax.set(xlabel="Reference SNR (dB)",ylabel="Mean client accuracy (%)",ylim=(0,101),
               title=f"Mean system accuracy ({mode})")
        ax.grid(alpha=0.3);ax.legend();fig.tight_layout()
        fig.savefig(out_dir/f"comparison_mean_accuracy_{mode}_vs_snr.png",dpi=180)
        plt.close(fig)