import torch
from quantization import index_to_bits,bits_to_index, quantize_features, dequantize_features
from channel import los_symbol_channel,los_channel_model
from modulation import modulation, demodulation

t= torch.tensor([0.0, 1.23, 2.0, 4.0, 4.6,2.3,6.8,3,0.5,1.3,2.6,3.6,3.3,2.4,1.3,1.1,4.5], dtype = torch.float32)

transmitted_indices = quantize_features(t,bits=8,upper_bound=4 )
print("Transmitted indices:", transmitted_indices)

bits = index_to_bits(transmitted_indices)
print("Transmitted bits:",bits)

x_modulated = modulation(bits)
print("Modulated bits:",x_modulated)

x_hat, y = los_symbol_channel(x_modulated, h=0.5, tx_power = 1.0, noise_std_rx=0.5)

print("Received Symbols:", y)
print("Equalized Symbols:", x_hat)


bits_received = demodulation(x_hat)
print("Received bits:", bits_received)


indices_received = bits_to_index(bits_received)
print("Recovered indices:", indices_received)

received_features = dequantize_features(indices_received, bits=8, upper_bound=4, dtype=t.dtype)

expected_features = dequantize_features(transmitted_indices, bits=8, upper_bound=4, dtype=t.dtype)

print("Received features:", received_features)
print("Expected features:", expected_features)

# assert torch.equal(transmitted_indices, indices_received), \
#     "Recovered indices do not match!"
#
# assert torch.equal(expected_features, received_features), \
#     "Receiver reconstruction does not match transmitter reconstruction!"

bit_errors = (bits_received != bits).sum().item()
total_bits = bits.numel()

print("Bit errors:", bit_errors, "/", total_bits)
print("Bit error rate:", bit_errors / total_bits)

