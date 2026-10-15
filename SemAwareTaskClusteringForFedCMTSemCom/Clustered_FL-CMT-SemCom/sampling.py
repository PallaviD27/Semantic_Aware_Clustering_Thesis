"""Start"""

import numpy as np


def custom_skewed_partition(dataset, num_users, seed=None):

 assert num_users == 3
# Iteration1
#  Target_Allocation = {
#   0: [1975, 0, 0],
#   1: [0, 2552, 200],
#   2: [2340, 2079, 0],
#   3: [2000, 0, 500],
#   4: [1947, 0, 1947],
#   5: [0, 1807, 0],
#   6: [0, 0, 2700],
#   7: [2089, 700, 0],
#   8: [200, 0, 1958],
#   9: [0, 1980, 1986],
#  }
 #  Iteration2/3/4
 # Target_Allocation = {
 #  0: [0, 0, 1974],
 #  1: [0, 2552, 0],
 #  2: [2340, 2079, 0],
 #  3: [0, 500, 0],
 #  4: [0, 0, 1947],
 #  5: [0, 0, 1807],
 #  6: [0, 0, 2700],
 #  7: [2089, 0, 0],
 #  8: [1951, 0, 500],
 #  9: [1983, 0, 200],
 # }
 Target_Allocation = {
  0: [0, 0, 1974],
  1: [0, 2552, 0],
  2: [1000, 2079, 0],
  3: [0, 500, 0],
  4: [0, 0, 200],
  5: [0, 0, 500],
  6: [0, 500, 2000],
  7: [1500, 0, 0],
  8: [1551, 0, 500],
  9: [1583, 0, 200],
 }

 # Target_Allocation = {
 #  0: [1975, 1974, 1974],
 #  1: [2190, 2552, 2000],
 #  2: [2340, 2079, 1539],
 #  3: [2000, 2131, 2000],
 #  4: [1947, 1947, 1948],
 #  5: [1807, 1807, 1807],
 #  6: [1718, 1500, 2700],
 #  7: [2089, 2088, 2088],
 #  8: [1951, 1942, 1958],
 #  9: [1983, 1980, 1986],
 # }
 #  Get labels as a NumPy array; e.g. [5,0,9,1,...]
 y = dataset.targets.numpy()


 counts = np.bincount(y, minlength=10)
 print("MNIST train counts per digit (0...9:", counts.tolist())

 rng = np.random.default_rng(seed)
 indices_by_digit = {d: np.where(y==d)[0] for d in range(10)} # For each digit d (0 to 9) find all
                                                              # row indices whose label equals d
                                                              # gives result like {
                                                              #   0: array([  13,  102,  377, ...]), - positions where label==0
                                                              #   1: array([   5,   44,  210, ...]),
 # randomly shuffle indices per digit
 # If all “2” labels are at positions[7, 50, 88, 91, ...], after shuffling they might become[91, 7, 88, 50, ...].
 # Then,if Client 1 needs 2340 twos, we take the first 2340 from this shuffled list—i.e., a deterministic random sample of the “2” images.
 for d in range(10):
  rng.shuffle(indices_by_digit[d])


 # Sanity check if more samples are allocated than existing for any digit
 row_counts = [len(indices_by_digit[d]) for d in range(10)]
 for d in range(10):
  allocated = sum(Target_Allocation[d])
  available = row_counts[d]
  assert allocated <= available, (f"Digit {d}: allocation {allocated} > available {available}")

 # compute how many samples each client will get (for info + later checks)
 col_sums = [sum(Target_Allocation[d][u] for d in range(10)) for u in range(num_users)]
 print("Planned samples per client:", col_sums)

 # Slice per digit
 dict_users = {u: [] for u in range(num_users)}
 for d in range(10):
  start = 0
  for u in range(num_users):
   k = Target_Allocation[d][u]
   if k == 0:
    continue
   dict_users[u].extend(indices_by_digit[d][start:start+k].tolist())
   start += k


 for u in range(num_users):
  dict_users[u] = np.array(dict_users[u], dtype=np.int64)
  assert len(dict_users[u]) == col_sums[u], (
            f"Client {u}: got {len(dict_users[u])} indices, expected {col_sums[u]}"
        )

 all_ids = np.concatenate([dict_users[u] for u in range(num_users)])
 assert len(all_ids) == len(set(all_ids.tolist())), "Overlap found between client splits"

 return dict_users


from collections import Counter

# Print raw MNIST digit counts for each client subset.
def print_raw_label_distribution(dict_users,dataset, title="Raw label distribution"):
 print(f"\n======= {title} =======")
 for client_id, indices in dict_users.items():
  labels = dataset.targets[indices]
  counts = Counter(labels.numpy())
  total = len(indices)

  print(f"\n Client {client_id} raw labels (Total:{total}):")
  for d in range(10):
   print(f" digit {d}: {counts.get(d, 0)}")
