import numpy as np

def entropy(labels):
  unique_labels, counts = np.unique(labels, return_counts=True)

  probabilities = counts / len(labels)
  entropy_value = -np.sum(probabilities * np.log2(probabilities))
  return entropy_value

def gain_ratio(data, target, feature):
  total_entropy = entropy(data[target])
  feature_entropy = 0
  for value in data[feature].unique():
    subset = data[data[feature] == value]
    prob = len(subset) / len(data)
    entropy_value = entropy(subset[target])
    feature_entropy += prob * entropy_value

  information_gain = total_entropy - feature_entropy

  split_info = -np.sum([(len(data[data[feature] == value]) / len(data)) * np.log2(len(data[data[feature] == value]) / len(data)) for value in data[feature].unique()])

  if split_info == 0:
    gain_ratio_value = 0
  else:
    gain_ratio_value = information_gain / split_info
  return gain_ratio_value
