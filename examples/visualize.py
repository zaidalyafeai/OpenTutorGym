import matplotlib.pyplot as plt
import json
import numpy as np
import glob

files = glob.glob("examples/conversations_*.json")

# Define colors for each model
colors = ['skyblue', 'lightgreen', 'salmon','orange']
model_names = []
all_lengths = []

# Collect data for all models
for file, color in zip(files, colors):
    model_name = file.split("_")[1].replace(".json", "")
    model_names.append(model_name)
    with open(file, "r") as f:
        conversations = json.load(f)[:50]
        lengths = [len(conv) for conv in conversations]
        all_lengths.append(lengths)

# Plot all models in the same figure
plt.figure(figsize=(12, 4))

# Plot histograms
n_bins = 20
min_len = min([min(lengths) for lengths in all_lengths])
max_len = max([max(lengths) for lengths in all_lengths])
bins = np.linspace(min_len, max_len, n_bins)

for lengths, color, model_name in zip(all_lengths, colors, model_names):
    plt.hist(lengths, bins=bins, alpha=0.6, label=model_name, color=color)

plt.title("Distribution of Conversation Lengths Across Models", fontsize=15)
plt.xlabel("Conversation Length", fontsize=15)
plt.ylabel("Frequency", fontsize=15)
plt.legend(fontsize=15, loc='upper right')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("conversation_lengths_comparison.png")
plt.close()