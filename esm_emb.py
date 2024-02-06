import torch
import esm
import csv

model_path = "/home/guxingyue/gxy/DPI/esm2_t6_8M_UR50D.pt"
model_lm, alphabet = esm.pretrained.load_model_and_alphabet_local(model_path)
batch_converter = alphabet.get_batch_converter()
model_lm.eval()  # disables dropout for deterministic results

# Function to calculate sequence representations
def get_sequence_representation(sequence):
    sequences = [("protein", sequence)]
    batch_labels, batch_strs, batch_tokens = batch_converter(sequences)
    batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

    with torch.no_grad():
        results = model_lm(batch_tokens, repr_layers=[6], return_contacts=True)

    token_representations = results["representations"][6]

    sequence_representations = []
    for i, tokens_len in enumerate(batch_lens):
        sequence_representations.append(token_representations[i, 1: tokens_len - 1].mean(0))

    return sequence_representations[0]


# Read sequences from protein_teneg.txt and calculate representations
file_path = "/mnt/sda/guxingyue/DPI/data/pro_test_shut.txt"
protein_data = []

with open(file_path, "r") as f:
    for line in f:
        seq = line.strip()
        sequence_representation = get_sequence_representation(seq)
        protein_data.append((seq, sequence_representation.tolist()))

# Write data to CSV file with each number in a separate cell
output_file = "/mnt/sda/guxingyue/DPI/data/pro_esmte320_shut.csv"
with open(output_file, mode="w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Sequence", "Representation"])

    for seq, representation in protein_data:
        writer.writerow([seq] + representation)
