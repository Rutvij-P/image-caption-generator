# compute_vocab_and_length.py
import json
from data_preprocessing import load_coco_data, build_vocab_and_encode_captions

# Load and preprocess the COCO dataset
image_paths, captions = load_coco_data()

# Build vocabulary and encode captions, also get the max_length of captions
_, _, _, vocab, max_length = build_vocab_and_encode_captions(captions)

# Determine the vocabulary size (plus one for zero padding)
vocab_size = len(vocab) + 1

# Save vocab_size and max_length to a file
with open('model_params.json', 'w') as f:
    json.dump({'vocab_size': vocab_size, 'max_length': max_length}, f)

print("Saved vocab_size and max_length to model_params.json")
