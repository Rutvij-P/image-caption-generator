import os
import matplotlib.pyplot as plt
from data_preprocessing import preprocess_image, build_vocab_and_encode_captions, load_coco_data, pad_encoded_captions  # Updated import

print("Current working directory:", os.getcwd())
# Define the base directory and load a subset of the COCO data
base_dir = 'dataset'  # Adjusted path
image_paths, captions = load_coco_data(base_dir=base_dir)
subset_size = 5  # Working with a small subset for testing
image_paths, captions = image_paths[:subset_size], captions[:subset_size]

# Preprocess the images
preprocessed_images = [preprocess_image(path) for path in image_paths]

# Build vocabulary and encode captions
encoded_captions, word_to_index, index_to_word, vocab, max_length = build_vocab_and_encode_captions(captions)

# Find the maximum length of the encoded captions for padding
max_length = max(len(caption) for caption in encoded_captions)

# Pad the encoded captions
padded_captions = pad_encoded_captions(encoded_captions, max_length)  # Updated function call

# Display the results
for i in range(subset_size):
    print(f"Original Caption: {captions[i]}")
    print(f"Encoded & Padded Caption: {padded_captions[i]}")
    img = plt.imread(image_paths[i])
    plt.imshow(img)
    plt.title('Sample Image')
    plt.show()
