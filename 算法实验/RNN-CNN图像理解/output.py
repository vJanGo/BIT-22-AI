import sys
from torchvision import transforms
import numpy as np
import torch
import os
from model import EncoderCNN, DecoderRNN
from PIL import Image
from vocabulary import Vocabulary
import warnings

# Ignoring specific warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Define the transformation for image preprocessing
transform_test = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])


# Define the image captioning function
def generate_caption(image_path):
    # Load the pre-trained encoder and decoder models
    encoder_file = 'encoder-3.pkl'
    decoder_file = 'decoder-3.pkl'

    # Load and preprocess the input image
    PIL_image = Image.open(image_path).convert('RGB')
    orig_image = np.array(PIL_image)
    image = transform_test(PIL_image)

    # Load the vocabulary
    vocab = Vocabulary(3, 'vocab.pkl', "<start>", "<end>", "<unk>", None, True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Select appropriate values for model hyperparameters
    embed_size = 512
    hidden_size = 512
    vocab_size = len(vocab)

    # Initialize the encoder and decoder in inference mode
    encoder = EncoderCNN(embed_size)
    encoder.eval()
    decoder = DecoderRNN(embed_size, hidden_size, vocab_size)
    decoder.eval()

    # Load the trained weights
    encoder.load_state_dict(torch.load(os.path.join('./models', encoder_file)))
    decoder.load_state_dict(torch.load(os.path.join('./models', decoder_file)))

    # Move models to GPU if CUDA is available
    encoder.to(device)
    decoder.to(device)

    # Define a function to clean the generated sentence

    def clean_sentence(output):
        captions = []
        for word_ids in output:
            word = vocab.idx2word[word_ids]
            captions.append(word)
            if word == '<end>':
                break
        sentence = ' '.join(captions)
        return sentence[7:-6]

    # Prepare the input image on the selected device
    image = image.unsqueeze(0).to(device)

    # Generate the caption for the input image
    features = encoder(image).unsqueeze(1)
    output = decoder.sample(features)
    sentence = clean_sentence(output)

    return sentence


# Usage example
if __name__ == '__main__':
    image_path = input("Enter the path to the image: ")
    generated_caption = generate_caption(image_path)
    print("Generated Caption: " + generated_caption)


