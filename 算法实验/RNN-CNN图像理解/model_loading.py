import sys
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
import torch
import os
from model import EncoderCNN, DecoderRNN
from PIL import Image
from vocabulary import Vocabulary
import warnings


encoder_file = 'encoder-3.pkl'
decoder_file = 'decoder-3.pkl'

# 忽略特定警告
warnings.filterwarnings("ignore", category=UserWarning)

transform_test = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))
    ]
)

# Input the Image
PIL_image = Image.open(os.path.join("C:\\Users\\vJanGo\\Downloads\\BucerosBicornis_ZH-CN7795050230_1920x1080.jpg")).convert('RGB')
orig_image = np.array(PIL_image)
image = transform_test(PIL_image)

vocab = Vocabulary(3, 'C:\\Users\\vJanGo\\Desktop\\python_big_work\\Image-Captioning-main\\vocab.pkl', "<start>",
                   "<end>", "<unk>", None, True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Select appropriate values for the Python variables below.
embed_size = 512
hidden_size = 512

# The size of the vocabulary.
vocab_size = len(vocab)

# Initialize the encoder and decoder, and set each to inference mode.

encoder = EncoderCNN(embed_size)
encoder.eval()
decoder = DecoderRNN(embed_size, hidden_size, vocab_size)
decoder.eval()

# Load the trained weights.
encoder.load_state_dict(torch.load(os.path.join('./models', encoder_file)))
decoder.load_state_dict(torch.load(os.path.join('./models', decoder_file)))

# Move models to GPU if CUDA is available
encoder.to(device)
decoder.to(device)

def clean_sentence(output):
    captions = []
    for word_ids in output:
        word = vocab.idx2word[word_ids]
        captions.append(word)
        if word == '<end>':
            break
    sentence = ' '.join(captions)
    return sentence[7:-6]

image = image.unsqueeze(0).to(device)

def get_prediction():
    features = encoder(image).unsqueeze(1)
    output = decoder.sample(features)
    sentence = clean_sentence(output)
    print(sentence)