import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from PIL import Image, ImageTk
import os
import sys
from torchvision import transforms
import numpy as np
import torch
from model import EncoderCNN, DecoderRNN
from PIL import Image
from vocabulary import Vocabulary
import warnings
# 忽略特定警告
warnings.filterwarnings("ignore", category=UserWarning)

# 创建一个Tkinter主窗口
window = tk.Tk()
window.title("图像标注")

# 标签和输入框
label = tk.Label(window, text="输入图像路径:")
label.pack()
input_entry = tk.Entry(window)
input_entry.pack()

# 显示图像的区域
image_label = tk.Label(window)
image_label.pack()


# 图像标注函数
def generate_caption(image_path):
    transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    PIL_image = Image.open(image_path).convert('RGB')
    orig_image = np.array(PIL_image)
    image = transform_test(PIL_image)

    vocab = Vocabulary(3, 'vocab.pkl', "<start>", "<end>", "<unk>", None, True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    embed_size = 512
    hidden_size = 512
    vocab_size = len(vocab)

    encoder = EncoderCNN(embed_size)
    encoder.eval()
    decoder = DecoderRNN(embed_size, hidden_size, vocab_size)
    decoder.eval()

    encoder.load_state_dict(torch.load(os.path.join('./models', 'encoder-3.pkl')))
    decoder.load_state_dict(torch.load(os.path.join('./models', 'decoder-3.pkl')))

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

    features = encoder(image).unsqueeze(1)
    output = decoder.sample(features)
    sentence = clean_sentence(output)

    return sentence


# 生成描述按钮的回调函数
def generate_description():
    image_path = input_entry.get()
    if not os.path.isfile(image_path):
        messagebox.showerror("错误", "文件不存在，请提供有效的图像路径。")
        return
    caption = generate_caption(image_path)
    messagebox.showinfo("生成的描述", caption)


# 浏览图像文件按钮的回调函数
def browse_image():
    file_path = input_entry.get()
    if file_path:
        input_entry.delete(0, tk.END)
        input_entry.insert(0, file_path)
        display_image(file_path)


# 在图像区域显示所选图像
def display_image(image_path):
    img = Image.open(image_path)
    img = ImageTk.PhotoImage(img)
    image_label.config(image=img)
    image_label.image = img


# 生成描述按钮
generate_button = tk.Button(window, text="生成描述", command=generate_description)
generate_button.pack()

# 浏览图像文件按钮
browse_button = tk.Button(window, text="浏览图像文件", command=browse_image)
browse_button.pack()

# 运行主事件循环
window.mainloop()


