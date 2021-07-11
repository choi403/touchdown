import numpy as np
import torch
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
import clip_beam_search as bs
from PIL import Image
import matplotlib.pyplot as plt


# ! Load model

model = torch.jit.load("model.pt").cuda().eval()
input_resolution = model.input_resolution.item()
context_length = model.context_length.item()
vocab_size = model.vocab_size.item()


# ! Perform greedy search


# image = Image.open('./test_images/test1.png')
image = Image.open('./test_images/pano_apqqbmmivfquan.jpg')

plt.imshow(torch.tensor(np.array(image)))
