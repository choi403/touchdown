
# Taken and modified from  https://colab.research.google.com/github/openai/clip/blob/master/notebooks/Interacting_with_CLIP.ipynb#scrollTo=w1l_muuhZ_Nk
# Setup
import numpy as np
import torch
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from PIL import Image
import gzip
import html
import os
from functools import lru_cache

import ftfy
import regex as re

import os
import skimage
# import IPython.display
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

from collections import OrderedDict
import torch


image_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).cuda()
image_std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).cuda()


@lru_cache()
def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a corresponding list of unicode strings.
    The reversible bpe codes work on unicode strings.
    This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
    When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
    This is a signficant percentage of your normal, say, 32K bpe vocab.
    To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
    And avoids mapping to whitespace/control characters the bpe code barfs on.
    """
    bs = list(range(ord("!"), ord("~")+1))+list(range(ord("¡"), ord("¬")+1))+list(range(ord("®"), ord("ÿ")+1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8+n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))

def get_pairs(word):
    """Return set of symbol pairs in a word.
    Word is represented as tuple of symbols (symbols being variable-length strings).
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs

def basic_clean(text):
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()

def whitespace_clean(text):
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text

class SimpleTokenizer(object):
    def __init__(self, bpe_path: str = "bpe_simple_vocab_16e6.txt.gz"):
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        merges = gzip.open(bpe_path).read().decode("utf-8").split('\n')
        merges = merges[1:49152-256-2+1]
        merges = [tuple(merge.split()) for merge in merges]
        vocab = list(bytes_to_unicode().values())
        vocab = vocab + [v+'</w>' for v in vocab]
        for merge in merges:
            vocab.append(''.join(merge))
        vocab.extend(['<|startoftext|>', '<|endoftext|>'])
        self.encoder = dict(zip(vocab, range(len(vocab))))
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.bpe_ranks = dict(zip(merges, range(len(merges))))
        self.cache = {'<|startoftext|>': '<|startoftext|>', '<|endoftext|>': '<|endoftext|>'}
        self.pat = re.compile(r"""<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+""", re.IGNORECASE)

    def bpe(self, token):
        if token in self.cache:
            return self.cache[token]
        word = tuple(token[:-1]) + ( token[-1] + '</w>',)
        pairs = get_pairs(word)

        if not pairs:
            return token+'</w>'

        while True:
            bigram = min(pairs, key = lambda pair: self.bpe_ranks.get(pair, float('inf')))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except:
                    new_word.extend(word[i:])
                    break

                if word[i] == first and i < len(word)-1 and word[i+1] == second:
                    new_word.append(first+second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        word = ' '.join(word)
        self.cache[token] = word
        return word

    def encode(self, text):
        bpe_tokens = []
        text = whitespace_clean(basic_clean(text)).lower()
        for token in re.findall(self.pat, text):
            token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))
            bpe_tokens.extend(self.encoder[bpe_token] for bpe_token in self.bpe(token).split(' '))
        return bpe_tokens

    def decode(self, tokens):
        text = ''.join([self.decoder[token] for token in tokens])
        text = bytearray([self.byte_decoder[c] for c in text]).decode('utf-8', errors="replace").replace('</w>', ' ')
        return text

# ! Implementations

# ! Slice image into m * n pieces
# Taken from https://stackoverflow.com/questions/16856788/slice-2d-array-into-smaller-2d-arrays
def blockshaped(arr, nrows, ncols):
    """
    Return an array of shape (n, nrows, ncols) where
    n * nrows * ncols = arr.size

    If arr is a 2D array, the returned array should look like n subblocks with
    each subblock preserving the "physical" layout of arr.
    """
    h, w = arr.shape
    assert h % nrows == 0, "{} rows is not evenly divisble by {}".format(h, nrows)
    assert w % ncols == 0, "{} cols is not evenly divisble by {}".format(w, ncols)
    return (arr.reshape(h//nrows, nrows, -1, ncols)
               .swapaxes(1,2)
               .reshape(-1, nrows, ncols))
    
from PIL import Image

# ! Slice image
# ! Taken and modified from https://stackoverflow.com/questions/5953373/how-to-split-image-into-multiple-pieces-in-python
def crop(input, row, col):
    '''
        input: PIL Image
        returns a 2d nested list of cropped pieces (PIL Images)
    '''
    #print(input)
    #input.save('orig.jpg')
    cropped_images = []
    imgwidth, imgheight = input.size
    
    height = imgheight // row
    width = imgwidth // col
    
    for i in range(0,imgheight - (imgheight % height),height):
        cropped_row = []
        for j in range(0,imgwidth - (imgwidth % width),width):
            box = (j, i, j+width, i+height)
            a = input.crop(box)
            
            #a.save(f'b{i}{j}.jpg')
            cropped_row.append(a)
        cropped_images.append(cropped_row)
    #print(cropped_images)
    
    return cropped_images
    
# ! Greedy baseline search algorithm 
# ! 1. Slice image into m1 * n1 pieces
# ! 2. Get similarity scores using the first phrase
# ! 3. Slice image into m2 * n2 pieces
# ! 4. Get similarity scores using the second phrase
# ! 5. ... 
# ! 6. (optional, for the last phrase) Switch the word "touchdown" and edit something into the image
# ! 7. Slice image into mi * ni pieces
# ! 8. Get similarity scores using the last phrase


# ! ? 
image_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).cuda()
image_std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).cuda()

def get_image_features(cropped_images, model): 
    row = len(cropped_images)
    col = len(cropped_images[0])
    
    images = []
    
    preprocess = Compose([
        Resize(model.input_resolution.item(), interpolation=Image.BICUBIC),
        CenterCrop(model.input_resolution.item()),
        ToTensor()
    ])
    
    # Preprocess 
    for i in range(row):
        for j in range(col):
            image = preprocess(cropped_images[i][j])
            images.append(image)
            
    # Stack and normalize
    image_input = torch.tensor(np.stack(images)).cuda()
    image_input -= image_mean[:, None, None]
    image_input /= image_std[:, None, None]
    
    # Encode images
    with torch.no_grad():
        image_features = model.encode_image(image_input).float()
    
    image_features /= image_features.norm(dim=-1, keepdim=True)
    
    return image_features

def get_text_features(texts, model, tokenizer):
    text_tokens = [tokenizer.encode(desc) for desc in texts]
    
    text_input = torch.zeros(len(text_tokens), model.context_length, dtype=torch.long)
    sot_token = tokenizer.encoder['<|startoftext|>']
    eot_token = tokenizer.encoder['<|endoftext|>']

    for i, tokens in enumerate(text_tokens):
        tokens = [sot_token] + tokens + [eot_token]
        text_input[i, :len(tokens)] = torch.tensor(tokens)

    text_input = text_input.cuda()

    with torch.no_grad():
        text_features = model.encode_text(text_input).float()
    
    return text_features


# ! Test implementation
def greedy_search(image, texts, model, tokenizer, rows_columns):
    cropped_1 = crop(image, rows_columns[0][0], rows_columns[0][1])
    
    image_features_1 = get_image_features(cropped_1, model)
    text_features_1 = get_text_features(texts[0], model, tokenizer)
    
    similarity_1 = text_features_1.cpu().numpy() @ image_features_1.cpu().numpy().T
    
    # ! cropped_1[0]: hardcoded
    cropped_2 = crop(cropped_1[0][1], rows_columns[1][0], rows_columns[1][1])
    
    image_features_2 = get_image_features(cropped_2, model)
    text_features_2 = get_text_features(texts[1], model, tokenizer)
    
    similarity_2 = text_features_2.cpu().numpy() @ image_features_2.cpu().numpy().T
    
    return similarity_1, cropped_1, similarity_2, cropped_2

    
def greedy_search_custom_images(images, texts, model, tokenizer):
    image_features_1 = get_image_features(images, model)
    text_features_1 = get_text_features(texts[0], model, tokenizer)
    
    similarity_1 = text_features_1.cpu().numpy() @ image_features_1.cpu().numpy().T
    
    return similarity_1, images

def get_grid_search_input_images(input_image, num_rows, num_cols):
    scale = .5

    star_image = Image.open('yellow_star.png', 'r')
    star_image_w, star_image_h = star_image.size
    star_image = star_image.resize((int(star_image_w * scale), int(star_image_h * scale)), Image.ANTIALIAS)
    star_image_w, star_image_h = star_image.size

    cropped_images = []

    # input_image = Image.new('RGBA', (1440, 900), (255, 255, 255, 255))

    imgwidth, imgheight = input_image.size
    height = imgheight // num_rows
    width = imgwidth // num_cols


    for i in range(0,imgheight - (imgheight % height),height):
        cropped_row = []
        for j in range(0,imgwidth - (imgwidth % width),width):
            background = input_image.copy()

            offset = (j, i)
            background.paste(star_image, offset, star_image)
            

            
            #a.save(f'b{i}{j}.jpg')
            cropped_row.append(background)
        cropped_images.append(cropped_row)
    
    return cropped_images

def single_image_grid_search(input_image, texts, model, tokenizer, num_rows, num_cols):

    for i in texts:
        for j in i:
            j = j.lower().replace('touchdown', 'yellow star').replace('waldo', 'yellow star')

    images = get_grid_search_input_images(input_image, num_rows, num_cols)

    image_features_1 = get_image_features(images, model)
    text_features_1 = get_text_features(texts[0], model, tokenizer)
    
    similarity_1 = text_features_1.cpu().numpy() @ image_features_1.cpu().numpy().T
    
    return similarity_1, images


# Taken from https://www.pyimagesearch.com/2015/03/23/sliding-windows-for-object-detection-with-python-and-opencv/
def get_sliding_window_np(image_numpy, stepSize, windowSize):
	for y in range(0, image_numpy.shape[0], stepSize):
		for x in range(0, image_numpy.shape[1], stepSize):
			yield (x, y, image_numpy[y:y + windowSize[1], x:x + windowSize[0]])

def greedy_search_sliding_window_step(image_numpy, text_prompt, step_size, window_size, model, tokenizer):
    '''
        image_numpy: RGB image numpy array 
        text_prompt: A single sentence (ex: 'A photo of a dog')
        step_size: sliding window step size (int)
        window_size: sliding window size (int, int)
        model: CLIP model (model = torch.jit.load("model.pt").cuda().eval())
        tokenizer: tokenizer (SimpleTokenizer())
        returns: 
            similarity: nested list of all similarities ([[2.10, ..., 3.10]], nested list length = len(list_window_image_pil))
            list_window_image_npy: list of numpy window images (numpy arrays)
            list_window_positions: list of window positions ((x, y) tuples)
    '''
    # image_numpy = np.array(image) 

    list_window_tuple = list(get_sliding_window_np(image_numpy, step_size, window_size))
    list_window_image_npy = list(i[2] for i in list_window_tuple)
    list_window_positions = list((i[0], i[1]) for i in list_window_tuple)

    # print(list_window_positions)

    list_window_image_pil = [Image.fromarray(i) for i in list_window_image_npy]

    # print(list_window)
    
    image_features = get_image_features([list_window_image_pil,], model)

    text_features = get_text_features([text_prompt,], model, tokenizer)
    
    similarity = text_features.cpu().numpy() @ image_features.cpu().numpy().T

    print(text_features.cpu().numpy().shape, image_features.cpu().numpy().T.shape)
    
    return similarity, list_window_image_npy, list_window_positions

def 

if __name__ == '__main__':
    model = torch.jit.load("model.pt").cuda().eval()
    input_resolution = model.input_resolution.item()
    context_length = model.context_length.item()
    vocab_size = model.vocab_size.item()

    tokenizer = SimpleTokenizer()

    texts = [['look for the red beverage cooler or refrigerator to the left of the man setting the table.'], ['look to the bottom left of the cooler.'], ['waldo is at the bottom left, slightly off the ground.']]

    # ! texts input should be a list (text_features does so)

    temp = Image.open('./test_images/pano_apqqbmmivfquan.jpg').convert('RGB')
    similarity, list_window = greedy_search_sliding_window(temp, texts[0][0], 80, (1400, 1400), model, tokenizer)


    # similarity_flattened = []
    # for i in similarity:
    #     similarity_flattened.extend(i)


    # num_images = len(list_window)

    # fig, axes = plt.subplots(num_images)
    # fig.set_size_inches(40, 40)

    # print(num_images)
        
    # preprocess = Compose([
    #     Resize(model.input_resolution.item(), interpolation=Image.BICUBIC),
    #     CenterCrop(model.input_resolution.item()),
    #     ToTensor()
    # ])

    # for j in range(num_images):
    #     axes[j].imshow(preprocess(list_window[j].convert("RGB")).permute(1,2,0))
    #     axes[j].set_title(similarity_flattened[j])

    # plt.suptitle(texts[0][0])
    # plt.tight_layout()

    # plt.savefig('./result_images/windows.png')
    # fig.clf()


    # ! heatmap
    print(similarity)
    # print(list_window)

    # ! 100, (400, 400) -> (23, 46)
    # ! 75, (600, 600) -> (31, 61)
    # ! 80, (800, 800) -> (29, 57) 
    plt.imshow(np.array(similarity).reshape((29, 57)), cmap=plt.cm.gist_gray)
    plt.savefig('./result_images/heatmap.png')
    

    # test
