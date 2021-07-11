
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
    texts = [texts,]
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


# ! Test implementation, not used
def _greedy_search(image, texts, model, tokenizer, rows_columns):
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

# ! not used
def _greedy_search_custom_images(images, texts, model, tokenizer):
    image_features_1 = get_image_features(images, model)
    text_features_1 = get_text_features(texts[0], model, tokenizer)
    
    similarity_1 = text_features_1.cpu().numpy() @ image_features_1.cpu().numpy().T
    
    return similarity_1, images

def get_grid_overlayed_images(input_image, num_rows, num_cols, safe=False):
    scale = 1.2

    star_image = Image.open('ipod_timesnewroman_white.png', 'r').convert("RGBA")
    star_image_w, star_image_h = star_image.size
    star_image = star_image.resize((int(star_image_w * scale), int(star_image_h * scale)), Image.ANTIALIAS)
    star_image_w, star_image_h = star_image.size

    if safe:
        height_dec = star_image_h
        width_dec = star_image_w
    else:
        height_dec = 0
        width_dec = 0


    cropped_images = []
    list_position = []

    # input_image = Image.new('RGBA', (1440, 900), (255, 255, 255, 255))

    imgwidth, imgheight = input_image.size
    height = imgheight // num_rows
    width = imgwidth // num_cols

    for i in range(0,imgheight - (imgheight % height) - height_dec,height):
        cropped_row = []
        for j in range(0,imgwidth - (imgwidth % width) - width_dec,width):
            background = input_image.copy()

            offset = (j, i)
            list_position.append(offset)

            background.paste(star_image, offset, star_image)
            
            # background.save(f'./result_images/temp/{i}-{j}.jpg')
            cropped_row.append(background)
        cropped_images.append(cropped_row)
    
    return cropped_images, list_position, (star_image_w, star_image_h)

def single_image_grid_search(image_np, text, model, tokenizer, num_rows, num_cols, safe=False):
    text = text.lower().replace('touchdown', 'iPod').replace('waldo', 'iPod')

    images, list_position, (overlay_w, overlay_h) = get_grid_overlayed_images(Image.fromarray(image_np), num_rows, num_cols, safe=safe)

    image_features = get_image_features(images, model)
    text_features = get_text_features(text, model, tokenizer)

    print(len(image_features), len(text_features))
    
    similarity = text_features.cpu().numpy() @ image_features.cpu().numpy().T

    idx_max_similarity = similarity.argmax(axis=1)[0]

    
    return similarity, images, idx_max_similarity, list_position, (overlay_w, overlay_h)


# Taken from https://www.pyimagesearch.com/2015/03/23/sliding-windows-for-object-detection-with-python-and-opencv/
def get_sliding_window_np(image_numpy, stepSize, windowSize):
	for y in range(0, image_numpy.shape[0], stepSize):
		for x in range(0, image_numpy.shape[1], stepSize):
			yield (x, y, image_numpy[y:y + windowSize[1], x:x + windowSize[0]])

def greedy_search_sliding_window_step(image_numpy, text_prompt, step_size, window_size, model, tokenizer):
    '''
        Single step of the greedy search.

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

    # ! exclude windows on edges
    list_window_tuple = list(i for i in get_sliding_window_np(image_numpy, step_size, window_size) if i[2].shape[:2] == (window_size))
    list_window_image_npy = list(i[2] for i in list_window_tuple)
    list_window_positions = list((i[0], i[1]) for i in list_window_tuple)

    # print(list_window_positions)

    list_window_image_pil = [Image.fromarray(i) for i in list_window_image_npy]

    # print(list_window)
    
    image_features = get_image_features([list_window_image_pil,], model)

    text_features = get_text_features(text_prompt, model, tokenizer)
    
    similarity = text_features.cpu().numpy() @ image_features.cpu().numpy().T

    print(text_features.cpu().numpy().shape, image_features.cpu().numpy().T.shape)
    
    return similarity, list_window_image_npy, list_window_positions


# TODO: Per-text, per-prompt greedy search
# TODO: changing step_size, window_size along steps

def greedy_search_sliding_window(image_numpy, text_prompts, step_sizes, window_sizes, model, tokenizer):
    # initial input value for greedy search
    selected_window = image_numpy

    list_selected_window = []
    list_similarities = []
    list_positions = []
    list_selected_index = []

    final_x = 0
    final_y = 0

    for i, text_prompt in enumerate(text_prompts):
        similarity, list_window_image_npy, list_window_positions = greedy_search_sliding_window_step(
                                                                                            selected_window,
                                                                                            text_prompt,
                                                                                            step_sizes[i],
                                                                                            window_sizes[i],
                                                                                            model,
                                                                                            tokenizer)
        max_similarity = similarity.max(axis=1)[0]
        # print(max_similarity)
        idx_max_similarity = similarity.argmax(axis=1)[0]
        # print(idx_max_similarity)
        selected_window = list_window_image_npy[idx_max_similarity]

        # im = Image.fromarray(selected_window)
        # im.save(f'./result_images/temp/{i}.jpg')

        print(f'Step {i}: Prompt <{text_prompt}> --- Max similarity = {max_similarity} at pos {list_window_positions[idx_max_similarity]} with index {idx_max_similarity}')

        list_selected_window.append(selected_window)
        list_similarities.append(similarity)
        list_positions.append(list_window_positions)
        list_selected_index.append(idx_max_similarity)

        final_x += list_window_positions[idx_max_similarity][0]
        final_y += list_window_positions[idx_max_similarity][1]

    return list_similarities, list_selected_window, list_positions, list_selected_index, (final_x, final_y)

def get_width_height(list_positions):
    '''
        find width and height of the similarity heatmap from the flattened list of positions.
    '''
    width = len(list(i for i in list_positions if i[1] == 0))
    height = len(list_positions) / width

    assert int(width) == width and int(height) == height

    return int(height), int(width)
    

def split_long_sentence(text, break_len=60):
    sentences = text.split('. ')
    for i, e in enumerate(sentences):
        if i < len(sentences) - 1:
            sentences[i] = sentences[i] + '.'

    

    output = []
    for i in sentences:
        last_j = 0
        if len(i) > break_len: 
            for j in range(break_len, len(i) + break_len, break_len):
                split_sentence = i[last_j:j]
                output.append(split_sentence)
                last_j = j
        else: 
            output.append(i)

    return '\n'.join(output)







if __name__ == '__main__':
    model = torch.jit.load("model.pt").cuda().eval()
    tokenizer = SimpleTokenizer()
    input_resolution = model.input_resolution.item()
    context_length = model.context_length.item()
    vocab_size = model.vocab_size.item()

    preprocess = Compose([
        Resize(model.input_resolution.item(), interpolation=Image.BICUBIC),
        CenterCrop(model.input_resolution.item()),
        ToTensor()
    ])

    # original_texts = ['look for the red beverage cooler or refrigerator to the left of the man setting the table.', 'look to the bottom left of the cooler.', 'An iPod is at the bottom left, slightly off the ground.']
    original_texts = ['find a brown door on the right of the room.', 'locate the candles that are close to the door.', 'waldo is on the wall right on top of the last candle that is on the left at the same level that is the air ventilation.']
    # original_texts = ['find the small little step that you can step on to get up on the stage.', 'the back red colored wall has a large window.', 'above the window along where the wall meets the ceiling is a silver air vent.', 'waldo is on top of that air vent and his hand is pointing up to the round white light along the white pillar that people are sitting in front of.']
    accumulated_texts = [' '.join(original_texts[:i+1]) for (i, e) in enumerate(original_texts)]
    all_texts = [' '.join(original_texts) for i in range(len(original_texts) - 1)]
    all_but_last_texts = all_texts[:-1] + [original_texts[-1]]
    except_final_texts = [' '.join(original_texts[:-1]) for i in range(len(original_texts) - 1)] + [all_texts[0]]
    texts = except_final_texts


    # ! texts input should be a list (text_features does so)

    # fn = "../data/refer360images/indoor/restaurant/pano_apqqbmmivfquan.jpg"
    fn = "../data/refer360images/indoor/restaurant/pano_afzjkcqouejqpf.jpg"
    # fn = "../data/refer360images/indoor/restaurant/pano_aiqboxmskwofbk.jpg"
    temp = np.array(Image.open(fn).convert('RGB'))
    # similarity, list_window, list_positions = greedy_search_sliding_window_step(temp, texts[0], 800, (1400, 1400), model, tokenizer)

    list_similarity, list_selected_window, list_position, list_selected_index, (out_x, out_y) = greedy_search_sliding_window(temp, texts, [80, 80, 80, 80], [(2276, 2276), (1600, 1600), (1200, 1200), (1900, 1900)], model, tokenizer)

    print(f"after greedy search: (x, y) = {(out_x, out_y)}")

    for i in list_position:
        print(get_width_height(i))


    # ! plot step by step greedy search (heatmap, window)
    # ! n steps -> n heatmaps + n windows = 2n


    fig, axes = plt.subplots(len(list_similarity) + 2, 2)
    fig.set_size_inches(10, 15)

    # * First row: full panorama
    axes[0, 0].imshow(Image.fromarray(temp))


    # * Second ~ one before last row: sliding window search
    for i, similarity in enumerate(list_similarity):
        axes[i+1, 0].imshow(preprocess(Image.fromarray(list_selected_window[i]).convert("RGB")).permute(1,2,0))
        axes[i+1, 0].set_title(split_long_sentence(texts[i]))

        axes[i+1, 1].imshow(np.array(similarity).reshape(get_width_height(list_position[i])), cmap=plt.cm.viridis)
        axes[i+1, 1].set_title(f'{list_position[i][list_selected_index[i]]}, {similarity[0][list_selected_index[i]]}')

    # plt.suptitle(texts[0][0])


    # ! Then perform overlay grid search on the last selected window
    # * last row: "ipod" grid search
    last_similarity, last_images, idx_max_similarity, list_grid_position, (overlay_w, overlay_h) = single_image_grid_search(list_selected_window[-1], accumulated_texts[-1], model, tokenizer, 25, 25, safe=False)

    out_x += list_grid_position[idx_max_similarity][0]
    out_y += list_grid_position[idx_max_similarity][1]

    print(f"after grid overlay search: (x, y) = {(out_x + overlay_w/2, out_y + overlay_h/2)}")

    def get_coordinates(xlng, ylat,
                    full_w=4552,
                    full_h=2276):
        '''given lng lat returns coordinates in panorama image
        '''
        x = int(full_w * ((xlng + 180)/360.0))
        y = int(full_h - full_h * ((ylat + 90)/180.0))
        return x, y


    actual_x, actual_y = get_coordinates(99.12662544347826, -13.992525616350582)
    print(f"ground truth: (x, y) = {(actual_x, actual_y)}")

    flattened_last_images = []
    for i in last_images:
        flattened_last_images.extend(i)

    axes[-1, 0].imshow(preprocess(flattened_last_images[idx_max_similarity].convert("RGB")).permute(1,2,0))
    axes[-1, 0].set_title(split_long_sentence(texts[-1]))

    similarity_width = int(last_similarity.shape[1]**0.5)

    axes[-1, 1].imshow(np.array(last_similarity).reshape((similarity_width, similarity_width)), cmap=plt.cm.viridis)
    axes[-1, 1].set_title(last_similarity[0][list_selected_index[-1]])


    plt.tight_layout()
    plt.savefig('./result_images/greedy_search_1.png')
    fig.clf()

