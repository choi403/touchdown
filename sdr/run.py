import numpy as np
import torch
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
import clip_beam_search as bs
from PIL import Image
import matplotlib.pyplot as plt
import json

from tqdm import tqdm
from clip_beam_search import SimpleTokenizer, greedy_search_sliding_window, single_image_grid_search


test_seen_filename = '../data/data_refer360/test.seen.json'

def get_coordinates(xlng, ylat,
                full_w=4552,
                full_h=2276):
    '''given lng lat returns coordinates in panorama image
    '''
    x = int(full_w * ((xlng + 180)/360.0))
    y = int(full_h - full_h * ((ylat + 90)/180.0))
    return x, y

def load_refer360_data(filename):

    all_data = dict()

    with open(filename) as f:
        data = json.load(f)

    list_refexp = [i['refexp'] for i in data]
    list_positions = [(i['xlng_deg'], i['ylat_deg']) for i in data]
    list_coordinates = [get_coordinates(i['xlng_deg'], i['ylat_deg']) for i in data]
    list_filenames = [i['img_src'] for i in data]
    list_categories = [i['img_cat'] for i in data]
    list_locations = [i['img_loc'] for i in data]

    for i, l in enumerate(list_refexp):
        list_refexp[i] = [' '.join(tokens) for tokens in l]

    for i, e in enumerate(list_refexp):
        temp = dict()
        temp['refexp'] = list_refexp[i]
        temp['coordinates'] = list_coordinates[i]
        temp['filename'] = list_filenames[i]
        
        all_data[list_filenames[i]] = temp

    return all_data, all_data.keys()

    

# ! Load dataset

list_data_dict, list_panorama_filename = load_refer360_data(test_seen_filename)


# ! Load model
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



# ! Perform greedy search
# ^ Parameters
list_gs_step = [80, 80, 80, 80, 80]
list_gs_window_size = [(800, 800), (700, 700), (600, 600), (500, 500), (400, 400)]
ov_step_size = (30, 30)

list_predictions = []
list_answers = []

for fn in tqdm(list_panorama_filename):
    pred_x, pred_y = 0, 0

    image_pil = Image.open(fn).convert('RGB')

    list_original_text_prompt = list_data_dict[fn]['refexp']

    (xlng_deg, ylat_deg) = list_data_dict[fn]['coordinates']

    list_accumulated_text_prompt = [' '.join(list_original_text_prompt[:i+1]) for (i, e) in enumerate(list_original_text_prompt)]
    list_text_prompt = list_accumulated_text_prompt

    image_np = np.array(image_pil)

    # * Greedy search 
    gs_output = greedy_search_sliding_window(
        image_np,
        list_text_prompt, 
        list_gs_step, 
        list_gs_window_size, 
        model, 
        tokenizer
        )

    list_gs_similarity, list_gs_selected_window, list_gs_position, list_gs_selected_index, (gs_pred_x, gs_pred_y) = gs_output

    pred_x += gs_pred_x
    pred_y += gs_pred_y

    # * Overlayed grid search
    ov_output = single_image_grid_search(
        list_gs_selected_window[-1], 
        list_text_prompt[-1], 
        model, 
        tokenizer, 
        ov_step_size[0], 
        ov_step_size[1], 
        safe=False)

    list_ov_similarity, list_ov_input_images, ov_idx_max_similarity, ov_list_grid_position, (ov_pred_w, ov_pred_h) = ov_output

    pred_x += ov_list_grid_position[ov_idx_max_similarity][0]
    pred_y += ov_list_grid_position[ov_idx_max_similarity][1]

    if True:
        pred_x += ov_pred_w / 2
        pred_y += ov_pred_h / 2
    
    print(f'pred: {(pred_x, pred_y)}, ans: {(xlng_deg, ylat_deg)}')

    list_predictions.append((pred_x, pred_y))
    list_answers.append((xlng_deg, ylat_deg))
        












