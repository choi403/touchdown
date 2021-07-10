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

  


# texts = [
#             [          
#                 'Two parked bicycles, and a discarded couch, all on the left.',
#             ],
#             [          
#                 'Walk just past this couch, and stop before you pass another parked bicycle.',
#             ],
#             [          
#                 'This bike will be white and red, with a white seat.', 
#             ],
#             [          
#                 'Touchdown is sitting on top of the bike seat.',
#             ],
#             [
#                 'This is image 2'
#             ]
# ] 

texts = [['look for the red beverage cooler or refrigerator to the left of the man setting the table.'], ['look to the bottom left of the cooler.'], ['waldo is at the bottom left, slightly off the ground.']]
texts = [['look for the red beverage cooler or refrigerator to the left of the man setting the table. look to the bottom left of the cooler. Yellow star is at the bottom left, slightly off the ground.'], ['']]

tokenizer = bs.SimpleTokenizer()

# i1 = Image.open('./test_images/one_sentence/1.png').convert('RGB')
# i2 = Image.open('./test_images/one_sentence/2.png').convert('RGB')
# i3 = Image.open('./test_images/one_sentence/3.png').convert('RGB')
# input_images= [[i1, i2, i3]]

input_image = Image.open('./test_images/one_sentence/right.png').convert('RGB')

# similarity_1, images_nested_1, similarity_2, images_nested_2 = bs.greedy_search_custom_images(input_images, texts, model, tokenizer)
# similarity_1, images_nested_1 = bs.greedy_search_custom_images(input_images, texts, model, tokenizer)
similarity_1, images_nested_1 = bs.single_image_grid_search(input_image, texts, model, tokenizer, 20, 20)





# total = sum(similarity_1[0])
# for i in range(2):
#     similarity_1[0][0] = similarity_1[0][0] / total
#     similarity_1[0][1] = similarity_1[0][1] / total
# print(similarity_1, total)

preprocess = Compose([
    Resize(model.input_resolution.item(), interpolation=Image.BICUBIC),
    CenterCrop(model.input_resolution.item()),
    ToTensor()
])

images = []

num_rows = len(images_nested_1)
num_columns = len(images_nested_1[0])

for i in range(num_rows):
    for j in range(num_columns):
        image = preprocess(images_nested_1[i][j].convert("RGB"))
        images.append(image)

fig, axes = plt.subplots(num_rows, num_columns)
fig.set_size_inches(40, 40)

if num_rows != 1:
    for i in range(num_rows):
        for j in range(num_columns):
            axes[i,j].imshow(preprocess(images_nested_1[i][j].convert("RGB")).permute(1,2,0))
            axes[i,j].set_title(similarity_1[0][i * num_rows + j])
else:
    for j in range(num_columns):
        axes[j].imshow(preprocess(images_nested_1[0][j].convert("RGB")).permute(1,2,0))
        axes[j].set_title(similarity_1[0][j])

plt.suptitle(texts[0][0])
plt.tight_layout()

plt.savefig('./result_images/out1.png')
fig.clf()



# preprocess = Compose([
#     Resize(model.input_resolution.item(), interpolation=Image.BICUBIC),
#     CenterCrop(model.input_resolution.item()),
#     ToTensor()
# ])

# images = []

# num_rows = len(images_nested_2)
# num_columns = len(images_nested_2[0])

# print(num_rows, num_columns)

# for i in range(num_rows):
#     for j in range(num_columns):
#         image = preprocess(images_nested_2[i][j].convert("RGB"))
#         images.append(image)

# fig, axes = plt.subplots(num_rows, num_columns)
# fig.set_size_inches(20, 10)

# if num_rows != 1:
#     for i in range(num_rows):
#         for j in range(num_columns):
#             axes[i,j].imshow(preprocess(images_nested_2[i][j].convert("RGB")).permute(1,2,0))
#             axes[i,j].set_title(similarity_2[0][i * num_rows + j])
# else:
#     for j in range(num_columns):
#         axes[j].imshow(preprocess(images_nested_2[0][j].convert("RGB")).permute(1,2,0))
#         axes[j].set_title(similarity_2[0][j])

# plt.savefig('./result_images/out2.png')