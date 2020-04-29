import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.optim import Adam
import torch
from skimage.segmentation import slic
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import argparse
import cv2
import sys
import os

from lime import lime_image
from pdb import set_trace

from Dataset import *
from Model import *


## 處理餵入參數
dataset_dir = sys.argv[1]
output_dir = sys.argv[2]
model_path = sys.argv[3]


#分別將 training set、validation set、testing set 用 readfile 函式讀進來
data_dir = dataset_dir
print("Reading data")

train_x, train_y = readfile(os.path.join(data_dir, "training"), True)
print("Size of training data = {}".format(len(train_x)))


## Create Model & Dataset
train_set = ImgDataset(train_x, train_y, test_transform)
model = Classifier().cuda()
checkpoint = torch.load(model_path)
model.load_state_dict(checkpoint)


## Saliency map
def normalize(image):
    return (image - image.min()) / (image.max() - image.min())

def compute_saliency_maps(x, y, model):
    model.eval()
    x = x.cuda()

    x.requires_grad_()

    y_pred = model(x)
    loss_func = torch.nn.CrossEntropyLoss()
    loss = loss_func(y_pred, y.cuda())
    loss.backward()

    saliencies = x.grad.abs().detach().cpu()
    saliencies = torch.stack([normalize(item) for item in saliencies])
    return saliencies

# 指定想要一起 visualize 的圖片 indices
img_indices = [83, 1500, 4500, 9000]
images, labels = train_set.getbatch(img_indices)
saliencies = compute_saliency_maps(images, labels, model)

# 使用 matplotlib 畫出來
fig, axs = plt.subplots(2, len(img_indices), figsize=(15, 8))

for col, index in enumerate(img_indices):
    axs[0][col].imshow(train_set.x[index][:, :, [2,1,0]])

for col, img in enumerate(saliencies):
    img = cv2.cvtColor(img.permute(1, 2, 0).numpy()[:, :, [2,1,0]], cv2.COLOR_BGR2RGB)
    axs[1][col].imshow(img)


plt.savefig(os.path.join(output_dir, "Saliency_Map.png"))
## Saliency map

## Filter explaination
layer_activations = None
def filter_explaination(x, model, cnnid, filterid, iteration=100, lr=1):
    model.eval()

    def hook(model, input, output):
        global layer_activations
        layer_activations = output
  
    hook_handle = model.cnn[cnnid].register_forward_hook(hook)
    model(x.cuda())

    filter_activations = layer_activations[:, filterid, :, :].detach().cpu()
    x = x.cuda()
    # 從一張 random noise 的圖片開始找 (也可以從一張 dataset image 開始找)
    x.requires_grad_()
    # 我們要對 input image 算偏微分
    optimizer = Adam([x], lr=lr)
    # 利用偏微分和 optimizer，逐步修改 input image 來讓 filter activation 越來越大
    for iter in range(iteration):
        optimizer.zero_grad()
        model(x)
    
        objective = -layer_activations[:, filterid, :, :].sum()

        objective.backward()
        # 計算 filter activation 對 input image 的偏微分
        optimizer.step()
    # 修改 input image 來最大化 filter activation
    filter_visualization = x.detach().cpu().squeeze()[0]
    # 完成圖片修改，只剩下要畫出來，因此可以直接 detach 並轉成 cpu tensor

    hook_handle.remove()

    return filter_activations, filter_visualization


img_indices = [7000, 7001, 100, 9001]
images, labels = train_set.getbatch(img_indices)
filter_activations, filter_visualization = filter_explaination(images, model, cnnid=5, filterid=0, iteration=100, lr=0.1)

# 畫出 filter visualization
plt.clf()
plt.imshow(normalize(filter_visualization.permute(1, 2, 0)))
plt.savefig(os.path.join(output_dir, "FE_filter.png"))


fig, axs = plt.subplots(2, len(img_indices), figsize=(15, 8))

for col, index in enumerate(img_indices):
    axs[0][col].imshow(train_set.x[index][:, :, [2,1,0]])

for i, img in enumerate(filter_activations):
    img = cv2.cvtColor(normalize(img).numpy(), cv2.COLOR_BGR2RGB)
    axs[1][i].imshow(normalize(img))


plt.savefig(os.path.join(output_dir, "FE_example.png")) 
## Filter explaination



## Lime
def predict(input):
    # input: numpy array, (batches, height, width, channels)
    
    model.eval()
    input = torch.FloatTensor(input).permute(0, 3, 1, 2)
    # 需要先將 input 轉成 pytorch tensor，且符合 pytorch 習慣的 dimension 定義
    # 也就是 (batches, channels, height, width)

    output = model(input.cuda())
    return output.detach().cpu().numpy()

def segmentation(input):
    # 利用 skimage 提供的 segmentation 將圖片分成 100 塊
    return slic(input, n_segments=100, compactness=1, sigma=1)


img_indices = [2000, 3001, 5002, 5003]

images, labels = train_set.getbatch(img_indices)
fig, axs = plt.subplots(2, 4, figsize=(15, 8))
np.random.seed(16)

mask = None
for idx, (image, label) in enumerate(zip(images.permute(0, 2, 3, 1).numpy(), labels)):
    x = image.astype(np.double)
    
    label = label.cuda()
    
    explainer = lime_image.LimeImageExplainer()
    explaination = explainer.explain_instance(image=x, classifier_fn=predict, segmentation_fn=segmentation, top_labels=11)

    lime_img, mask = explaination.get_image_and_mask(                                                                                                                         
                                label=label.item(),                                                                                                                           
                                positive_only=False,                                                                                                                         
                                hide_rest=False,                                                                                                                             
                                num_features=11,                                                                                                                              
                                min_weight=0.05                                                                                                                              
                            )
    
    img = train_set.x[img_indices[idx]][:, :, [2,1,0]]
    mask = mask.reshape((128,128,1)).astype('uint8')
    axs[0][idx].imshow(img)
    axs[1][idx].imshow(cv2.bitwise_and(img, img, mask = mask))

plt.savefig(os.path.join(output_dir, "LIME_example.png"))
## Lime

## Smooth Grad
def func(module, grad_in, grad_out):
    # Pass only positive gradients
    if isinstance(module, nn.ReLU):
        return (torch.clamp(grad_in[0], min=0.0),)

train_set = ImgDataset(train_x, train_y, test_transform)
model = Classifier().cuda()
checkpoint = torch.load(model_path)
model.load_state_dict(checkpoint)

# 指定想要一起 visualize 的圖片 indices
from torch.autograd import Variable
img_indices = [4000, 7001, 3002, 8005]
images, labels = train_set.getbatch(img_indices)
n_samples = 200
ratio = 0.1


for module in model.named_modules():
    module[1].register_backward_hook(func)
    
    
sigma = (image.max() - image.min()) * ratio
grads = []
for i in range(n_samples):
    noised_image = images + torch.randn(images.size()) * sigma
    noised_image = noised_image.cuda()
    noised_image = Variable(noised_image, volatile=False, requires_grad=True)

    saliencies = compute_saliency_maps(noised_image, labels, model)
        
    grads.append(saliencies.permute(0, 2, 3, 1).numpy())
    
grads = np.asarray(grads)

fig, axs = plt.subplots(2, len(img_indices), figsize=(15, 8))
for col, index in enumerate(img_indices):
    axs[0][col].imshow(train_set.x[index][:, :, [2,1,0]])

for i in range(len(img_indices)):
    tmp_1 = grads[:, i, :, :]
    sgimg = np.nanmean(np.nanmean(tmp_1, axis=0), axis=2)
    axs[1][i].imshow(sgimg, cmap='gray', vmin=np.min(sgimg), vmax=np.max(sgimg))

plt.savefig(os.path.join(output_dir, "SmoothGrad_ex.png"))
## Smooth Grad