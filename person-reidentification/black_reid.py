import torch
import torch.nn as nn
import torchvision
from torchvision import models, transforms, utils
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import scipy.misc
from PIL import Image
import json

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=0., std=1.)
])

image = Image.open(str('sample_img2.jpg'))
plt.imshow(image)

model = models.resnet50(pretrained=True)
print(model)

model_weights = []
conv_layers = []
model_children = list(model.children())
counter = 0
for i in range(len(model_children)):
    if type(model_children[i]) == nn.Conv2d:
        counter += 1
        model_weights.append(model_children[i].weight)
        conv_layers.append(model_children[i])
    elif type(model_children[i]) == nn.Sequential:
        for j in range(len(model_children[i])):
            for child in model_children[i][j].children():
                if type(child) == nn.Conv2d:
                    counter += 1
                    model_weights.append(child.weight)
                    conv_layers.append(child)
print(f"Total convolution layers: {counter}")
print("conv_layers")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

image = transform(image)
print(f"Image shape before: {image.shape}")
image = image.unsqueeze(0)
print(f"Image shape after: {image.shape}")
image = image.to(device)

outputs = []
names = []
for layer in conv_layers[0:]:
    image = layer(image)
    outputs.append(image)
    names.append(str(layer))
print(len(outputs))
for feature_map in outputs:
    print(feature_map.shape)

processed = []
for feature_map in outputs:
    feature_map = feature_map.squeeze(0)
    gray_scale = torch.sum(feature_map, 0)
    gray_scale = gray_scale / feature_map.shape[0]
    processed.append(gray_scale.data.cpu().numpy())
for fm in processed:
    print(fm.shape)

fig = plt.figure(figsize=(30, 50))
for i in range(len(processed)):
    a = fig.add_subplot(13, 4, i + 1)
    imgplot = plt.imshow(processed[i])
    a.axis("off")
    a.set_title(names[i].split('(')[0], fontsize=30)
plt.savefig(str('feature_maps.jpg'), bbox_inches='tight')

class Counter:
    def __init__(self):
        self.counter = 0

    def increment(self):
        self.counter += 1

model = models.resnet50(weights=True)
model.eval()

conv_layers = []
model_weights = []
counter = Counter()

def hook_fn(module, input, output):
    if isinstance(module, nn.Conv2d):
        counter.increment()
        conv_layers.append(output)
        model_weights.append(module.weight)
    else:
        conv_layers.append(None)

model.apply(lambda module: hook_fn(module, None, None))

print(f"Total convolution layers: {counter.counter}")

input_image_path = 'sample_img2.jpg'
input_image = Image.open(input_image_path).convert('RGB')
preprocess = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
input_tensor = preprocess(input_image)
input_tensor = input_tensor.unsqueeze(0)

with torch.no_grad():
    _ = model(input_tensor)

fig = plt.figure(figsize=(30, 50))

for i, feature_map in enumerate(conv_layers):
    if feature_map is not None and isinstance(feature_map, torch.Tensor):
        feature_map = feature_map.squeeze(0)
        num_features = feature_map.size(0)

        processed_feature_maps = []
        for j in range(num_features):
            feature = feature_map[j, :, :]
            feature = feature.cpu().numpy()
            processed_feature_maps.append(feature)

        for j, fm in enumerate(processed_feature_maps):
            a = fig.add_subplot(counter.counter, num_features, i * num_features + j + 1)
            imgplot = plt.imshow(fm, cmap='viridis')
            a.axis("off")
            a.set_title(f'Layer: {i + 1}, Feature map: {j + 1}', fontsize=20)
plt.savefig(str('feature_maps.jpg'), bbox_inches='tight')
plt.show()

def hook_fn(module, input, output):
    if isinstance(module, nn.Conv2d) and output is not None and isinstance(output, torch.Tensor):
        counter.increment()
        conv_layers.append(output)

        feature_map = output.squeeze(0)
        num_features = feature_map.size(0)
        for j in range(num_features):
            feature = feature_map[j, :, :]
            feature = feature.cpu().numpy()
            plt.figure(figsize=(8, 8))
            plt.imshow(feature, cmap='viridis')
            plt.axis('off')
            plt.title(f'Layer: {counter.counter}, Feature map: {j + 1}', fontsize=20)
            plt.show()

model.apply(lambda module: hook_fn(module, None, None))

input_image_path = 'sample_img2.jpg'
input_image = Image.open(input_image_path).convert('RGB')
preprocess = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
input_tensor = preprocess(input_image)
input_tensor = input_tensor.unsqueeze(0)

with torch.no_grad():
    _ = model(input_tensor)
