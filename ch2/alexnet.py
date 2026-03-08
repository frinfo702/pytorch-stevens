# %%
import torch
from PIL import Image
from torchvision import models, transforms

# %%
weight = models.AlexNet_Weights
resnet = models.alexnet(weight)

preprocess = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),  # determine mean and std for 3 channels (RGB) respectively
    ]
)

# %%
img = Image.open("data/p1ch2/B10AFD79-3B07-4C33-8DA1-BAEA2E061D5D.jpeg")

img_t = preprocess(img)
batch_t = torch.unsqueeze(img_t, 0)

# %%
resnet.eval()
output = resnet(batch_t)

with open("data/p1ch2/imagenet_classes.txt") as f:
    labels = [line.strip() for line in f.readlines()]

_, indices = torch.sort(output, descending=True)
print(indices.shape)

percentage = torch.nn.functional.softmax(output, dim=1)[0] * 100
print([(labels[idx], percentage[idx].item()) for idx in indices[0][0:4]])
