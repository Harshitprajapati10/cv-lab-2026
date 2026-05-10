import numpy as np
import matplotlib.pyplot as plt

def tanh(z):
    return np.tanh(z)

def dtanh(a):
    return 1 - a**2

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def dsigmoid(a):
    return a * (1 - a)

def relu(z):
    return np.maximum(0, z)

def drelu(a):
    return (a > 0).astype(float)

ACT = {
    "tanh": (tanh, dtanh),
    "sigmoid": (sigmoid, dsigmoid),
    "relu": (relu, drelu)
}

def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def one_hot(y, num_classes):
    oh = np.zeros((len(y), num_classes))
    oh[np.arange(len(y)), y] = 1
    return oh

def init_params(layers):
    params = []

    for i in range(1, len(layers)):

        # Xavier Initialization
        limit = np.sqrt(1 / layers[i - 1])

        W = np.random.randn(layers[i], layers[i - 1]) * limit
        b = np.zeros((layers[i], 1))

        params.append({
            "W": W,
            "b": b
        })

    return params

def forward(X, params, activations):
    A = X
    cache = [{"A": X}]
    for l in range(len(params)):
        W = params[l]["W"]
        b = params[l]["b"]
        Z = A @ W.T + b.T
        if l == len(params) - 1:
            A = softmax(Z)
        else:
            act_fn = ACT[activations[l]][0]
            A = act_fn(Z)
        cache.append({
            "Z": Z,
            "A": A
        })
    return A, cache

def loss(y, yhat):
    eps = 1e-8
    yhat = np.clip(yhat, eps, 1 - eps)
    return -np.mean(np.sum(y * np.log(yhat), axis=1))

def backward(y, yhat, params, cache, activations):
    grads = []
    m = y.shape[0]
    dZ = yhat - y
    for l in reversed(range(len(params))):
        A_prev = cache[l]["A"]
        W = params[l]["W"]
        dW = (1 / m) * (dZ.T @ A_prev)
        db = (1 / m) * np.sum(dZ, axis=0, keepdims=True).T
        grads.insert(0, {
            "dW": dW,
            "db": db
        })
        if l > 0:
            dA_prev = dZ @ W
            A_prev_act = cache[l]["A"]
            deriv_fn = ACT[activations[l - 1]][1]
            dZ = dA_prev * deriv_fn(A_prev_act)
    return grads

def update(params, grads, lr):
    for l in range(len(params)):
        params[l]["W"] -= lr * grads[l]["dW"]
        params[l]["b"] -= lr * grads[l]["db"]
    return params

def metrics(y_true, y_pred):
    y_true_cls = np.argmax(y_true, axis=1)
    y_pred_cls = np.argmax(y_pred, axis=1)
    acc = np.mean(y_true_cls == y_pred_cls)
    num_classes = y_true.shape[1]
    precisions = []
    recalls = []
    f1s = []
    for c in range(num_classes):
        TP = np.sum((y_pred_cls == c) & (y_true_cls == c))
        FP = np.sum((y_pred_cls == c) & (y_true_cls != c))
        FN = np.sum((y_pred_cls != c) & (y_true_cls == c))
        precision = TP / (TP + FP + 1e-8)
        recall = TP / (TP + FN + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
    return (
        acc,
        np.mean(precisions),
        np.mean(recalls),
        np.mean(f1s)
    )

def train(X,y,layers,activations,epochs=100,lr=0.01):
    params = init_params(layers)
    losses = []
    for i in range(epochs):
        yhat, cache = forward(X, params, activations)
        l = loss(y, yhat)
        losses.append(l)
        grads = backward(y, yhat, params, cache, activations)
        params = update(params, grads, lr)
    yhat, _ = forward(X, params, activations)
    acc, p, r, f1 = metrics(y, yhat)
    return params, losses, acc, p, r, f1

if __name__ == "__main__":
    np.random.seed(42)

    samples = 10000
    features = 600
    classes = 5
    X = np.random.randn(samples, features)
    y_raw = np.random.randint(0, classes, size=samples)
    y = one_hot(y_raw, classes)

    layers = [600, 20, 10, 5]

    learning_rates = [0.001, 0.01, 0.05, 0.1, 0.5]

    epoch_list = [50, 100, 200, 500]

    activations_to_test = ["tanh", "relu", "sigmoid"]

    for activation_name in activations_to_test:

        print("\n")
        print("=" * 70)
        print(f"ACTIVATION FUNCTION : {activation_name.upper()}")
        print("=" * 70)

        plt.figure(figsize=(12, 8))

        experiment_no = 1

        for lr in learning_rates:

            for epochs in epoch_list:

                print("\n")
                print("-" * 60)

                print(
                    f"Experiment {experiment_no} | "
                    f"Activation={activation_name} | "
                    f"LR={lr} | Epochs={epochs}"
                )

                activations = [activation_name, activation_name]

                params, losses, acc, p, r, f1 = train(
                    X,
                    y,
                    layers,
                    activations,
                    epochs=epochs,
                    lr=lr
                )

                print(f"Final Loss     : {losses[-1]:.6f}")
                print(f"Accuracy       : {acc:.6f}")
                print(f"Precision      : {p:.6f}")
                print(f"Recall         : {r:.6f}")
                print(f"F1 Score       : {f1:.6f}")

                plt.plot(
                    losses,
                    label=f"lr={lr}, ep={epochs}"
                )

                experiment_no += 1

        plt.title(
            f"Convergence Behaviour - {activation_name.upper()}"
        )

        plt.xlabel("Epoch")

        plt.ylabel("Loss")

        plt.legend(fontsize=8)

        plt.grid(True)

        plt.show()

samples = 200
features = 100
classes = 10
X = np.random.randn(samples, features)
y_raw = np.random.randint(0, classes, size=samples)
y = one_hot(y_raw, classes)
layers = [100, 20, 10]

activations = ["tanh"]
params, losses, acc, p, r, f1 = train(
    X,
    y,
    layers,
    activations,
    epochs=100,
    lr=0.01
)
print("Loss:", losses[-1])
print("Accuracy:", acc)
print("Precision:", p)
print("Recall:", r)
print("F1:", f1)


# FILE_PATH = "CUB_200_2011.tgz"
# !tar -xvzf "{FILE_PATH}" -C .

import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.cluster import MeanShift

img = cv2.imread(
    "CUB_200_2011/images/005.Crested_Auklet/Crested_Auklet_0070_785261.jpg"
)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.imshow(img)
plt.axis("off")
plt.show()

img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_blur = cv2.GaussianBlur(img_gray, (3,3), 0)
edges = cv2.Canny(
    image=img_blur,
    threshold1=100,
    threshold2=200
)

fig, axes = plt.subplots(1, 3, figsize=(15,5))
axes[0].imshow(img_gray, cmap='gray')
axes[0].set_title("Gray Image")
axes[0].axis("off")

axes[1].imshow(img_blur, cmap='gray')
axes[1].set_title("Blurred Image")
axes[1].axis("off")

axes[2].imshow(edges, cmap='gray')
axes[2].set_title("Canny Edges")
axes[2].axis("off")

plt.tight_layout()
plt.show()

#segmentation
ret, thresh = cv2.threshold(
    img_gray,
    0,
    255,
    cv2.THRESH_BINARY + cv2.THRESH_OTSU
)

plt.figure(figsize=(6,6))
plt.imshow(thresh, cmap='gray')
plt.title("Otsu Threshold Segmentation")
plt.axis("off")
plt.show()

#thresholding
test_img = cv2.imread("CUB_200_2011/images/065.Slaty_backed_Gull/Slaty_Backed_Gull_0003_796032.jpg")
image_rgb = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)
test_img_gray = cv2.cvtColor(test_img,cv2.COLOR_BGR2GRAY)

plt.figure(figsize=(12,6))
plt.subplot(231)
plt.imshow(test_img)
plt.title('original image')
plt.axis('off')

_, thresh = cv2.threshold(test_img_gray, 127, 255, cv2.THRESH_BINARY)
plt.subplot(232)
plt.imshow(thresh)
plt.title('thresholding')
plt.axis('off')

# kmeans
pixel_values = image_rgb.reshape((-1,3))
pixel_values = np.float32(pixel_values)
print(pixel_values.shape)

k = 3
kmeans = KMeans(n_clusters=k)
labels = kmeans.fit_predict(pixel_values)

centers = np.uint8(kmeans.cluster_centers_)
segmented = centers[labels.flatten()]
segmented_image = segmented.reshape(image_rgb.shape)

plt.subplot(231)
plt.imshow(segmented_image)
plt.title('k-means')
plt.axis('off')

# edge detection
edges = cv2.Canny(test_img_gray, 100, 200)
plt.subplot(233)
plt.imshow(edges, cmap='gray')
plt.title('edge detection')
plt.axis('off')

# watershed
img = test_img.copy()
gray_ws = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, thresh_ws = cv2.threshold(gray_ws,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

kernel = np.ones((3,3),np.uint8)
opening = cv2.morphologyEx(thresh_ws, cv2.MORPH_OPEN, kernel, iterations=2)

sure_bg = cv2.dilate(opening,kernel,iterations=3)

dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
_, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)

sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg,sure_fg)

_, markers = cv2.connectedComponents(sure_fg)

markers = markers + 1
markers[unknown==255] = 0

markers = cv2.watershed(img,markers)
img[markers == -1] = [255,0,0]

plt.subplot(235)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title("Watershed Segmentation")
plt.axis("off")

# edge detection roberts
# Roberts Cross kernels
roberts_cross_v = np.array([
    [1, 0],
    [0,-1]
], dtype=np.float32)

roberts_cross_h = np.array([
    [0, 1],
    [-1,0]
], dtype=np.float32)

# Apply filters
vertical = cv2.filter2D(
    test_img_gray,
    -1,
    roberts_cross_v
)

horizontal = cv2.filter2D(
    test_img_gray,
    -1,
    roberts_cross_h
)

roberts = cv2.addWeighted(
    vertical,
    0.5,
    horizontal,
    0.5,
    0
)

plt.figure(figsize=(6,6))
plt.imshow(roberts, cmap='gray')
plt.title("Roberts Edge Detection")
plt.axis("off")
plt.show()

# Prewitt kernels
kernelx = np.array([
    [ 1, 0,-1],
    [ 1, 0,-1],
    [ 1, 0,-1]
], dtype=np.float32)

kernely = np.array([
    [ 1, 1, 1],
    [ 0, 0, 0],
    [-1,-1,-1]
], dtype=np.float32)

# Apply filters
prewitt_x = cv2.filter2D(
    test_img_gray,
    -1,
    kernelx
)

prewitt_y = cv2.filter2D(
    test_img_gray,
    -1,
    kernely
)

# Combine gradients
prewitt = cv2.addWeighted(
    prewitt_x,
    0.5,
    prewitt_y,
    0.5,
    0
)

# Display result
plt.figure(figsize=(6,6))
plt.imshow(prewitt, cmap='gray')
plt.title("Prewitt Edge Detection")
plt.axis("off")
plt.show()

# Sobel gradients
sobelx = cv2.Sobel(
    test_img_gray,
    cv2.CV_64F,
    1,
    0,
    ksize=3
)

sobely = cv2.Sobel(
    test_img_gray,
    cv2.CV_64F,
    0,
    1,
    ksize=3
)

# Gradient magnitude
sobel = cv2.magnitude(sobelx, sobely)

# Display
plt.figure(figsize=(6,6))
plt.imshow(sobel, cmap='gray')
plt.title("Sobel Edge Detection")
plt.axis("off")
plt.show()

# Laplacian Edge Detection
laplacian = cv2.Laplacian(
    test_img_gray,
    cv2.CV_64F
)

# Display
plt.figure(figsize=(6,6))
plt.imshow(laplacian, cmap='gray')
plt.title("Laplacian Edge Detection")
plt.axis("off")
plt.show()

# Canny Edge Detection
canny = cv2.Canny(
    test_img_gray,
    100,
    200
)

# Display
plt.figure(figsize=(6,6))
plt.imshow(canny, cmap='gray')
plt.title("Canny Edge Detection")
plt.axis("off")
plt.show()


import torch
import torchvision.models as models
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torchvision import transforms
from numpy.fft import fft2, fftshift

model = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
print(model)

print(model.features)
layer = model.features[0]
filters = layer.weight.data.clone()
print(filters.shape)

#plot 8 filters
import random

fig, axes = plt.subplots(2, 4, figsize=(10,5))
random_indices = random.sample(range(filters.shape[0]), 8)

for ax, idx in zip(axes.flat, random_indices):
    f = filters[idx]

    # Convert CHW → HWC
    f = f.permute(1,2,0)

    # Normalize
    f = (f - f.min()) / (f.max() - f.min())

    ax.imshow(f)
    ax.set_title(f'Filter {idx}')
    ax.axis('off')

plt.tight_layout()
plt.show()

# plot frequency map
fig, axes = plt.subplots(2,4, figsize=(12,6))

for ax, idx in zip(axes.flat, random_indices):
    kernel = filters[idx,0].cpu().numpy()

    freq = np.abs(
        fftshift(
            fft2(kernel, s=(32,32))
        )
    )

    ax.imshow(freq, cmap='gray')
    ax.set_title(f'Freq {idx}')
    ax.axis('off')

plt.tight_layout()
plt.show()

import os
import random
import torch
import torchvision.models as models
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

weights = models.VGG16_Weights.IMAGENET1K_V1
model = models.vgg16(weights=weights)
model.eval()

transform = weights.transforms()
imagenet_classes = weights.meta["categories"]

dataset_path = "CUB_200_2011"
classes_file = os.path.join(dataset_path, "classes.txt")

cub_classes = {}

with open(classes_file, "r") as f:
    for line in f:
        idx, name = line.strip().split(" ", 1)
        cub_classes[int(idx)] = name

images_txt = os.path.join(dataset_path, "images.txt")
labels_txt = os.path.join(dataset_path, "image_class_labels.txt")

image_paths = {}
image_labels = {}

with open(images_txt) as f:
    for line in f:
        idx, path = line.strip().split(" ", 1)
        image_paths[int(idx)] = path

with open(labels_txt) as f:
    for line in f:
        idx, label = line.strip().split(" ")
        image_labels[int(idx)] = int(label)

random_classes = random.sample(list(cub_classes.keys()), 8)
selected_images = []

for cls in random_classes:
    candidates = [
        idx for idx, lbl in image_labels.items()
        if lbl == cls
    ]

    img_id = random.choice(candidates)
    selected_images.append((cls, img_id))

results = []

for cls, img_id in selected_images:
    img_path = os.path.join(
        dataset_path,
        "images",
        image_paths[img_id]
    )

    img = Image.open(img_path).convert("RGB")

    x = transform(img).unsqueeze(0)

    with torch.no_grad():
        out = model(x)

    pred_idx = out.argmax(1).item()
    pred_class = imagenet_classes[pred_idx]
    true_class = cub_classes[cls]

    results.append((img, true_class, pred_class))

fig, axes = plt.subplots(4, 2, figsize=(12,16))

for ax, (img, true_cls, pred_cls) in zip(axes.flat, results):
    ax.imshow(img)

    ax.set_title(
        f"CUB: {true_cls}\nImageNet: {pred_cls}",
        fontsize=9
    )

    ax.axis("off")

plt.tight_layout()
plt.show()


import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torchvision.models as models
import pytorch_lightning as pl
import torchmetrics
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning import Trainer
from torch.utils.data import random_split

class CUBDataset(Dataset):
    def __init__(self, root_dir, train=True, transform=None):
        self.root = root_dir
        self.transform = transform

        images = pd.read_csv(os.path.join(root_dir,"images.txt"), sep=" ", names=["img_id","path"])
        labels = pd.read_csv(os.path.join(root_dir,"image_class_labels.txt"), sep=" ", names=["img_id","label"])
        split = pd.read_csv(os.path.join(root_dir,"train_test_split.txt"), sep=" ", names=["img_id","is_train"])

        data = images.merge(labels, on="img_id").merge(split, on="img_id")

        if train:
            data = data[data.is_train==1]
        else:
            data = data[data.is_train==0]

        self.data = data.reset_index(drop=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        img_path = os.path.join(self.root,"images",row.path)
        label = row.label - 1

        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, label

class CUBDataModule(pl.LightningDataModule):

    def __init__(self, data_dir, batch_size=32):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size

        self.transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],
                                 [0.229,0.224,0.225])
        ])

    def setup(self, stage=None):

        full_train = CUBDataset(self.data_dir, train=True, transform=self.transform)

        train_size = int(0.8 * len(full_train))
        val_size = len(full_train) - train_size

        self.train_dataset, self.val_dataset = random_split(
            full_train, [train_size, val_size]
        )

        self.test_dataset = CUBDataset(
            self.data_dir, train=False, transform=self.transform
        )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=32, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=32)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=32)

class VGG16FineTune(pl.LightningModule):

    def __init__(self, num_classes=200, lr=1e-4):
        super().__init__()

        self.lr = lr

        self.model = models.vgg16(pretrained=True)

        # freeze early layers
        for param in self.model.features[:20].parameters():
            param.requires_grad = False

        # replace classifier
        self.model.classifier[6] = nn.Linear(4096, num_classes)

        self.loss = nn.CrossEntropyLoss()

        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.test_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)

    def forward(self,x):
        return self.model(x)

    def training_step(self,batch,batch_idx):
        x,y = batch
        logits = self(x)

        loss = self.loss(logits,y)

        preds = torch.argmax(logits,dim=1)

        acc = self.train_acc(preds,y)

        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):

      x, y = batch

      logits = self(x)

      loss = self.loss(logits, y)

      preds = torch.argmax(logits, dim=1)

      acc = self.val_acc(preds, y)

      self.log("val_loss", loss, prog_bar=True)
      self.log("val_acc", acc, prog_bar=True)

    def test_step(self,batch,batch_idx):
        x,y = batch
        logits = self(x)

        loss = self.loss(logits,y)

        preds = torch.argmax(logits,dim=1)

        acc = self.test_acc(preds,y)

        self.log("test_loss",loss, prog_bar=True)
        self.log("test_acc",acc, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.1,
            patience=3
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss"
        }

data_dir = "CUB_200_2011"

data_module = CUBDataModule(data_dir, batch_size=32)

model = VGG16FineTune()

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    mode="min"
)

checkpoint = ModelCheckpoint(
    monitor="val_acc",
    mode="max",
    save_top_k=1,
    filename="best-vgg16-cub"
)

trainer = Trainer(
    max_epochs=50,
    accelerator="auto",
    callbacks=[early_stop, checkpoint],
    log_every_n_steps=10
)

trainer.fit(model, datamodule=data_module)

# !find -name best-vgg16-cub.ckpt

model = VGG16FineTune.load_from_checkpoint("./lightning_logs/version_4/checkpoints/best-vgg16-cub.ckpt")

trainer.test(model, datamodule=data_module)