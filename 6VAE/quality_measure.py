import numpy as np
import tensorflow as tf
import os
import pathlib
from PIL import Image
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from scipy.linalg import sqrtm

# 計算Inception Score（IS）
def calculate_inception_score(images, model, batch_size=32, splits=10):
    preds = []
    n_batches = int(np.ceil(float(len(images)) / float(batch_size)))

    for i in range(n_batches):
        start_idx = i * batch_size
        end_idx = (i + 1) * batch_size
        batch = images[start_idx:end_idx]
        batch = preprocess_input(batch)
        preds.append(model.predict(batch))

    preds = np.concatenate(preds, axis=0)
    scores = []
    for i in range(splits):
        part = preds[(i * preds.shape[0] // splits):((i + 1) * preds.shape[0] // splits), :]
        kl = part * (np.log(part) - np.log(np.expand_dims(np.mean(part, axis=0), 0)))
        kl = np.mean(np.sum(kl, axis=1))
        scores.append(np.exp(kl))
    return np.mean(scores), np.std(scores)

# 計算Fréchet Inception Distance（FID）
def calculate_fid(real_images, generated_images, model, batch_size=32):
    act1 = model.predict(real_images, batch_size=batch_size)
    act2 = model.predict(generated_images, batch_size=batch_size)
    mu1, sigma1 = np.mean(act1, axis=0), np.cov(act1, rowvar=False)
    mu2, sigma2 = np.mean(act2, axis=0), np.cov(act2, rowvar=False)
    ssdiff = np.sum((mu1 - mu2) ** 2.0)
    covmean = sqrtm(sigma1.dot(sigma2))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid

# 載入Inception模型
def load_inception_model():
    base_model = InceptionV3(include_top=True, pooling='avg', input_shape=(299, 299, 3))
    model = Model(inputs=base_model.input, outputs=base_model.layers[-1].output)
    return model

# 載入圖像
def load_images(image_paths, image_size=(299, 299)):
    images = []
    for path in image_paths:
        img = Image.open(path).resize(image_size)
        img = np.asarray(img)
        images.append(img)
    return np.array(images)

# 設定圖像路徑
real_image_paths = ['path/to/real/image1.jpg', 'path/to/real/image2.jpg', ...]
generated_image_paths = ['path/to/generated/image1.jpg', 'path/to/generated/image2.jpg', ...]

# 載入Inception模型
inception_model = load_inception_model()

# 載入圖像
real_images = load_images(real_image_paths)
generated_images = load_images(generated_image_paths)

# 計算並輸出Inception Score
is_mean, is_std = calculate_inception_score(generated_images, inception_model)
print("Inception Score: Mean = {}, Std = {}".format(is_mean, is_std))

# 計算並輸出Fréchet Inception Distance
fid = calculate_fid(real_images, generated_images, inception_model)
print("Fréchet Inception Distance:", fid)
