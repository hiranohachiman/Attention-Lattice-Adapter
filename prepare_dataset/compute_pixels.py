from PIL import Image
import numpy as np
import os
import json
from tqdm import tqdm

paths = []
# 画像パスが保存されているファイルの読み込み
with open('../datasets/CUB/train_label.jsonl', 'r') as file:
    for line in file:
        data = json.loads(line)
        paths.append(data["image_path"])
# 各チャンネルの合計値と画像の数を初期化
total = np.array([0.0, 0.0, 0.0])
total_squared = np.array([0.0, 0.0, 0.0])
count = 0


# 画像を1つずつ読み込み、RGBチャンネルの平均を計算
root_dir = "../datasets/CUB"
for path in tqdm(paths):
    path = os.path.join(root_dir, path)
    try:
        with Image.open(path) as img:
            # 画像をnumpy配列に変換
            np_img = np.array(img).astype(np.float32)
            # RGBチャンネルの合計と二乗の合計を計算
            total += np_img.mean(axis=(0, 1))
            total_squared += (np_img ** 2).mean(axis=(0, 1))
            count += 1
    except Exception as e:
        print(f"Error loading image {path}: {e}")

# 各チャンネルの平均値と標準偏差を計算
if count > 0:
    pixel_mean = total / count / 255
    pixel_std = np.sqrt(total_squared / count - pixel_mean ** 2) / 255
    print("Pixel Mean (RGB):", pixel_mean)
    print("Pixel Std (RGB):", pixel_std)
else:
    print("No images processed.")