import cv2
import os

img_path = "datasets/ImageNetS919/extracted_validation/ILSVRC2012_val_00036987.JPEG"
attn_path = "output/test2/gradient_backprop_attn/ILSVRC2012_val_00036987.JPEG"
output_path = "output/test2/teisei"

img = cv2.imread(img_path)
img = cv2.resize(img, (384, 384), interpolation=cv2.INTER_LINEAR)
attn = cv2.imread(attn_path, cv2.IMREAD_GRAYSCALE)
attn = cv2.resize(attn, (384, 384), interpolation=cv2.INTER_LINEAR)
attn = cv2.applyColorMap(attn, cv2.COLORMAP_JET)

# 画像の重ね合わせ
alpha = 0.5
beta = 1.0 - alpha
gamma = 0.0
img = cv2.addWeighted(img, alpha, attn, beta, gamma)

cv2.imwrite(os.path.join(output_path, f"GuidedBP_{os.path.basename(img_path)}"), img)
