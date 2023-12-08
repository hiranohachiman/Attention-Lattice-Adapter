import json

old_path = "../datasets/CUB/test_label.jsonl"
new_path = "../datasets/CUB/new_test_label.jsonl"

def get_mask_path(img_path):
    return img_path.replace("/", "_").replace(".jpg", ".png").replace("test_", "extracted_test_segmentation_old/")


with open(old_path, 'r') as f:
    lines = f.readlines()
    datas = []
    for line in lines:
        line = json.loads(line)
        line["image_path"] = line["image_path"]
        line["label"] = line["label"]
        line["caption"] = line["caption"]
        line["mask_path"] = get_mask_path(line["image_path"])
        datas.append(line)

with open(new_path, 'w') as f:
    for data in datas:
        json.dump(data, f)
        f.write("\n")
