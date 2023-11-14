from predict import Predictor
import os
import glob
from tqdm import tqdm
from san.data.datasets.register_cub import CLASS_NAMES

label_file = "datasets/CUB/valid_label.txt"
config_file = "configs/san_clip_vit_res4_coco.yaml"

def get_image_data_details(line):
    img_path = line.split(',')[0]
    label = int(line.split(',')[1].replace('\n', '').replace(' ', ''))
    output_file = os.path.join(args.output_dir, img_path.replace("test/","",).replace("/","_").replace(" ",""))
    img_path = os.path.join('datasets/CUB/', img_path.replace(' ', ''))
    return (img_path, label, output_file)


def load_pth_files(directory):
    pth_files = []
    # 指定されたディレクトリ内のファイルを走査
    for filename in os.listdir(directory):
        if filename.endswith(".pth"):
            pth_files.append(filename)
    pth_files.sort()
    return pth_files


def main(args):
    pth_files = load_pth_files(args.model_path)
    for pth_file in pth_files:
        print("predicting class ...")
        sample = 0
        correct = 0
        model_path = os.path.join(args.model_path, pth_file)
        predictor = Predictor(config_file=config_file, model_path=model_path)
        with open(label_file) as (f):
            lines = f.readlines()
            for line in tqdm(lines):
                img_path, label, output_file = get_image_data_details(line)
                pred_class = predictor.predict(
                    img_path,
                    [],
                    False,
                    output_file=output_file,
                    label=label,
                    predict_class=True
                )
                assert type(pred_class) == int
                if pred_class == int(label):
                    correct += 1
                sample += 1
        print(f"accuracy: {correct/sample}")
        print("ended predicting class ...")


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument(
        "--predict_mode", type=str, required=True, help="select from ['bird', 'name', 'class', 'none']"
    )

    parser.add_argument(
        "--model_path", type=str, required=True, help="path to model file"
    )

    parser.add_argument(
        "--output_dir", type=str, default=None, help="path to output file."
    )
    args = parser.parse_args()
    main(args)
