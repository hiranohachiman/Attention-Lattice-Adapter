from predict import Predictor
import os
import glob
from tqdm import tqdm
from san.data.datasets.register_cub import CLASS_NAMES

label_file = "datasets/CUB/id_score_sample.txt"
config_file = "configs/san_clip_vit_res4_coco.yaml"

def get_img_path(img_dir):
    img_types = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.gif', '*.tiff']
    img_paths = []
    for img_type in img_types:
        img_paths.extend(glob.glob(os.path.join(img_dir, img_type)))
    return img_paths

def get_label_name(label):
    classes = CLASS_NAMES
    return classes[int(label)]

def get_image_data_details(line):
    img_path = line.split(',')[0]
    label = int(line.split(',')[1].replace('\n', '').replace(' ', ''))
    output_file = os.path.join(args.output_dir, img_path.replace("test/","",).replace("/","_").replace(" ",""))
    img_path = os.path.join('datasets/CUB/', img_path.replace(' ', ''))
    return (img_path, label, output_file)


def predict_with_bird(args):
    predictor = Predictor(config_file=config_file, model_path=args.model_path)
    print("predicting with word \"bird\" ... ")
    with open(label_file) as (f):
        lines = f.readlines()
        for line in tqdm(lines):
            img_path, label, output_file = get_image_data_details(line)
            pred_class = predictor.predict(
                img_path,
                ["bird"],
                False,
                output_file=output_file,
                label=label,
            )
    print("ended predicting with word \"bird\" ... ")


def predict_with_name(args):
    predictor = Predictor(config_file=config_file, model_path=args.model_path)
    print("predicting with word the bird gt name ... ")
    with open(label_file) as (f):
        lines = f.readlines()
        for line in tqdm(lines):
            img_path, label, output_file = get_image_data_details(line)
            label_name = get_label_name(label)
            pred_class = predictor.predict(
                img_path,
                [label_name],
                False,
                output_file=output_file,
                label=label,
            )
    print("ended predicting with word the bird gt name ... ")

def predict_without_vocab(args):
    predictor = Predictor(config_file=config_file, model_path=args.model_path)
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
            )


def predict_class(args):
    print("predicting class ...")
    sample = 0
    correct = 0
    predictor = Predictor(config_file=config_file, model_path=args.model_path)
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

def main(args):
    if args.predict_mode == "bird":
        predict_with_bird(args)
    elif args.predict_mode == "none":
        predict_without_vocab(args)
    elif args.predict_mode == "name":
        predict_with_name(args)
    else:
        predict_class(args)

if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument(
        "--predict_mode", type=str, required=True, help="select from ['bird', 'name', 'class', 'none']"
    )

    parser.add_argument(
        "--model_path", type=str, required=True, help="path to model file"
    )
    # parser.add_argument(
    #     '--img_dir', type=str, required=True, help='path to image dir.'
    # )

    parser.add_argument(
        "--output_dir", type=str, default=None, help="path to output file."
    )
    args = parser.parse_args()
    main(args)
