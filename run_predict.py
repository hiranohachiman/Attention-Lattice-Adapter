from predict import Predictor
import os
import glob
from tqdm import tqdm

label_file = "datasets/CUB/test_label.txt"
config_file = "configs/san_clip_vit_res4_coco.yaml"

def get_img_path(img_dir):
    img_types = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.gif', '*.tiff']
    img_paths = []
    for img_type in img_types:
        img_paths.extend(glob.glob(os.path.join(img_dir, img_type)))
    return img_paths

def get_image_data_details(line):
    img_path = line.split(',')[0]
    label = int(line.split(',')[1].replace('\n', '').replace(' ', ''))
    output_file = os.path.join(args.output_dir, img_path.replace("test/","",).replace("/","_").replace(" ",""))
    img_path = os.path.join('datasets/CUB/', img_path.replace(' ', ''))
    return (img_path, label, output_file)


def predict_with_bird(args):
    sample = 0
    correct = 0
    predictor = Predictor(config_file=config_file, model_path=args.model_path)
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
            assert type(pred_class) == int
            if pred_class == int(label):
                correct += 1
            sample += 1
        print(f"accuracy: {correct/sample}")


def main(args):
    predict_with_bird(args)

def predict_with_name(args):
    pass


def predict_class(args):
    pass

if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument(
        "--predict_mode", type=str, required=True, help="select from ['bird', 'name', 'class']"
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
    predictor = Predictor(config_file=args.config_file, model_path=args.model_path)
    predictor.predict(
        args.img_path,
        args.vocab.split(","),
        args.aug_vocab,
        output_file=args.output_file,
    )
