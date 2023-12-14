# uncompyle6 version 3.9.0
# Python bytecode version base 3.8.0 (3413)
# Decompiled from: Python 3.8.10 (default, Mar  8 2023, 16:27:05)
# [GCC 9.4.0]
# Embedded file name: /home/initial/workspace/smilab23/graduation_research/SAN/san/data/datasets/register_cub.py
# Compiled at: 2023-10-13 17:57:53
# Size of source mod 2**32: 6548 bytes
import os, torch
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_sem_seg
import json
CLASS_NAMES = ('Black footed Albatross', 'Laysan Albatross', 'Sooty Albatross', 'Groove billed Ani',
               'Crested Auklet', 'Least Auklet', 'Parakeet Auklet', 'Rhinoceros Auklet',
               'Brewer Blackbird', 'Red winged Blackbird', 'Rusty Blackbird', 'Yellow headed Blackbird',
               'Bobolink', 'Indigo Bunting', 'Lazuli Bunting', 'Painted Bunting',
               'Cardinal', 'Spotted Catbird', 'Gray Catbird', 'Yellow breasted Chat',
               'Eastern Towhee', 'Chuck will Widow', 'Brandt Cormorant', 'Red faced Cormorant',
               'Pelagic Cormorant', 'Bronzed Cowbird', 'Shiny Cowbird', 'Brown Creeper',
               'American Crow', 'Fish Crow', 'Black billed Cuckoo', 'Mangrove Cuckoo',
               'Yellow billed Cuckoo', 'Gray crowned Rosy Finch', 'Purple Finch',
               'Northern Flicker', 'Acadian Flycatcher', 'Great Crested Flycatcher',
               'Least Flycatcher', 'Olive sided Flycatcher', 'Scissor tailed Flycatcher',
               'Vermilion Flycatcher', 'Yellow bellied Flycatcher', 'Frigatebird',
               'Northern Fulmar', 'Gadwall', 'American Goldfinch', 'European Goldfinch',
               'Boat tailed Grackle', 'Eared Grebe', 'Horned Grebe', 'Pied billed Grebe',
               'Western Grebe', 'Blue Grosbeak', 'Evening Grosbeak', 'Pine Grosbeak',
               'Rose breasted Grosbeak', 'Pigeon Guillemot', 'California Gull', 'Glaucous winged Gull',
               'Heermann Gull', 'Herring Gull', 'Ivory Gull', 'Ring billed Gull',
               'Slaty backed Gull', 'Western Gull', 'Anna Hummingbird', 'Ruby throated Hummingbird',
               'Rufous Hummingbird', 'Green Violetear', 'Long tailed Jaeger', 'Pomarine Jaeger',
               'Blue Jay', 'Florida Jay', 'Green Jay', 'Dark eyed Junco', 'Tropical Kingbird',
               'Gray Kingbird', 'Belted Kingfisher', 'Green Kingfisher', 'Pied Kingfisher',
               'Ringed Kingfisher', 'White breasted Kingfisher', 'Red legged Kittiwake',
               'Horned Lark', 'Pacific Loon', 'Mallard', 'Western Meadowlark', 'Hooded Merganser',
               'Red breasted Merganser', 'Mockingbird', 'Nighthawk', 'Clark Nutcracker',
               'White breasted Nuthatch', 'Baltimore Oriole', 'Hooded Oriole', 'Orchard Oriole',
               'Scott Oriole', 'Ovenbird', 'Brown Pelican', 'White Pelican', 'Western Wood Pewee',
               'Sayornis', 'American Pipit', 'Whip poor Will', 'Horned Puffin', 'Common Raven',
               'White necked Raven', 'American Redstart', 'Geococcyx', 'Loggerhead Shrike',
               'Great Grey Shrike', 'Baird Sparrow', 'Black throated Sparrow', 'Brewer Sparrow',
               'Chipping Sparrow', 'Clay colored Sparrow', 'House Sparrow', 'Field Sparrow',
               'Fox Sparrow', 'Grasshopper Sparrow', 'Harris Sparrow', 'Henslow Sparrow',
               'Le Conte Sparrow', 'Lincoln Sparrow', 'Nelson Sharp tailed Sparrow',
               'Savannah Sparrow', 'Seaside Sparrow', 'Song Sparrow', 'Tree Sparrow',
               'Vesper Sparrow', 'White crowned Sparrow', 'White throated Sparrow',
               'Cape Glossy Starling', 'Bank Swallow', 'Barn Swallow', 'Cliff Swallow',
               'Tree Swallow', 'Scarlet Tanager', 'Summer Tanager', 'Artic Tern',
               'Black Tern', 'Caspian Tern', 'Common Tern', 'Elegant Tern', 'Forsters Tern',
               'Least Tern', 'Green tailed Towhee', 'Brown Thrasher', 'Sage Thrasher',
               'Black capped Vireo', 'Blue headed Vireo', 'Philadelphia Vireo', 'Red eyed Vireo',
               'Warbling Vireo', 'White eyed Vireo', 'Yellow throated Vireo', 'Bay breasted Warbler',
               'Black and white Warbler', 'Black throated Blue Warbler', 'Blue winged Warbler',
               'Canada Warbler', 'Cape May Warbler', 'Cerulean Warbler', 'Chestnut sided Warbler',
               'Golden winged Warbler', 'Hooded Warbler', 'Kentucky Warbler', 'Magnolia Warbler',
               'Mourning Warbler', 'Myrtle Warbler', 'Nashville Warbler', 'Orange crowned Warbler',
               'Palm Warbler', 'Pine Warbler', 'Prairie Warbler', 'Prothonotary Warbler',
               'Swainson Warbler', 'Tennessee Warbler', 'Wilson Warbler', 'Worm eating Warbler',
               'Yellow Warbler', 'Northern Waterthrush', 'Louisiana Waterthrush',
               'Bohemian Waxwing', 'Cedar Waxwing', 'American Three toed Woodpecker',
               'Pileated Woodpecker', 'Red bellied Woodpecker', 'Red cockaded Woodpecker',
               'Red headed Woodpecker', 'Downy Woodpecker', 'Bewick Wren', 'Cactus Wren',
               'Carolina Wren', 'House Wren', 'Marsh Wren', 'Rock Wren', 'Winter Wren',
               'Common Yellowthroat')

def _get_cub_meta(cat_list):
    ret = {'stuff_classes': cat_list}
    return ret


def load_seg_label(seg_dir, label_file):
    dataset_dicts = []
    with open(label_file) as f:
        lines = f.readlines()
        for line in lines:
            json_obj = json.loads(line)
            record = {}
            img_path = json_obj['image_path']
            label = torch.tensor(int(json_obj['label']))
            caption = json_obj['caption']
            record['file_name'] = os.path.join('datasets/CUB/', img_path.replace(' ', ''))
            record['label'] = label
            record['caption'] = caption
            record['sem_seg_file_name'] = os.path.join(seg_dir, img_path.replace('train/', '').replace("test/", "").replace('/', '_').replace('.jpg', '.png'))
            dataset_dicts.append(record)

    return dataset_dicts


def register_all_cub(root):
    root = os.path.join(root, 'CUB')
    meta = _get_cub_meta(CLASS_NAMES)
    for name, image_dirname, sem_seg_dirname, label_name in (
                                                             ('val', 'val', 'extracted_val_segmentation', 'valid_label.jsonl'),
                                                            #  ('train', 'train', 'extracted_train_segmentation', 'train_label.jsonl'),
                                                             ('train', 'train', 'extracted_train_segmentation', 'train_small.jsonl'),
                                                             ):
        image_dir = os.path.join(root, image_dirname)
        gt_dir = os.path.join(root, sem_seg_dirname)
        label_file = os.path.join(root, label_name)
        all_name = f"cub_{name}"
        DatasetCatalog.register(all_name, lambda: load_seg_label(gt_dir, label_file))
        (MetadataCatalog.get(all_name).set)(image_root=image_dir,
         sem_seg_root=gt_dir,
         evaluator_type='sem_seg',
         ignore_label=255, **meta)


_root = os.getenv('DETECTRON2_DATASETS', 'datasets')
register_all_cub(_root)
