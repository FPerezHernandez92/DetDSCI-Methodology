import os
import argparse
from PIL import Image

from models import ZoomLevelClassifier, DetectionModel
from processing import process_images

Image.MAX_IMAGE_PIXELS = 933120000

parser = argparse.ArgumentParser()
parser.add_argument('--value_threshold', default=0.5, type=float)
parser.add_argument('--max_width_crop', default=2000, type=int)
parser.add_argument('--max_height_crop', default=1500, type=int)
parser.add_argument('--dir_img', default='Images/', type=str)
parser.add_argument('--dir_out', default='out', type=str)
args = parser.parse_args()


if __name__ == '__main__':

    # Get list of images to be detected
    list_images = []
    list_facilities = os.listdir(args.dir_img)
    list_facilities.sort()
    for facility in list_facilities:
        input_image = os.path.join(args.dir_img, facility)
        list_images.append(input_image)

    if args.dir_out == '':
        out_dir = 'out_' + args.dir_img
    else:
        out_dir = args.dir_out

    print("Loading SmallScale model at GPU...")
    small_object_detector = DetectionModel(path_to_model_dir='SmallScale', scale='small')
    detect_fn1 = small_object_detector.load_model_in_gpu()
    print("OK: SmallScale model loaded")

    print('\n-----------------------------------------\n')

    print("Loading LargeScale model at GPU...")
    large_object_detector = DetectionModel(path_to_model_dir='LargeScale', scale='large')
    detect_fn2 = large_object_detector.load_model_in_gpu()
    print("OK: LargeScale model loaded")

    print("Loading ZoomLevelClassifier...")
    last_classifier = 'ZoomLevelClassifier'
    classifier = ZoomLevelClassifier(path_to_model_dir='ZoomLevelClassifier')
    print("OK: ZoomLevelClassifier model loaded")

    process_images(
        value_threshold=args.value_threshold,
        list_images=list_images,
        max_width_crop=args.max_width_crop,
        max_height_crop=args.max_height_crop,
        small_object_detector=small_object_detector,
        large_object_detector=large_object_detector,
        classifier=classifier,
        output_dir=out_dir
    )

    print("OK: Images processed")
