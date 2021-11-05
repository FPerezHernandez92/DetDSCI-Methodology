import time
import numpy as np
import tensorflow as tf
import os
from PIL import Image

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util


def detect_objects(value_threshold, max_width_crop, max_height_crop, image, loadedDetectionModel1,
                  loadedDetectionModel2, classifier):
    """ Function to perform image detection

        Parameters
        ----------
        value_threshold: str
            Threshold value for an object to be consider. Between 0 and 1.
        max_width_crop: str
            Maximum width value to crop. Greater than 0.
        max_height_crop: str
            Maximum height value to crop. Greater than 0.
        image: PIL.Image
            Image to be processed.
        loadedDetectionModel1: function
            Detection function of the model 1
        loadedDetectionModel2: function
            Detection function of the model 2
        classifier: Classifier
            Level zoom classifier model

        Returns
        -------
        image_np
            Image detected as numpy array
    """

    width, height = image.size

    num_rows = int(width / (max_width_crop / 2))
    num_cols = int(height / (max_height_crop / 2))
    if num_cols == 0:
        num_cols = 1
    if num_rows == 0:
        num_rows = 1
    crop_x = 0
    crop_y = 0
    objects_detected = []
    objects_detected_dict = {}
    clases_detected = []

    for row in range(num_rows):
        last_crop_x = crop_x
        for col in range(num_cols):
            top_crop_x = crop_x + max_width_crop
            top_crop_y = crop_y + max_height_crop
            # Ensure that we maintain crop dimensions inside image dimensions
            if top_crop_x > width:
                top_crop_x = width
            if top_crop_y > height:
                top_crop_y = height

            last_crop_y = crop_y
            if max_width_crop > top_crop_x - crop_x:
                crop_x = top_crop_x - max_width_crop
            if max_height_crop > top_crop_y - crop_y:
                crop_y = top_crop_y - max_height_crop

            # Get cropped image
            crop_image = image.crop((crop_x, crop_y, top_crop_x, top_crop_y))

            if loadedDetectionModel2 is not None:
                out_class = classifier.predictImageWithClassifier(crop_image)
            image_np = np.array(crop_image).astype(np.uint8)

            input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
            if loadedDetectionModel2 is None:
                detections, predictions_dict, shapes, num_classes, categories, useful_objects = \
                    loadedDetectionModel1.predictImageWithDetector(input_tensor)
            elif out_class[0] == '14151617':
                detections, predictions_dict, shapes, num_classes, categories, useful_objects = \
                    loadedDetectionModel2.predictImageWithDetector(input_tensor)
            else:
                detections, predictions_dict, shapes, num_classes, categories, useful_objects = \
                    loadedDetectionModel1.predictImageWithDetector(input_tensor)

            label_id_offset = 1
            # Get objects detected in the cropped image
            x_height, x_width, channels = image_np.shape
            for i in range((np.squeeze(detections['detection_boxes'][0].numpy())).shape[0]):
                if (np.squeeze(detections['detection_scores'][0].numpy())[i]) > value_threshold:
                    num_clase = np.squeeze((detections['detection_classes'][0].numpy() +
                                            label_id_offset).astype(int)).astype(np.int32)[i] - 1
                    if num_clase <= num_classes:
                        for elem_list in categories:
                            if elem_list['id'] == (num_clase + 1):
                                name_class = elem_list['name']

                        # Only include useful objects
                        if name_class in useful_objects:
                            x_min = int(np.squeeze(detections['detection_boxes'][0].numpy())[i][1] * x_width)
                            x_max = int(np.squeeze(detections['detection_boxes'][0].numpy())[i][3] * x_width)
                            y_min = int(np.squeeze(detections['detection_boxes'][0].numpy())[i][0] * x_height)
                            y_max = int(np.squeeze(detections['detection_boxes'][0].numpy())[i][2] * x_height)
                            final_string = str(x_min + crop_x) + '_' + str(y_min + crop_y) + '_' + \
                                           str(x_max + crop_x) + '_' + str(y_max + crop_y) + '_' + \
                                           str(np.squeeze(detections['detection_scores'][0].numpy())[i]) + '_' + \
                                           str(num_clase + 1)
                            if name_class in objects_detected_dict:
                                if final_string in objects_detected_dict[name_class]:
                                    pass
                                else:
                                    aux_list = objects_detected_dict[name_class]
                                    aux_list.append(final_string)
                                    objects_detected_dict[name_class] = aux_list
                            else:
                                objects_detected_dict[name_class] = [final_string]
                            clases_detected.append(np.squeeze(
                                (detections['detection_classes'][0].numpy() + label_id_offset).astype(
                                    int)).astype(
                                np.int32)[i])
                        else:
                            print("DEBUG: Detected class without interest:", name_class)

            crop_y = last_crop_y + int(max_height_crop / 2)

        crop_x = last_crop_x + int(max_width_crop / 2)
        crop_y = 0

    # Create image with bounding boxes
    final_box = ""
    final_scores = ""
    final_classes = ""
    count_objet = 0
    for name_class in objects_detected_dict:
        for obj in objects_detected_dict[name_class]:
            split_obj = obj.split('_')
            new_object = [name_class, int(split_obj[0]), int(split_obj[1]), int(split_obj[2]),
                            int(split_obj[3]), split_obj[4], split_obj[5]]
            objects_detected.append(new_object)
    for object in objects_detected:
        aux_box = np.array([[object[2] / height, object[1] / width, object[4] / height, object[3] / width]])
        aux_scores = np.array([float(object[5])])
        aux_classes = np.array([clases_detected[count_objet]])
        if type(final_box) == type(aux_box):
            final_box = np.append(final_box, aux_box, axis=0)
            final_scores = np.append(final_scores, aux_scores, axis=0)
            final_classes = np.append(final_classes, aux_classes, axis=0)
        else:
            final_box = aux_box
            final_scores = aux_scores
            final_classes = aux_classes
        count_objet += 1

    image_np = np.array(image).astype(np.uint8)

    if len(objects_detected) >= 1:
        vis_util.visualize_boxes_and_labels_on_image_array(
            image_np,
            final_box,
            final_classes,
            final_scores,
            label_map_util.create_category_index(categories),
            use_normalized_coordinates=True,
            min_score_thresh=value_threshold,
            line_thickness=8,
            max_boxes_to_draw=3000)

    return image_np


def process_images(value_threshold, list_images, max_width_crop, max_height_crop,
                   small_object_detector, large_object_detector=None, classifier=None, output_dir='out'):
    """ Function to perform object detection in images

        Parameters
        ----------
        value_threshold: str
            Threshold value for an object to be consider. Between 0 and 1.
        list_images: dict
            List with images and their codes.
        max_width_crop: str
            Maximum width value to crop. Greater than 0.
        max_height_crop: str
            Maximum height value to crop. Greater than 0.
        small_object_detector: function
            Detection function of the model 1
        large_object_detector: function
            Detection function of the model 2
        classifier: Classifier
            Level zoom classifier model
        output_dir: str
            Directory where images are saved

        Returns
        -------
        detection_result: list
            List of lists with objects detected per image
    """

    detection_result = []
    value_threshold = float(value_threshold)

    total_images = len(list_images)

    for idx, x_image in enumerate(list_images):
        image = Image.open(x_image)
        print("Detecting image " + str(idx+1) + '/' + str(total_images) + " " + x_image)
        image_np = detect_objects(
            value_threshold,
            max_width_crop,
            max_height_crop,
            image,
            small_object_detector,
            large_object_detector,
            classifier
        )

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Save image
        final_image_path = os.path.join(output_dir, str(idx+1) + "_" + str(time.strftime("%Y%m%d-%H%M%S")) + '.png')
        vis_util.save_image_array_as_png(image_np, final_image_path)

    return detection_result
