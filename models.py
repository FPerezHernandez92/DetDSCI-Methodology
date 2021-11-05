import numpy as np
import tensorflow as tf
import os
import time

from tensorflow.keras.applications.resnet import ResNet50
from tensorflow.keras.models import load_model

from object_detection.utils import config_util
from object_detection.builders import model_builder
from object_detection.utils import label_map_util

from utils import get_useful_objects

MEMORY_USE_GPU = 5300


class ZoomLevelClassifier:
    """ Python class where level zoom classification will be performed.

    Attributes
    ----------
    batch_size: int
        Batch size of the training phase
    path_to_model_dir: str
        Path to the classification model
    base_model: model
        Classifier's base model
    model: model
        Deep learning model
    classes: list
        List with all possible classes of the model
    """

    def __init__(self, path_to_model_dir):

        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                tf.config.experimental.set_virtual_device_configuration(gpus[0], [
                    tf.config.experimental.VirtualDeviceConfiguration(memory_limit=MEMORY_USE_GPU)])
            except RuntimeError as e:
                print(e)

        seed = 1
        np.random.seed(seed)
        self.batch_size = 16
        self.path_to_model_dir = path_to_model_dir
        self.base_model = ResNet50(include_top=False, weights="imagenet", pooling='avg')
        self.model = load_model(os.path.join(self.path_to_model_dir, 'resnet50.h5'))
        file_classes = open(os.path.join(self.path_to_model_dir, 'classes.txt'), 'r')
        classes = file_classes.readline()[2:-2]
        file_classes.close()
        self.classes = classes.split('\', \'')

    def predictImageWithClassifier(self, img_tensor):
        """ Function to perform predictions over images.

        Parameters
        ----------
        img_tensor: Tensor
            Image's tensor

        Returns
        -------
        predict: str
            Classifier's output
        """

        img_tensor = np.expand_dims(img_tensor, axis=0)
        features = self.base_model.predict(img_tensor, batch_size=self.batch_size)
        init_time_cla = time.time()
        predictions = self.model.predict(features)
        print("Classification time:", time.time() - init_time_cla)
        precision = '%.2f' % (predictions[0][np.argmax(predictions[0])] * 100)
        out_class = [self.classes[int(np.argmax(predictions[0]))], precision]
        print("Classification out:", out_class[0])

        return out_class


class DetectionModel:
    """ Python class where model will be loaded in the GPU.

    Attributes
    ----------
    path_to_model_dir: str
        Path to model's directory
    scale: str
        Scale of the detector
    detection_model: DetectionModel
        Detection model
    ckpt: Checkpoint
        Detection model's checkpoint
    detect_fn: function
        Detection function
    path_to_labels: str
        Path to the file "label_map.pbtxt"
    path_to_config: str
        Path to the file "model.config"
    useful_objects: list
        Useful objects list
    label_map: list
        Label map list
    num_classes: int
        Number of classes of the model
    categories: list
        List of model's categories
    """

    def __init__(self, path_to_model_dir, scale):

        self.path_to_model_dir = path_to_model_dir
        self.scale = scale
        self.detection_model = None
        self.ckpt = None
        self.detect_fn = None
        self.path_to_labels = os.path.join(self.path_to_model_dir, 'label_map.pbtxt')
        self.path_to_config = os.path.join(self.path_to_model_dir, 'model.config')
        self.useful_objects = get_useful_objects(path_to_json='useful_objects.json', scale=self.scale)
        self.label_map = label_map_util.load_labelmap(self.path_to_labels)
        self.num_classes = label_map_util.get_max_label_map_index(self.label_map)
        self.categories = label_map_util.convert_label_map_to_categories(self.label_map,
                                                                         max_num_classes=self.num_classes,
                                                                         use_display_name=True)

    def load_model_in_gpu(self):
        """ Function to load the model in the GPU.

            Returns
            -------
            function
                Detect function
        """

        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                tf.config.experimental.set_virtual_device_configuration(
                    gpus[0],
                    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=MEMORY_USE_GPU)]
                )
            except RuntimeError as e:
                print(e)

        configs = config_util.get_configs_from_pipeline_file(self.path_to_config)
        model_config = configs['model']
        self.detection_model = model_builder.build(model_config=model_config, is_training=False)
        self.ckpt = tf.compat.v2.train.Checkpoint(model=self.detection_model)
        self.ckpt.restore(os.path.join(self.path_to_model_dir, 'ckpt-0')).expect_partial()

        def get_model_detection_function(model):
            """Get a tf.function for detection.

                Parameters
                ----------
                model: DetectionModel
                    Detection model

                Returns
                -------
                detect_fn: function
                    Detect function
            """

            @tf.function
            def detect_fn(image):
                """Detect objects in image.

                    Parameters
                    ----------
                    image: image
                        Image to perform detection

                    Returns
                    -------
                    detections: list
                        Model's detections
                    prediction_dict: dict
                        Dictionary with predictions
                    tf.reshape(): Tensor
                        Model's tensor
                """

                image, shapes = model.preprocess(image)
                prediction_dict = model.predict(image, shapes)
                detections = model.postprocess(prediction_dict, shapes)

                return detections, prediction_dict, tf.reshape(shapes, [-1])

            return detect_fn

        self.detect_fn = get_model_detection_function(self.detection_model)

        return self.detect_fn

    def predictImageWithDetector(self, input_tensor):
        """ Function to detect an image.

            Parameters
            ----------
            input_tensor: Tensor
                Image's tensor

            Returns
            -------
            detections: list
                Model's detections
            predictions_dict: dict
                Dictionary with predictions
            shapes: Tensor
                Model's tensor
            num_classes: int
                Number of classes
            categories: list
                Model's categories
            useful_objects: list
                Useful objects list
        """

        init_time_det = time.time()
        detections, predictions_dict, shapes = self.detect_fn(input_tensor)
        print("Detection time:", time.time() - init_time_det)
        return detections, predictions_dict, shapes, self.num_classes, self.categories, self.useful_objects
