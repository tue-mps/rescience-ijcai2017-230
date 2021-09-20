import json
import os
import torch
import xml.etree.ElementTree as ET


# Work with GPU if available, otherwise CPU.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Path of the project.
path_project = os.path.abspath(os.path.join(__file__,  "../.."))

# Label map for VOC dataset.
voc_labels = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
              'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
              )
label_map = {k: v + 1 for v, k in enumerate(voc_labels)}
label_map['background'] = 0
rev_label_map = {v: k for k, v in label_map.items()}  # Inverse mapping

# Color map for bounding boxes of detected objects from https://sashat.me/2017/01/11/list-of-20-simple-distinct-colors/
distinct_colors = ['#e6194b', '#3cb44b', '#ffe119', '#0082c8', '#f58231', '#911eb4', '#46f0f0', '#f032e6',
                   '#d2f53c', '#fabebe', '#008080', '#000080', '#aa6e28', '#fffac8', '#800000', '#aaffc3',
                   '#808000', '#ffd8b1', '#e6beff', '#808080', '#FFFFFF'
                   ]
label_color_map = {k: distinct_colors[i] for i, k in enumerate(label_map.keys())}

# Label map for COCO dataset.
coco_labels = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'street sign', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
               'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'hat', 'backpack', 'umbrella', 'shoe',
               'eye glasses', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
               'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'plate',
               'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
               'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'mirror', 'dining table', 'window', 'desk', 'toilet', 'door', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'blender', 'book',
               'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush', 'hair brush'
               )
coco_label_map = {k: v + 1 for v, k in enumerate(coco_labels)}
coco_label_map['background'] = 0
coco_rev_label_map = {v: k for k, v in coco_label_map.items()}  # Inverse mapping

# Color map for bounding boxes of detected objects from https://sashat.me/2017/01/11/list-of-20-simple-distinct-colors/
# (includes double colors).
coco_distinct_colors = ['#e6194b', '#3cb44b', '#ffe119', '#0082c8', '#f58231', '#911eb4', '#46f0f0', '#f032e6',
                        '#d2f53c', '#fabebe', '#008080', '#000080', '#aa6e28', '#fffac8', '#800000', '#aaffc3',
                        '#808000', '#ffd8b1', '#e6beff', '#808080', '#FFFFFF',
                        '#e6194b', '#3cb44b', '#ffe119', '#0082c8', '#f58231', '#911eb4', '#46f0f0', '#f032e6',
                        '#d2f53c', '#fabebe', '#008080', '#000080', '#aa6e28', '#fffac8', '#800000', '#aaffc3',
                        '#808000', '#ffd8b1', '#e6beff', '#808080', '#FFFFFF',
                        '#e6194b', '#3cb44b', '#ffe119', '#0082c8', '#f58231', '#911eb4', '#46f0f0', '#f032e6',
                        '#d2f53c', '#fabebe', '#008080', '#000080', '#aa6e28', '#fffac8', '#800000', '#aaffc3',
                        '#808000', '#ffd8b1', '#e6beff', '#808080', '#FFFFFF',
                        '#e6194b', '#3cb44b', '#ffe119', '#0082c8', '#f58231', '#911eb4', '#46f0f0', '#f032e6',
                        '#d2f53c', '#fabebe', '#008080', '#000080', '#aa6e28', '#fffac8', '#800000', '#aaffc3',
                        '#808000', '#ffd8b1', '#e6beff', '#808080', '#FFFFFF',
                        '#e6194b', '#3cb44b', '#ffe119', '#0082c8', '#f58231', '#911eb4', '#46f0f0', '#f032e6',
                        '#d2f53c', '#fabebe', '#008080', '#000080'
                        ]
coco_label_color_map = {k: coco_distinct_colors[i] for i, k in enumerate(coco_label_map.keys())}


def parse_annotation(annotation_path):
    """
    Create lists of images, the bounding boxes and labels of the objects in these images, and save these to file.
    :param annotation_path: path to the 'VOC2007' folder
    :return dictionary containing boxes, labels and difficulties
    source: https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/blob/master/utils.py
    """

    tree = ET.parse(annotation_path)
    root = tree.getroot()
    boxes = list()
    labels = list()
    difficulties = list()
    for object in root.iter('object'):

        difficult = int(object.find('difficult').text == '1')
        label = object.find('name').text.lower().strip()
        if label not in label_map:
            continue

        bbox = object.find('bndbox')
        xmin = int(bbox.find('xmin').text) - 1
        ymin = int(bbox.find('ymin').text) - 1
        xmax = int(bbox.find('xmax').text) - 1
        ymax = int(bbox.find('ymax').text) - 1

        boxes.append([xmin, ymin, xmax, ymax])
        labels.append(label_map[label])
        difficulties.append(difficult)

    return {'boxes': boxes, 'labels': labels, 'difficulties': difficulties}


def create_data_lists(voc07_path, output_folder):
    """
    Create lists of images, the bounding boxes, labels and difficulties of the objects in these images,
    and save these to a file. Separate files are created for training, validation and testing data.
    :param voc07_path: path to the 'VOC2007' folder where dataset is stored
    :param output_folder: folder where the JSONs must be saved
    Source: https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/blob/master/utils.py (adapted)
    Source: http://jultika.oulu.fi/files/nbnfi-fe202001131851.pdf (number of objects logics)
    """

    voc07_path = os.path.abspath(voc07_path)

    # Training data.
    train_images = list()
    train_objects = list()
    n_objects = 0

    for path in [voc07_path]:

        # Find IDs of images in training data.
        with open(os.path.join(path, 'ImageSets/Main/train.txt')) as f:
            ids = f.read().splitlines()

        for id in ids:
            # Parse annotation's XML file.
            objects = parse_annotation(os.path.join(path, 'Annotations', id + '.xml'))
            if len(objects['boxes']) == 0:
                continue
            n_objects += len(objects['boxes'])
            train_objects.append(objects)
            train_images.append(os.path.join(path, 'JPEGImages', id + '.jpg'))

    assert len(train_objects) == len(train_images)

    # Save to file.
    with open(os.path.join(output_folder, 'TRAIN_images.json'), 'w') as j:
        json.dump(train_images, j)
    with open(os.path.join(output_folder, 'TRAIN_objects.json'), 'w') as j:
        json.dump(train_objects, j)
    with open(os.path.join(output_folder, 'TRAIN_label_map.json'), 'w') as j:
        json.dump(label_map, j)  # Save label map too

    print('\nThere are %d training images containing a total of %d objects. Files have been saved to %s.' % (
        len(train_images), n_objects, os.path.abspath(output_folder)))

    # Validation data.
    validation_images = list()
    validation_objects = list()
    n_objects = 0

    for path in [voc07_path]:

        # Find IDs of images in training data.
        with open(os.path.join(path, 'ImageSets/Main/val.txt')) as f:
            ids = f.read().splitlines()

        for id in ids:
            # Parse annotation's XML file.
            objects = parse_annotation(os.path.join(path, 'Annotations', id + '.xml'))
            if len(objects['boxes']) == 0:
                continue
            n_objects += len(objects['boxes'])
            validation_objects.append(objects)
            validation_images.append(os.path.join(path, 'JPEGImages', id + '.jpg'))

    assert len(validation_objects) == len(validation_images)

    # Save to file.
    with open(os.path.join(output_folder, 'VALIDATION_images.json'), 'w') as j:
        json.dump(validation_images, j)
    with open(os.path.join(output_folder, 'VALIDATION_objects.json'), 'w') as j:
        json.dump(validation_objects, j)
    with open(os.path.join(output_folder, 'VALIDATION_label_map.json'), 'w') as j:
        json.dump(label_map, j)  # Save label map too

    print('\nThere are %d training images containing a total of %d objects. Files have been saved to %s.' % (
        len(validation_images), n_objects, os.path.abspath(output_folder)))

    # Test data.
    test_images = list()
    test_objects = list()
    n_objects = 0

    # Find IDs of images in the test data.
    with open(os.path.join(voc07_path, 'ImageSets/Main/test.txt')) as f:
        ids = f.read().splitlines()

    for id in ids:
        # Parse annotation's XML file.
        objects = parse_annotation(os.path.join(voc07_path, 'Annotations', id + '.xml'))
        if len(objects) == 0:
            continue
        test_objects.append(objects)
        n_objects += len(objects['boxes'])
        test_images.append(os.path.join(voc07_path, 'JPEGImages', id + '.jpg'))

    assert len(test_objects) == len(test_images)

    # Save to file
    with open(os.path.join(output_folder, 'TEST_images.json'), 'w') as j:
        json.dump(test_images, j)
    with open(os.path.join(output_folder, 'TEST_objects.json'), 'w') as j:
        json.dump(test_objects, j)

    print('\nThere are %d test images containing a total of %d objects. Files have been saved to %s.' % (
        len(test_images), n_objects, os.path.abspath(output_folder)))