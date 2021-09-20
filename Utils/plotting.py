import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms
from Utils.dataloading import *


def plot_data_samples(images, title):
    """
    test function, not really used anymore
    todo: adjust for variable amount of images
    """
    fig = plt.figure(figsize=[12.8, 9.6])
    plt.suptitle(title)
    for i in range(16):
        plt.subplot(4, 4, i+1)
        plt.tight_layout()
        plt.subplots_adjust(wspace=0.05, hspace=0.05)
        plt.imshow(images[i].permute(1, 2, 0))
        plt.xticks([])
        plt.yticks([])


def plot_annotated_data_samples(images, targets, title, settype):
    """
    Plot a figure with image samples from the datasets, including the ground truth bounding boxes and labels
    Works for both VOC and COCO datasets.
    :param images: list of images (returned by dataloader)
    :param targets: list of targets, including boxes, labels and difficulties/areas (returned by dataloader)
    :param title: title for the image
    :param settype: either 'voc' or 'coco', to determine which label_map to use
    Source: https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/blob/master/detect.py (for drawing boxes)
    todo: adjust for variable amount of images based on batchsize
    todo: make a function that is able to plot predictions
    """

    fig = plt.figure(figsize=[12.8, 9.6])
    plt.suptitle(title)
    boxes = []
    labels = []

    for j in range(len(targets)):
        boxes.append(targets[j]['boxes'])
        labels.append(targets[j]['labels'])

    for k in range(len(targets)):
        annotated_image = transforms.ToPILImage()(images[k]).convert('RGB')
        draw = ImageDraw.Draw(annotated_image)
        font = ImageFont.truetype("./calibril.ttf", 15)

        if settype == 'voc':
            for i in range(boxes[k].size(0)):
                # Boxes
                box_location = (boxes[k][i]).tolist()
                draw.rectangle(xy=box_location, outline=label_color_map[voc_labels[labels[k][i].item()-1]])
                draw.rectangle(xy=[l+1. for l in box_location], outline=label_color_map[
                               voc_labels[labels[k][i].item()-1]]
                               )
                # Text
                text_size = font.getsize(voc_labels[labels[k][i].item()-1].upper())
                text_location = [box_location[0]+2., box_location[1] - text_size[1]]
                textbox_location = [box_location[0], box_location[1] - text_size[1],
                                    box_location[0] + text_size[0]+4.,
                                    box_location[1]
                                    ]
                draw.rectangle(xy=textbox_location, fill=label_color_map[voc_labels[labels[k][i].item()-1]])
                draw.text(xy=text_location, text=voc_labels[labels[k][i].item()-1].upper(), fill='white',
                          font=font
                          )
            del draw
        else:
            for i in range(boxes[k].size(0)):
                # Boxes
                box_location = (boxes[k][i]).tolist()
                draw.rectangle(xy=box_location, outline=coco_label_color_map[coco_labels[labels[k][i].item()-1]])
                draw.rectangle(xy=[l+1. for l in box_location], outline=coco_label_color_map[
                               coco_labels[labels[k][i].item()-1]]
                               )
                # Text
                text_size = font.getsize(coco_labels[labels[k][i].item()-1].upper())
                text_location = [box_location[0]+2., box_location[1] - text_size[1]]
                textbox_location = [box_location[0], box_location[1] - text_size[1],
                                    box_location[0] + text_size[0]+4.,
                                    box_location[1]
                                    ]
                draw.rectangle(xy=textbox_location, fill=coco_label_color_map[coco_labels[labels[k][i].item()-1]])
                draw.text(xy=text_location, text=coco_labels[labels[k][i].item()-1].upper(), fill='white',
                          font=font
                          )
            del draw

        plt.subplot(len(targets), 1, k+1)
        plt.tight_layout()
        plt.subplots_adjust(wspace=0.05, hspace=0.05)
        plt.imshow(annotated_image)
        plt.xticks([])
        plt.yticks([])


def plot_predicted_data_samples(images, box_k, lab_k, title, settype):
    """
    Plot a figure with image samples from the datasets, including the ground truth bounding boxes and labels
    Works for both VOC and COCO datasets.
    :param images: list of images (returned by dataloader)
    :param targets: list of targets, including boxes, labels and difficulties/areas (returned by dataloader)
    :param title: title for the image
    :param settype: either 'voc' or 'coco', to determine which label_map to use
    Source: https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/blob/master/detect.py (for drawing boxes)
    todo: adjust for variable amount of images based on batchsize
    todo: make a function that is able to plot predictions
    """

    fig = plt.figure(figsize=[12.8, 9.6])
    plt.suptitle(title)
    boxes = []
    labels = []
    boxes.append(box_k)
    labels.append(lab_k)

    # for i in range(box100.shape[0]):
    #     boxes.append(box100[i].item())
    #     labels.append(lab100[i].item())
    #     print("box in pred plot: ", boxes)


    for k in range(len(boxes)):
        annotated_image = transforms.ToPILImage()(images[k]).convert('RGB')
        draw = ImageDraw.Draw(annotated_image)
        font = ImageFont.truetype("./calibril.ttf", 15)

        if settype == 'voc':
            for i in range(boxes[k].size(0)):
                # Boxes
                box_location = (boxes[k][i]).tolist()
                draw.rectangle(xy=box_location, outline=label_color_map[voc_labels[labels[k][i].item()-1]])
                draw.rectangle(xy=[l+1. for l in box_location], outline=label_color_map[
                               voc_labels[labels[k][i].item()-1]]
                               )
                # Text
                text_size = font.getsize(voc_labels[labels[k][i].item()-1].upper())
                text_location = [box_location[0]+2., box_location[1] - text_size[1]]
                textbox_location = [box_location[0], box_location[1] - text_size[1],
                                    box_location[0] + text_size[0]+4.,
                                    box_location[1]
                                    ]
                draw.rectangle(xy=textbox_location, fill=label_color_map[voc_labels[labels[k][i].item()-1]])
                draw.text(xy=text_location, text=voc_labels[labels[k][i].item()-1].upper(), fill='white',
                          font=font
                          )
            del draw
        else:
            for i in range(boxes[k].size(0)):
                # Boxes
                box_location = (boxes[k][i]).tolist()
                draw.rectangle(xy=box_location, outline=coco_label_color_map[coco_labels[labels[k][i].item()-1]])
                draw.rectangle(xy=[l+1. for l in box_location], outline=coco_label_color_map[
                               coco_labels[labels[k][i].item()-1]]
                               )
                # Text
                text_size = font.getsize(coco_labels[labels[k][i].item()-1].upper())
                text_location = [box_location[0]+2., box_location[1] - text_size[1]]
                textbox_location = [box_location[0], box_location[1] - text_size[1],
                                    box_location[0] + text_size[0]+4.,
                                    box_location[1]
                                    ]
                draw.rectangle(xy=textbox_location, fill=coco_label_color_map[coco_labels[labels[k][i].item()-1]])
                draw.text(xy=text_location, text=coco_labels[labels[k][i].item()-1].upper(), fill='white',
                          font=font
                          )
            del draw

        plt.subplot(len(boxes), 1, k+1)
        plt.tight_layout()
        plt.subplots_adjust(wspace=0.05, hspace=0.05)
        plt.imshow(annotated_image)
        plt.xticks([])
        plt.yticks([])