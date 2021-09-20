from Utils.dataloading import *
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def find_intersection(set_1, set_2):
    """
    Find the intersection of every box combination between two sets of boxes that are in boundary coordinates.
    :param set_1: set 1, a tensor of dimensions (n1, 4)
    :param set_2: set 2, a tensor of dimensions (n2, 4)
    :return: intersection of each of the boxes in set 1 with respect to each of the boxes in set 2, a tensor of dimensions (n1, n2)
    Source: https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/blob/master/utils.py
    """

    # PyTorch auto-broadcasts singleton dimensions.
    lower_bounds = torch.max(set_1[:, :2].unsqueeze(1), set_2[:, :2].unsqueeze(0))  # (n1, n2, 2)
    upper_bounds = torch.min(set_1[:, 2:].unsqueeze(1), set_2[:, 2:].unsqueeze(0))  # (n1, n2, 2)
    intersection_dims = torch.clamp(upper_bounds - lower_bounds, min=0)  # (n1, n2, 2)
    return intersection_dims[:, :, 0] * intersection_dims[:, :, 1]  # (n1, n2)


def find_jaccard_overlap(set_1, set_2):
    """
    Find the Jaccard Overlap (IoU) of every box combination between two sets of boxes that are in boundary coordinates.
    :param set_1: set 1, a tensor of dimensions (n1, 4)
    :param set_2: set 2, a tensor of dimensions (n2, 4)
    :return: Jaccard Overlap of each of the boxes in set 1 with respect to each of the boxes in set 2, a tensor of dimensions (n1, n2)
    Source: https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/blob/master/utils.py
    """

    # Find intersections.
    intersection = find_intersection(set_1, set_2)  # (n1, n2)

    # Find areas of each box in both sets.
    areas_set_1 = (set_1[:, 2] - set_1[:, 0]) * (set_1[:, 3] - set_1[:, 1])  # (n1)
    areas_set_2 = (set_2[:, 2] - set_2[:, 0]) * (set_2[:, 3] - set_2[:, 1])  # (n2)

    # Find the union.  PyTorch auto-broadcasts singleton dimensions.
    union = areas_set_1.unsqueeze(1) + areas_set_2.unsqueeze(0) - intersection  # (n1, n2)

    return intersection / union  # (n1, n2)


def voc_metrics(det_boxes, det_labels, det_scores, true_boxes, true_labels, true_difficulties, settype):
    """
    Calculate the Mean Average Precision (mAP) of detected objects.
    See https://medium.com/@jonathan_hui/map-mean-average-precision-for-object-detection-45c121a31173 for an explanation
    :param det_boxes: list of tensors, one tensor for each image containing detected objects' bounding boxes
    :param det_labels: list of tensors, one tensor for each image containing detected objects' labels
    :param det_scores: list of tensors, one tensor for each image containing detected objects' labels' scores
    :param true_boxes: list of tensors, one tensor for each image containing actual objects' bounding boxes
    :param true_labels: list of tensors, one tensor for each image containing actual objects' labels
    :param true_difficulties: list of tensors, one tensor for each image containing actual objects' difficulty (0 or 1)
    :param settype: either 'voc' or 'coco'
    :return average_precisions: list of average precisions for each of the classes
    :return mean_average_precision: (mAP), averaged precision over all classes combined
    :return classwise_recall: list of recall for each of the classes
    :return all_recall: recall for all classes combined
    Source: https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/blob/master/utils.py (adapted)
    """

    assert len(det_boxes) == len(det_labels) == len(det_scores) == len(true_boxes)\
           == len(true_labels) == len(true_difficulties)  # These are all lists of tensors of the same length
    if settype == 'voc':
        n_classes = len(label_map)
    else:
        n_classes = len(coco_label_map)

    # Store all (true) objects in a single continuous tensor while keeping track of the image it is from.
    true_images = list()
    for i in range(len(true_labels)):
        true_images.extend([i] * true_labels[i].size(0))
    true_images = torch.LongTensor(true_images).to(device)  # (n_objects), no. of objects across all images
    true_boxes = torch.cat(true_boxes, dim=0)  # (n_objects, 4)
    true_labels = torch.cat(true_labels, dim=0)  # (n_objects)
    true_difficulties = torch.cat(true_difficulties, dim=0)  # (n_objects)

    assert true_images.size(0) == true_boxes.size(0) == true_labels.size(0)

    # Store all detections in a single continuous tensor while keeping track of the image it is from.
    det_images = list()
    for i in range(len(det_labels)):
        det_images.extend([i] * det_labels[i].size(0))
    det_images = torch.LongTensor(det_images).to(device)  # (n_detections)
    det_boxes = torch.cat(det_boxes, dim=0)  # (n_detections, 4)
    det_labels = torch.cat(det_labels, dim=0)  # (n_detections)
    det_scores = torch.cat(det_scores, dim=0)  # (n_detections)

    assert det_images.size(0) == det_boxes.size(0) == det_labels.size(0) == det_scores.size(0)

    # Calculate APs for each class (except background).
    average_precisions = torch.zeros((n_classes-1), dtype=torch.float)  # (n_classes - 1)
    classwise_recall = torch.zeros((n_classes-1), dtype=torch.float)  # (n_classes - 1)
    total_tp = torch.zeros((n_classes-1), dtype=torch.float)  # (n_classes - 1)
    total_fp = torch.zeros((n_classes - 1), dtype=torch.float)  # (n_classes - 1)
    n_easy_all_objects = (1 - true_difficulties).sum().item()  # Ignore difficult objects

    for c in range(1, n_classes):
        # Extract only objects with this class.
        true_class_images = true_images[true_labels == c]  # (n_class_objects)
        true_class_boxes = true_boxes[true_labels == c]  # (n_class_objects, 4)
        true_class_difficulties = true_difficulties[true_labels == c]  # (n_class_objects)
        n_easy_class_objects = (1 - true_class_difficulties).sum().item()  # Ignore difficult objects

        # Keep track of which true objects with this class have already been 'detected'.  So far, none.
        true_class_boxes_detected = torch.zeros((true_class_difficulties.size(0)),
                                                dtype=torch.uint8).to(device)  # (n_class_objects)

        # Extract only detections with this class.
        det_class_images = det_images[det_labels == c]  # (n_class_detections)
        det_class_boxes = det_boxes[det_labels == c]  # (n_class_detections, 4)
        det_class_scores = det_scores[det_labels == c]  # (n_class_detections)
        n_class_detections = det_class_boxes.size(0)
        if n_class_detections == 0:
            continue

        # Sort detections in decreasing order of confidence/scores.
        det_class_scores, sort_ind = torch.sort(det_class_scores, dim=0, descending=True)  # (n_class_detections)
        det_class_images = det_class_images[sort_ind]  # (n_class_detections)
        det_class_boxes = det_class_boxes[sort_ind]  # (n_class_detections, 4)

        # In the order of decreasing scores, check if true or false positive.
        true_positives = torch.zeros(n_class_detections, dtype=torch.float).to(device)  # (n_class_detections)
        false_positives = torch.zeros(n_class_detections, dtype=torch.float).to(device)  # (n_class_detections)
        for d in range(n_class_detections):
            this_detection_box = det_class_boxes[d].unsqueeze(0)  # (1, 4)
            this_image = det_class_images[d]  # (), scalar

            # Find objects in the same image with this class, their difficulties, and whether they have been detected
            # before.
            object_boxes = true_class_boxes[true_class_images == this_image]  # (n_class_objects_in_img)
            object_difficulties = true_class_difficulties[true_class_images == this_image]  # (n_class_objects_in_img)
            # If no such object in this image, then the detection is a false positive.
            if object_boxes.size(0) == 0:
                false_positives[d] = 1
                continue

            # Find maximum overlap of this detection with objects in this image of this class.
            overlaps = find_jaccard_overlap(this_detection_box, object_boxes)  # (1, n_class_objects_in_img)
            max_overlap, ind = torch.max(overlaps.squeeze(0), dim=0)  # (), () - scalars

            # 'ind' is the index of the object in these image-level tensors 'object_boxes', 'object_difficulties'
            # In the original class-level tensors 'true_class_boxes', etc., 'ind' corresponds to object with index...
            # We need 'original_ind' to update 'true_class_boxes_detected'.
            original_ind = torch.LongTensor(range(true_class_boxes.size(0)))[true_class_images == this_image][ind]

            # If the maximum overlap is greater than the threshold of 0.5, it's a match.
            if max_overlap.item() > 0.5:
                # If the object it matched with is 'difficult', ignore it.
                if object_difficulties[ind] == 0:
                    # If this object has already not been detected, it's a true positive.
                    if true_class_boxes_detected[original_ind] == 0:
                        true_positives[d] = 1
                        true_class_boxes_detected[original_ind] = 1  # this object has now been detected/accounted for
                    # Otherwise, it's a false positive (since this object is already accounted for).
                    else:
                        false_positives[d] = 1
            # Otherwise, the detection occurs in a different location than the actual object, and is a false positive.
            else:
                false_positives[d] = 1

        # Compute cumulative precision and recall at each detection in the order of decreasing scores.
        cum_true_positives = torch.cumsum(true_positives, dim=0)  # (n_class_detections)
        cum_false_positives = torch.cumsum(false_positives, dim=0)  # (n_class_detections)
        cum_precision = cum_true_positives \
            / (cum_true_positives + cum_false_positives + 1e-10)  # (n_class_detections)
        cum_recall = cum_true_positives / n_easy_class_objects  # (n_class_detections)
        classwise_recall[c-1] = cum_recall[-1]
        total_tp[c-1] = cum_true_positives[-1]
        total_fp[c-1] = cum_false_positives[-1]

        # Find the mean of the maximum of the precisions corresponding to recalls above the threshold 't'.
        recall_thresholds = torch.arange(start=0, end=1.01, step=.01).tolist()  # (101)
        precisions = torch.zeros((len(recall_thresholds)), dtype=torch.float).to(device)  # (101)
        for i, t in enumerate(recall_thresholds):
            recalls_above_t = cum_recall >= t
            if recalls_above_t.any():
                precisions[i] = cum_precision[recalls_above_t].max()
            else:
                precisions[i] = 0.
        average_precisions[c-1] = precisions.mean()  # c is in [1, n_classes - 1]

    # Calculate Mean Average Precision (mAP).
    mean_average_precision = average_precisions.mean().item()

    if settype == 'voc':
        # Keep class-wise average precisions in a dictionary.
        average_precisions = {rev_label_map[c+1]: v for c, v in enumerate(average_precisions.tolist())}

        # Keep class-wise recall in a dictionary.
        classwise_recalls = {rev_label_map[c+1]: v for c, v in enumerate(classwise_recall.tolist())}
    else:
        # Keep class-wise average precisions in a dictionary.
        average_precisions = {coco_rev_label_map[c+1]: v for c, v in enumerate(average_precisions.tolist())}

        # Keep class-wise recall in a dictionary.
        classwise_recalls = {coco_rev_label_map[c+1]: v for c, v in enumerate(classwise_recall.tolist())}

    all_recall = (torch.sum(total_tp, dim=0) / n_easy_all_objects).item()
    all_recall_by_average = classwise_recall.mean().item()

    tp = torch.sum(total_tp, dim=0)
    fn = n_easy_all_objects - tp
    fp = torch.sum(total_fp, dim=0)

    print("TP: ", tp)
    print("FP: ", fp)
    print("FN: ", fn)

    return average_precisions, mean_average_precision, classwise_recalls, all_recall, all_recall_by_average


def coco_metrics(det_boxes, det_labels, det_scores, true_boxes, true_labels, true_areas):
    """
    Calculate the Mean Average Precision (mAP) of detected objects.
    See https://medium.com/@jonathan_hui/map-mean-average-precision-for-object-detection-45c121a31173 for an explanation
    :param det_boxes: list of tensors, one tensor for each image containing detected objects' bounding boxes
    :param det_labels: list of tensors, one tensor for each image containing detected objects' labels
    :param det_scores: list of tensors, one tensor for each image containing detected objects' labels' scores
    :param true_boxes: list of tensors, one tensor for each image containing actual objects' bounding boxes
    :param true_labels: list of tensors, one tensor for each image containing actual objects' labels
    :param true_areas: list of tensors, one tensor for each image containing actual objects' surface area
    :return average_precisions: list of average precisions for each of the classes
    :return mean_average_precision: (mAP), averaged precision over all classes combined
    :return classwise_recall: list of recall for each of the classes
    :return all_recall_by_average: recall for all classes combined
    :return recall_small_by_average: recall for all classes combined, only considering small objects
    :return recall_medium_by_average: recall for all classes combined, only considering medium objects
    :return recall_large_by_average: recall for all classes combined, only considering large objects
    Source: https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/blob/master/utils.py (heavily adapted)
    """

    assert len(det_boxes) == len(det_labels) == len(det_scores) == len(true_boxes) \
           == len(true_labels) == len(true_areas)  # These are all lists of tensors of the same length
    n_classes = len(coco_label_map)     # (92)

    # Store all (true) objects in a single continuous tensor while keeping track of the image it is from.
    true_images = list()
    for i in range(len(true_labels)):
        true_images.extend([i] * true_labels[i].size(0))
    true_images = torch.LongTensor(true_images).to(device)  # (n_objects), no. of objects across all images
    true_boxes = torch.cat(true_boxes, dim=0)  # (n_objects, 4)
    true_labels = torch.cat(true_labels, dim=0)  # (n_objects)
    true_areas = torch.cat(true_areas, dim=0)  # (n_objects)

    assert true_images.size(0) == true_boxes.size(0) == true_labels.size(0)

    # Store all detections in a single continuous tensor while keeping track of the image it is from.
    det_images = list()
    for i in range(len(det_labels)):
        det_images.extend([i] * det_labels[i].size(0))
    det_images = torch.LongTensor(det_images).to(device)  # (n_detections)
    det_boxes = torch.cat(det_boxes, dim=0)  # (n_detections, 4)
    det_labels = torch.cat(det_labels, dim=0)  # (n_detections)
    det_scores = torch.cat(det_scores, dim=0)  # (n_detections)

    assert det_images.size(0) == det_boxes.size(0) == det_labels.size(0) == det_scores.size(0)

    iou = np.linspace(.5, 0.95, int(np.round((0.95 - .5) / .05)) + 1, endpoint=True)  # [0.5 : 0.05 : 0.95]

    # initialize some empty tensors (n_classes -1)
    classwise_recall = torch.zeros((n_classes-1), dtype=torch.float)
    classwise_recall_small = torch.zeros((n_classes-1), dtype=torch.float)
    classwise_recall_medium = torch.zeros((n_classes-1), dtype=torch.float)
    classwise_recall_large = torch.zeros((n_classes-1), dtype=torch.float)
    n_all_objects_small = torch.zeros((n_classes-1), dtype=torch.float)
    n_all_objects_medium = torch.zeros((n_classes-1), dtype=torch.float)
    n_all_objects_large = torch.zeros((n_classes-1), dtype=torch.float)
    ap_class = torch.zeros((n_classes-1), dtype=torch.float)  # (91)

    # For each class (except background).
    for c in range(1, n_classes):
        # initialize/clear some more empty tensors (10)
        ap_iou = torch.zeros(len(iou), dtype=torch.float)
        recall_iou = torch.zeros(len(iou), dtype=torch.float)
        recall_iou_small = torch.zeros(len(iou), dtype=torch.float)
        recall_iou_medium = torch.zeros(len(iou), dtype=torch.float)
        recall_iou_large = torch.zeros(len(iou), dtype=torch.float)

        # Extract only objects with this class.
        true_class_images = true_images[true_labels == c]  # (n_class_objects)
        true_class_boxes = true_boxes[true_labels == c]  # (n_class_objects, 4)
        true_class_areas = true_areas[true_labels == c]  # (n_class_objects)
        n_class_objects = true_class_images.size(0)

        # Keep track of which true objects with this class have already been 'detected'.  So far, none.
        true_class_boxes_detected = torch.zeros((n_class_objects, len(iou)),
                                                dtype=torch.uint8).to(device)  # (n_class_objects)

        # Extract only detections with this class.
        det_class_images = det_images[det_labels == c]  # (n_class_detections)
        det_class_boxes = det_boxes[det_labels == c]  # (n_class_detections, 4)
        det_class_scores = det_scores[det_labels == c]  # (n_class_detections)
        n_class_detections = det_class_boxes.size(0)

        if n_class_detections == 0:
            continue

        # Sort detections in decreasing order of confidence/scores.
        det_class_scores, sort_ind = torch.sort(det_class_scores, dim=0, descending=True)  # (n_class_detections)
        det_class_images = det_class_images[sort_ind]  # (n_class_detections)
        det_class_boxes = det_class_boxes[sort_ind]  # (n_class_detections, 4)

        # Initialize empty tensors (n_class_detections, 10) and scalars to count TP and FP
        true_positives = torch.zeros((n_class_detections, len(iou)), dtype=torch.float).to(device)
        false_positives = torch.zeros((n_class_detections, len(iou)), dtype=torch.float).to(device)
        tp_small = torch.zeros((n_class_detections, len(iou)), dtype=torch.float).to(device)
        tp_medium = torch.zeros((n_class_detections, len(iou)), dtype=torch.float).to(device)
        tp_large = torch.zeros((n_class_detections, len(iou)), dtype=torch.float).to(device)
        n_class_objects_small = 0
        n_class_objects_medium = 0
        n_class_objects_large = 0

        # Per class, count how many true objects are small, medium and large based on area.
        for i in range(len(true_class_areas)):
            if true_class_areas[i] < 32 ** 2:
                n_class_objects_small = n_class_objects_small + 1  # (n_class_objects_small)
            elif true_class_areas[i] > 96 ** 2:
                n_class_objects_large = n_class_objects_large + 1  # (n_class_objects_large)
            else:
                n_class_objects_medium = n_class_objects_medium + 1  # (n_class_objects_medium)

        # For each detection (per class).
        for d in range(n_class_detections):
            this_detection_box = det_class_boxes[d].unsqueeze(0)  # (1, 4)
            this_image = det_class_images[d]  # (), scalar

            # Find objects in the same image with this class and whether they have been detected before.
            object_boxes = true_class_boxes[true_class_images == this_image]  # (n_class_objects_in_img)
            # If no such object in this image, then the detection is a false positive.
            if object_boxes.size(0) == 0:
                false_positives[d] = 1
                continue

            # Find maximum overlap of this detection with objects in this image of this class.
            overlaps = find_jaccard_overlap(this_detection_box, object_boxes)  # (1, n_class_objects_in_img)
            max_overlap, ind = torch.max(overlaps.squeeze(0), dim=0)  # (), () - scalars
            # 'ind' is the index of the object in these image-level tensors 'object_boxes', 'object_difficulties'
            # In the original class-level tensors 'true_class_boxes', etc., 'ind' corresponds to object with index...
            # We need 'original_ind' to update 'true_class_boxes_detected'
            original_ind = torch.LongTensor(range(true_class_boxes.size(0)))[true_class_images == this_image][ind]

            # Determine the TP and FP for different IoU thresholds, for all detected objects (per class).
            for iou_th in range(len(iou)):
                # If the maximum overlap is greater than the threshold of 0.5 (for the first iteration), it's a match.
                if max_overlap.item() > iou[iou_th]:
                    # If this object has already not been detected, it's a true positive.
                    if true_class_boxes_detected[original_ind, iou_th] == 0:
                        true_positives[d, iou_th] = 1  # (n_class_detections, 10)
                        # Count the number of TP per surface area as well.
                        if true_class_areas[original_ind] < 32 ** 2:
                            tp_small[d, iou_th] = 1  # (n_class_detections, 10)
                        elif true_class_areas[original_ind] > 96 ** 2:
                            tp_large[d, iou_th] = 1  # (n_class_detections, 10)
                        else:
                            tp_medium[d, iou_th] = 1  # (n_class_detections, 10)
                        true_class_boxes_detected[original_ind, iou_th] = 1  # This object has now been detected
                    # Otherwise, it's a false positive (since this object is already accounted for).
                    else:
                        false_positives[d, iou_th] = 1  # (n_class_detections, 10)
                # Otherwise, the detection occurs in a different location than the actual object, thus a false positive.
                else:
                    false_positives[d, iou_th] = 1  # (n_class_detections, 10)

        # Store the counted number of objects per area per class.  (91)
        n_all_objects_small[c-1] = n_class_objects_small
        n_all_objects_medium[c-1] = n_class_objects_medium
        n_all_objects_large[c-1] = n_class_objects_large

        # Find cumulative number of TPs and FPs per class per IoU.  (n_class_detections, 10)
        cum_TP_all = torch.cumsum(true_positives, dim=0)
        cum_TP_small = torch.cumsum(tp_small, dim=0)
        cum_TP_medium = torch.cumsum(tp_medium, dim=0)
        cum_TP_large = torch.cumsum(tp_large, dim=0)
        cum_FP_all = torch.cumsum(false_positives, dim=0)
        # Transpose for easier calculations per IoU.  (10, n_class_detections, 10)
        cum_TP_all_transpose = torch.transpose(cum_TP_all, 0, 1)
        cum_TP_small_transpose = torch.transpose(cum_TP_small, 0, 1)
        cum_TP_medium_transpose = torch.transpose(cum_TP_medium, 0, 1)
        cum_TP_large_transpose = torch.transpose(cum_TP_large, 0, 1)
        cum_FP_all_transpose = torch.transpose(cum_FP_all, 0, 1)

        # We want to find the cumulative recall and precision for each class per IoU (total, small, medium and large).
        # (10, n_class_detections)
        cum_rec_all = torch.zeros((len(iou), n_class_detections), dtype=torch.float).to(device)
        cum_rec_small = torch.zeros((len(iou), n_class_detections), dtype=torch.float).to(device)
        cum_rec_medium = torch.zeros((len(iou), n_class_detections), dtype=torch.float).to(device)
        cum_rec_large = torch.zeros((len(iou), n_class_detections), dtype=torch.float).to(device)
        cum_prec_all = torch.zeros((len(iou), n_class_detections), dtype=torch.float).to(device)

        for iou_th in range(len(iou)):  # (10, n_class_detections)
            cum_rec_all[iou_th] = cum_TP_all_transpose[iou_th] / n_class_objects
            cum_rec_small[iou_th] = cum_TP_small_transpose[iou_th] / n_class_objects_small
            cum_rec_medium[iou_th] = cum_TP_medium_transpose[iou_th] / n_class_objects_medium
            cum_rec_large[iou_th] = cum_TP_large_transpose[iou_th] / n_class_objects_large
            cum_prec_all[iou_th] = cum_TP_all_transpose[iou_th] \
                / (cum_TP_all_transpose[iou_th] + cum_FP_all_transpose[iou_th])

            # Replace all NaNs with 0's (caused by 0 objects in a class).  (10, n_class_detections)
            cum_rec_all[iou_th][cum_rec_all[iou_th] != cum_rec_all[iou_th]] = 0

            recall_thresholds = torch.arange(start=0, end=1.01, step=.01).tolist()  # (101)
            precisions = torch.zeros((len(recall_thresholds)), dtype=torch.float).to(device)  # (101)
            for i, t in enumerate(recall_thresholds):
                recalls_above_t_all = cum_rec_all[iou_th] >= t
                if recalls_above_t_all.any():
                    precisions[i] = cum_prec_all[iou_th][recalls_above_t_all].max()
                else:
                    precisions[i] = 0.

            # Find the average precision and recall for each IoU threshold.  (10)
            ap_iou[iou_th] = precisions.mean()
            recall_iou[iou_th] = cum_rec_all[iou_th, -1]    # (take last cumulative value per IoU)
            recall_iou_small[iou_th] = cum_rec_small[iou_th, -1]
            recall_iou_medium[iou_th] = cum_rec_medium[iou_th, -1]
            recall_iou_large[iou_th] = cum_rec_large[iou_th, -1]

        # The average precision per class is the mean of AP per IoU (same for recall).  (n_classes)
        ap_class[c-1] = ap_iou.mean()
        classwise_recall[c-1] = recall_iou.mean()
        classwise_recall_small[c-1] = recall_iou_small.mean()
        classwise_recall_medium[c-1] = recall_iou_medium.mean()
        classwise_recall_large[c-1] = recall_iou_large.mean()


    # Total AP and recall is calculated based on the 80/91 used classes in the COCO dataset.
    # https://tech.amikelive.com/node-718/what-object-categories-labels-are-in-coco-dataset/
    ap_class_corrected = torch.cat(
        [ap_class[0:11], ap_class[12:25], ap_class[26:28],
         ap_class[30:44], ap_class[45:65], ap_class[66:67],
         ap_class[69:70], ap_class[71:82], ap_class[83:90]]
        )   # (80)
    classwise_recall_corrected = torch.cat(
        [classwise_recall[0:11], classwise_recall[12:25], classwise_recall[26:28],
         classwise_recall[30:44], classwise_recall[45:65], classwise_recall[66:67],
         classwise_recall[69:70], classwise_recall[71:82], classwise_recall[83:90]]
        )  # (80)
    classwise_recall_small_corrected = torch.cat(
        [classwise_recall_small[0:11], classwise_recall_small[12:25], classwise_recall_small[26:28],
         classwise_recall_small[30:44], classwise_recall_small[45:65], classwise_recall_small[66:67],
         classwise_recall_small[69:70], classwise_recall_small[71:82], classwise_recall_small[83:90]]
        )  # (80)
    classwise_recall_medium_corrected = torch.cat(
        [classwise_recall_medium[0:11], classwise_recall_medium[12:25], classwise_recall_medium[26:28],
         classwise_recall_medium[30:44], classwise_recall_medium[45:65], classwise_recall_medium[66:67],
         classwise_recall_medium[69:70], classwise_recall_medium[71:82], classwise_recall_medium[83:90]]
        )  # (80)
    classwise_recall_large_corrected = torch.cat(
        [classwise_recall_large[0:11], classwise_recall_large[12:25], classwise_recall_large[26:28],
         classwise_recall_large[30:44], classwise_recall_large[45:65], classwise_recall_large[66:67],
         classwise_recall_large[69:70], classwise_recall_large[71:82], classwise_recall_large[83:90]]
        )  # (80)

    # Some classes contain no objects with a small/medium/large area, which causes recall to be NaN.
    # Instead of setting those values to 0, they are excluded from calculating the mean over all classes.
    classwise_recall_small_corrected = torch.unsqueeze(classwise_recall_small_corrected, dim=1)
    classwise_recall_medium_corrected = torch.unsqueeze(classwise_recall_medium_corrected, dim=1)
    classwise_recall_large_corrected = torch.unsqueeze(classwise_recall_large_corrected, dim=1)

    classwise_recall_small_corrected = classwise_recall_small_corrected[
        ~torch.any(classwise_recall_small_corrected.isnan(), dim=1)]
    classwise_recall_medium_corrected = classwise_recall_medium_corrected[
        ~torch.any(classwise_recall_medium_corrected.isnan(), dim=1)]
    classwise_recall_large_corrected = classwise_recall_large_corrected[
        ~torch.any(classwise_recall_large_corrected.isnan(), dim=1)]

    # The total recall is found by calculating the recall over all classes per IoU and taking the mean of those.
    all_recall_by_average = classwise_recall_corrected.mean().item()
    recall_small_by_average = classwise_recall_small_corrected.mean().item()
    recall_medium_by_average = classwise_recall_medium_corrected.mean().item()
    recall_large_by_average = classwise_recall_large_corrected.mean().item()

    # Calculate Mean Average Precision (mAP).
    mean_average_precision = ap_class_corrected.mean().item()
    classwise_recall = {coco_rev_label_map[c+1]: v for c, v in enumerate(classwise_recall.tolist())}
    average_precisions = {coco_rev_label_map[c+1]: v for c, v in enumerate(ap_class.tolist())}

    return average_precisions, mean_average_precision, classwise_recall, all_recall_by_average,\
        recall_small_by_average, recall_medium_by_average, recall_large_by_average