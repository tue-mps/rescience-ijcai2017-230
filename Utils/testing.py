import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def find_p_hat(boxes, predictions, bk, lk, S, num_iterations, epsilon):
    """
    Compute the knowledge aware predictions, based on the object detector's output and semantic consistency.
    :param boxes: tensor of bounding boxes
    :param predictions: tensor of prediction scores
    :param bk: number of neighbouring bounding boxes to consider for p_hat
    :param lk: number of largest semantic consistent classes to consider for p_hat
    :param S: semantic consistency matrix
    :param num_iterations: number of iterations to calculate p_hat
    :param epsilon: trade-off parameter for traditional detections and knowledge aware detections
    :return p_hat: tensor of knowledge aware predictions
    """

    num_boxes = predictions.shape[0]
    num_classes = predictions.shape[1]

    if num_boxes <= 1:
        return predictions

    if num_boxes <= bk:
        bk = num_boxes - 1

    if num_classes <= lk:
        lk = num_classes

    box_centers = torch.empty(size=(num_boxes, 2), dtype=torch.double).to(device)
    box_centers[:, 0] = ((boxes[:, 2] - boxes[:, 0]) / 2) + boxes[:, 0]
    box_centers[:, 1] = ((boxes[:, 3] - boxes[:, 1]) / 2) + boxes[:, 1]

    box_nearest = torch.empty(size=(num_boxes, bk), dtype=torch.long).to(device)
    for i in range(len(boxes)):
        box_center = box_centers[i]
        distances = torch.sqrt((box_center[0] - box_centers[:, 0]) ** 2 + (box_center[1] - box_centers[:, 1]) ** 2)
        distances[i] = float('inf')
        box_nearest[i] = torch.argsort(distances)[0:bk]

    S_highest = torch.zeros(size=(num_classes, num_classes), dtype=torch.double).to(device)
    for i in range(len(S)):
        S_args = torch.argsort(S[i])[-lk:]
        S_highest[i, S_args] = S[i, S_args]

    p_hat_init = torch.full(size=(num_boxes, num_classes), fill_value=(1 / num_classes), dtype=torch.double).to(device)
    p_hat = p_hat_init
    for i in range(num_iterations):
        p_hat_temp = torch.clone(p_hat)
        for b in range(num_boxes):
            p = predictions[b]
            num = torch.sum(torch.mm(S_highest, torch.transpose(p_hat_temp[box_nearest[b]], 0, 1)), 1)
            denom = torch.sum(S_highest, dim=1) * bk
            p_hat[b] = (1 - epsilon) * torch.squeeze(torch.div(num, denom)) + epsilon * p
            p_hat[b] = torch.nan_to_num(p_hat[b])

    return p_hat


def find_top_k(predictions, boxes, k):
    """
    Find the top k highest scoring predictions
    :param predictions: tensor of prediction scores
    :param boxes: tensor of bounding boxes
    :param k: (maximum) number of object to return
    :return predictions2: k amount of highest scoring predictions
    :return boxes2: k amount of bounding boxes corresponding to the highest scoring predictions
    :return labels2: k amount of labels corresponding to the highest scoring predictions
    :return scores2: k amount of scores corresponding to the highest scoring predictions
    """

    if predictions.shape[0] == 0:
        predictions2 = torch.Tensor([]).to(device)
        labels2 = torch.Tensor([]).to(device)
        boxes2 = torch.Tensor([]).to(device)
        scores2 = torch.Tensor([]).to(device)

    else:
        predictions0 = predictions
        scores0 = torch.max(predictions0, dim=1)[0]
        labels0 = torch.argmax(predictions0, dim=1)
        boxes0 = boxes

        sort = torch.argsort(scores0, descending=True)
        boxes1, labels1, scores1, predictions1 = boxes0[sort], labels0[sort], scores0[sort], predictions0[sort]

        boxes2, labels2, scores2, predictions2 = boxes1[:k], labels1[:k] + 1, scores1[:k], predictions1[:k]

    return predictions2, boxes2, labels2, scores2


def test_function_kg(test_loader, model, settype, S, lk, bk, num_iters, epsilon, num_classes, topk):
    """
    Iterate over the batches in the test set data loader, run the model to get bounding box, label and score predictions.
    :param test_loader: (pytorch) data_loader containing the test data
    :param model: the model. make sure it is in evaluation mode
    :param settype: either 'voc' or 'coco'
    :param bk: number of neighbouring bounding boxes to consider for p_hat
    :param lk: number of largest semantic consistent classes to consider for p_hat
    :param S: semantic consistency matrix
    :param num_iters: number of iterations to calculate p_hat
    :param num_classes: the number of classes (excluding background)
    :param topk: maximum number of detections to be considered for metrics (recall@k / mAP@k)
    :param epsilon: trade-off parameter for traditional detections and knowledge aware detections
    :return det_boxes: list of tensors, one tensor for each image containing detected objects' bounding boxes
    :return det_labels: list of tensors, one tensor for each image containing detected objects' labels
    :return det_scores: list of tensors, one tensor for each image containing detected objects' labels' scores
    :return true_boxes: list of tensors, one tensor for each image containing actual objects' bounding boxes
    :return true_labels: list of tensors, one tensor for each image containing actual objects' labels
    :return true_difficulties: list of tensors, one tensor for each image containing actual objects' difficulty (0 or 1) (VOC only, zeros for COCO)
    :return true_areas: list of tensors, one tensor for each image containing actual objects' areas (COCO only, zeros for VOC)
    Source: https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/blob/master/eval.py (adapted)
    """

    det_boxes = []
    det_labels = []
    det_scores = []
    true_boxes = []
    true_labels = []
    true_difficulties = []
    true_areas = []

    with torch.no_grad():
        for i, (images, targets) in enumerate(test_loader):

            # Move to default device.
            images = [im.to(device) for im in images]

            # Some (1021) images of COCO contain no objects at all.  These are filtered out in the data loader, but
            # return an empty list, which raises an error in the model, so they are skipped.
            if len(images) == 0:
                continue

            prediction = model(images)

            for p in range(len(prediction)-1):
                true_boxes.append(targets[p]['boxes'].to(device))
                true_labels.append(targets[p]['labels'].to(device))

                if settype == 'voc':
                    true_difficulties.append(targets[p]['difficulties'].to(device))
                    # true_difficulties.append(torch.zeros(len(targets[p]['boxes'])).to(device))
                    true_areas.append(torch.zeros(len(targets[p]['boxes'])).to(device))
                else:
                    true_difficulties.append(torch.zeros(len(targets[p]['boxes'])).to(device))
                    true_areas.append(targets[p]['areas'].to(device))

            boxes_temp = prediction[1][0]['boxes']
            labels_temp = prediction[1][0]['labels']
            scores_temp = prediction[1][0]['scores']

            new_predictions = torch.zeros((boxes_temp.shape[0], num_classes)).to(device)

            for l in range(new_predictions.shape[0]):
                label = labels_temp[l] - 1
                new_predictions[l, label] = scores_temp[l]

            p_hat = find_p_hat(boxes_temp, new_predictions, bk, lk, S, num_iters, epsilon)

            predk, boxk, labk, scok = find_top_k(p_hat, boxes_temp, topk)

            det_boxes.append(boxk)
            det_labels.append(labk)
            det_scores.append(scok)

            del prediction
            torch.cuda.empty_cache()

    return det_boxes, det_labels, det_scores, true_boxes, true_labels, true_difficulties, true_areas
