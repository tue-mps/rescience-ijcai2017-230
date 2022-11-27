import torchvision
from Datasets.datasets import VocDataset
from Utils.testing import *
from Utils.metrics import *
import numpy as np
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.backbone_utils import BackboneWithFPN, resnet_fpn_backbone

# CONFIG
model_type = 'voc-FRCNN-resnet50'   # choose between 'voc-FRCNN-resnet50', 'voc-FRCNN-resnet18' or 'voc-FRCNN-vgg16'
matrix_type = 'KG-CNet-55-VOC'      # choose between 'KF-All-VOC', 'KF-500-VOC', 'KG-CNet-57-VOC' or 'KG-CNet-55-VOC'
detections_per_image = 500          # the maximum number of detected objects per image
num_iterations = 10                 # number of iterations to calculate p_hat
box_score_threshold = 1e-5          # minimum score for a bounding box to be kept as detection (default = 0.05)
bk = 5                              # number of neighbouring bounding boxes to consider for p_hat
lk = 5                              # number of largest semantic consistent classes to consider for p_hat
epsilon = 1.0                       # trade-off parameter for traditional detections and knowledge aware detections
topk = 100                          # maximum number of detections to be considered for metrics (recall@k / mAP@k)

"""
Running this file will output the mAP@k and recall@k per class and averaged for the PASCAL VOC 2007 testset. Above
configurations will output the results as mentioned in the paper. A different model backbone and semantic consistency
matrix can be selected to generate the results for different approaches.  
"""



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':

    # load in test set
    batch_size_test = 1
    workers = 2

    voc_path = os.path.join(path_project, "Data raw/VOC2007")
    output_folder = os.path.join(path_project, "Datasets")
    create_data_lists(voc07_path=voc_path, output_folder=output_folder)

    voc_test_dataset = VocDataset(output_folder, split='test')
    voc_test_dataloader = torch.utils.data.DataLoader(
        voc_test_dataset,
        batch_size=batch_size_test,
        shuffle=False,
        collate_fn=voc_test_dataset.collate_fn,
        num_workers=workers,
        pin_memory=True
        )

    print("PASCAL test set: \t\t", voc_test_dataset.__len__(), " Images")

    # load in consistency matrices

    file_path = os.path.join(path_project, "Semantic Consistency/Stored matrices/CM_freq_info.json")
    if os.path.isfile(file_path):
        print("Loading frequency based consistency matrix")
        with open(file_path, 'r') as j:
            info = json.load(j)
        KF_All_VOC_info = info['KF_All_VOC_info']
        KF_500_VOC_info = info['KF_500_VOC_info']
        S_KF_All_VOC = np.asarray(KF_All_VOC_info['S'])
        S_KF_500_VOC = np.asarray(KF_500_VOC_info['S'])
    else:
        print("No matrix available")

    file_path = os.path.join(path_project, "Semantic Consistency/Stored matrices/CM_kg_55_info.json")
    if os.path.isfile(file_path):
        print("Loading knowledge based consistency matrix")
        with open(file_path, 'r') as j:
            info = json.load(j)
        KG_VOC_info = info['KG_VOC_info']
        S_KG_55_VOC = np.asarray(KG_VOC_info['S'])
    else:
        print("No matrix available")

    file_path = os.path.join(path_project, "Semantic Consistency/Stored matrices/CM_kg_57_info.json")
    if os.path.isfile(file_path):
        print("Loading knowledge based consistency matrix")
        with open(file_path, 'r') as j:
            info = json.load(j)
        KG_VOC_info = info['KG_VOC_info']
        S_KG_57_VOC = np.asarray(KG_VOC_info['S'])
    else:
        print("No matrix available")

    if matrix_type == 'KF-All-VOC':
        S = torch.from_numpy(S_KF_All_VOC).to(device)
    elif matrix_type == 'KF-500-VOC':
        S = torch.from_numpy(S_KF_500_VOC).to(device)
    elif matrix_type == 'KG-CNet-57-VOC':
        S = torch.from_numpy(S_KG_57_VOC).to(device)
    elif matrix_type == 'KG-CNet-55-VOC':
        S = torch.from_numpy(S_KG_55_VOC).to(device)
    else:
        print("Wrong matrix type selected")


    settype = 'voc'
    num_classes = 21
    test_loader = voc_test_dataloader

    if model_type == 'voc-FRCNN-resnet50':

        backbone = resnet_fpn_backbone('resnet50', pretrained=True, trainable_layers=5)

        backbone.out_channels = 256

        anchor_generator = AnchorGenerator(sizes=((32,), (64,), (128,), (256,), (512,)),
                                           aspect_ratios=(
                                               (0.5, 1.0, 2.0), (0.5, 1.0, 2.0), (0.5, 1.0, 2.0), (0.5, 1.0, 2.0),
                                               (0.5, 1.0, 2.0)))

        roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0', '1', '2', '3'],
                                                        output_size=7,
                                                        sampling_ratio=2)

        model = FasterRCNN(backbone,
                           num_classes=num_classes,
                           rpn_anchor_generator=anchor_generator,
                           box_roi_pool=roi_pooler,
                           box_detections_per_img=detections_per_image,
                           box_score_thresh=box_score_threshold)

    elif model_type == 'voc-FRCNN-vgg16':
        backbone_vgg = torchvision.models.vgg16(pretrained=False).features
        out_channels = 512
        in_channels_list = [128, 256, 512, 512]
        return_layers = {'9': '0', '16': '1', '23': '2', '30': '3'}
        backbone = BackboneWithFPN(backbone_vgg, return_layers, in_channels_list, out_channels)
        backbone.out_channels = 512

        anchor_generator = AnchorGenerator(sizes=((32,), (64,), (128,), (256,), (512,)),
                                           aspect_ratios=(
                                           (0.5, 1.0, 2.0), (0.5, 1.0, 2.0), (0.5, 1.0, 2.0), (0.5, 1.0, 2.0),
                                           (0.5, 1.0, 2.0)))

        roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0', '1', '2', '3'],
                                                        output_size=7,
                                                        sampling_ratio=2)
        model = FasterRCNN(backbone,
                           num_classes=num_classes,
                           rpn_anchor_generator=anchor_generator,
                           box_roi_pool=roi_pooler,
                           box_detections_per_img=detections_per_image,
                           box_score_thresh=box_score_threshold)

    elif model_type == 'voc-FRCNN-resnet18':
        backbone = resnet_fpn_backbone('resnet18', pretrained=True, trainable_layers=5)

        backbone.out_channels = 256

        anchor_generator = AnchorGenerator(sizes=((32,), (64,), (128,), (256,), (512,)),
                                           aspect_ratios=(
                                           (0.5, 1.0, 2.0), (0.5, 1.0, 2.0), (0.5, 1.0, 2.0), (0.5, 1.0, 2.0),
                                           (0.5, 1.0, 2.0)))

        roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0', '1', '2', '3'],
                                                        output_size=7,
                                                        sampling_ratio=2)

        model = FasterRCNN(backbone,
                           num_classes=num_classes,
                           rpn_anchor_generator=anchor_generator,
                           box_roi_pool=roi_pooler,
                           box_detections_per_img=detections_per_image,
                           box_score_thresh=box_score_threshold)
    else:
        print("please select a valid model")

    model.to(device)

    model_path = "Model Training/Trained models/" + model_type + ".pth"
    file_path = os.path.join(path_project, model_path)

    checkpoint = torch.load(file_path)
    model.load_state_dict(checkpoint['model_state_dict'])


    model.eval()

    print("Currently testing: ")
    print("threshold = ", box_score_threshold, "bk = ", bk, "lk = ", lk, "epsilon = ", epsilon, "S = ",
          matrix_type, "top k = ", topk)

    det_boxes, det_labels, det_scores, \
    true_boxes, true_labels, true_difficulties, true_areas = test_function_kg(test_loader, model, settype,
                                                                              S, lk, bk, num_iterations, epsilon,
                                                                              num_classes - 1, topk)

    average_precisions, mean_average_precision, classwise_recall, all_recall, \
    all_recall_by_average = voc_metrics(det_boxes, det_labels, det_scores, true_boxes,
                                        true_labels, true_difficulties, settype
                                        )

    print("AP @", topk, " per class: ", average_precisions)
    print("mAP @", topk, " : ", mean_average_precision)
    print("Recall @", topk, " per class: ", classwise_recall)
    print("Recall @", topk, " all classes (by average): ", all_recall_by_average)
