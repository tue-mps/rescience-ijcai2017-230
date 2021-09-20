import torchvision
from Datasets.datasets import VocDataset
from Utils.testing import *
from Utils.metrics import *
import numpy as np

# CONFIG
matrix_type = 'KG-CNet-55-VOC'  # choose between 'KF-All-VOC', 'KF-500-VOC', 'KG-CNet-57-VOC' or 'KG-CNet-55-VOC'
detections_per_image = 500  # the maximum number of detected objects per image
num_iterations = 10         # number of iterations to calculate p_hat
box_score_threshold = 1e-5  # minimum score for a bounding box to be kept as detection (default = 0.05)
bk = 5                      # number of neighbouring bounding boxes to consider for p_hat
lk = 5                      # number of largest semantic consistent classes to consider for p_hat
epsilon = 0.9               # trade-off parameter for traditional detections and knowledge aware detections
topk = 100                  # maximum number of detections to be considered for metrics (recall@k / mAP@k)

"""
Running this file will output the mAP@k and recall@k per class and averaged for the PASCAL VOC 2007 testset. Above
configurations will output the results as mentioned in the paper. A different semantic consistency matrix can be
selected to generate the results for different approaches.  
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

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False,
                                                                 progress=True,
                                                                 num_classes=num_classes,
                                                                 pretrained_backbone=True,
                                                                 trainable_backbone_layers=1,
                                                                 box_detections_per_img=detections_per_image,
                                                                 box_score_thresh=box_score_threshold
                                                                 )
    model.to(device)

    file_path = os.path.join(path_project, "Model Training/Trained models/voc-FRCNN-48e.pth")
    checkpoint = torch.load(file_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    model.eval()

    # for j in range(len(epsilons)):
    #     epsilon = epsilons[j]

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
