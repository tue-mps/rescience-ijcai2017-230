import numpy as np
import sys
from Datasets.datasets import VocDataset, CocoDetection
from Utils.dataloading import *

"""
Running this file will count the instances and co-occurrences in the VOC and COCO datasets. 4 matrices are extracted
from this information, namely KF-All-VOC and KF-All-COCO, which use both full datasets to count the frequencies of
co-occurrences. The other two are KF-500-VOC and KF-500-COCO, which only use 250 images of each dataset.
The resulting matrices will be stored in dictionary format in a JSON file. If the script does not create the JSON file
automatically, you should create the file with the correct name manually, in the Stored matrices folder.
"""


class NumpyEncoder(json.JSONEncoder):
    """
    Convert Numpy arrays to JSON
    Source: https://pynative.com/python-serialize-numpy-ndarray-into-json/
    """
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def count_a_dataset(dataloader, label_occurrences, total_instances, co_occurrence_matrix, num_classes, all):
    """
    Loop over a given dataset per image and count all object occurrences of each class and the number of co-occurrences
    between classes (or self-occurrence between the same class). A co-occurrence is defined as a unique combination between
    two objects. For self-occurrence this is the Gauss sum of the number of objects of that class minus 1. For co-occurrences
    this is the multiplication of occurrences of each classes' objects.
    :param dataloader: (pytorch) data_loader containing images and annotations
    :param label_occurrences: current count of occurrences of each class (initially 0)
    :param total_instances: current count of total instances/objects (initially 0)
    :param co_occurrence_matrix: current count of co-occurrences between classes (initially 0)
    :param num_classes: the number of classes
    :param all: boolean value, if False, only 250 images are considered of the dataset, otherwise, all images are considered
    :return label_occurrences: updated count of occurrences of each class
    :return total_instances: updated count of total instances/objects
    :return co_occurrence_matrix: updated count of co-occurrences between classes
    """
    # Per image in the dataset.
    for x in enumerate(dataloader):
        i, (images, targets) = x
        if len(images) == 0:
            continue
        labels = targets[0]['labels']
        # Count the objects in the image and assign them to the correct label.
        for label in labels:
            label_occurrences[label-1] = label_occurrences[label-1] + 1
            total_instances = total_instances + 1
        # For each combination of classes, check if any are in the image.
        for l1 in range(0, num_classes):
            if l1 + 1 not in labels:
                continue
            else:
                for l2 in range(0, num_classes):
                    if l2 + 1 not in labels:
                        continue
                    # Count the number of occurrences of both objects, compute handshake equation. If self-occurrence,
                    # add the handshake. If co-occurrence, add product of both object occurrences. This assures we count
                    # the unique co-occurrences.
                    else:
                        n_l1 = np.count_nonzero(labels.numpy() == (l1+1))
                        sum_l1 = (n_l1 * (n_l1-1)) / 2
                        n_l2 = np.count_nonzero(labels.numpy() == (l2+1))
                        if l1 == l2:
                            co_occurrence_matrix[l1, l2] = co_occurrence_matrix[l1, l2] + sum_l1
                        else:
                            co_occurrence_matrix[l1, l2] = co_occurrence_matrix[l1, l2] + (n_l1 * n_l2)
        # If 'all' is False, stop after the 250th image.
        if not all and i == 249:
            break

    print("number of instances: ", total_instances)
    print("number of summed label occurrences: ", label_occurrences.sum())
    print("number of summed co_occurrences: ", co_occurrence_matrix.sum())
    return label_occurrences, total_instances, co_occurrence_matrix


def consistency_matrix(num_classes, label_occurrences, total_instances, co_occurrence_matrix):
    """
    Compute the consistency matrix, with the given co-occurrence matrix, label occurrences and total number of instances
    S = max(log((n(l1,l2)*N / (n(l1)*n(l2)), 0)
    :param num_classes: the number of classes
    :param label_occurrences: number of occurrences of each class (n(l))
    :param total_instances: number of total instances/objects (N)
    :param co_occurrence_matrix: number of co-occurrences between classes (n(l1,l2))
    :return s: consistency matrix
    """

    # For each combination of classes, compute S.
    s = np.zeros((num_classes, num_classes))
    for l1 in range(0, num_classes):
        for l2 in range(0, num_classes):
            sys_cos = np.log10(
                (co_occurrence_matrix[l1, l2] * total_instances) / (label_occurrences[l1] * label_occurrences[l2]))
            s[l1, l2] = max(sys_cos, 0)
    s[s != s] = 0  # replace NaN by 0
    return s


def convert_coco_to_voc(co_occurrence_matrix_voc, co_occurrence_matrix_coco, label_occurrences_voc,
                        label_occurrences_coco, total_instances_voc, total_instances_coco):
    """
    Combine the info from VOC and COCO together in VOC format
    :param co_occurrence_matrix_voc: number of co-occurrences from the VOC dataset
    :param co_occurrence_matrix_coco: number of co-occurrences from the COCO dataset
    :param label_occurrences_voc: number of occurrences per class from the VOC dataset
    :param label_occurrences_coco: number of occurrences per class from the COCO dataset
    :param total_instances_voc: number of total instances/objects from the VOC dataset
    :param total_instances_coco: number of total instances/objects from the COCO dataset
    :return co_occurrence_matrix_cv: combined co-occurrences in VOC format
    :return label_occurrences_cv: combined label occurrences in VOC format
    :return total_instances_cv: combined total instances/objects
    """

    co_occurrence_matrix_temp = np.zeros((20, 20))
    label_occurrences_temp = np.zeros(20)
    # The label numbers of the COCO dataset which overlap VOC's objects.  The first VOC class is aeroplane, which is
    # equal to COCO's 5th class airplane, etc.
    coco_to_voc = [5, 2, 16, 9, 44, 6, 3, 17, 62, 21, 67, 18, 19, 4, 1, 64, 20, 63, 7, 72]
    total_instances_cv = total_instances_voc + total_instances_coco

    # Create a temporary matrix in VOC format taken from the COCO information.  From (91,91) of COCO, go to (20,20)
    # containing only relevant VOC data.
    for l1 in range(0, 20):
        label_occurrences_temp[l1] = label_occurrences_coco[coco_to_voc[l1]-1]
        for l2 in range(0, 20):
            co_occurrence_matrix_temp[l1, l2] = co_occurrence_matrix_coco[
                coco_to_voc[l1]-1, coco_to_voc[l2]-1]

    # Sum original VOC data with the converted COCO to VOC data to get combined results.
    co_occurrence_matrix_cv = co_occurrence_matrix_voc + co_occurrence_matrix_temp
    label_occurrences_cv = label_occurrences_voc + label_occurrences_temp

    return co_occurrence_matrix_cv, label_occurrences_cv, total_instances_cv


def convert_voc_to_coco(co_occurrence_matrix_voc, co_occurrence_matrix_coco, label_occurrences_voc,
                        label_occurrences_coco, total_instances_voc, total_instances_coco):
    """
    Combine the info from VOC and COCO together in COCO format
    :param co_occurrence_matrix_voc: number of co-occurrences from the VOC dataset
    :param co_occurrence_matrix_coco: number of co-occurrences from the COCO dataset
    :param label_occurrences_voc: number of occurrences per class from the VOC dataset
    :param label_occurrences_coco: number of occurrences per class from the COCO dataset
    :param total_instances_voc: number of total instances/objects from the VOC dataset
    :param total_instances_coco: number of total instances/objects from the COCO dataset
    :return co_occurrence_matrix_vc: combined co-occurrences in COCO format
    :return label_occurrences_vc: combined label occurrences in COCO format
    :return total_instances_vc: combined total instances/objects
    """
    co_occurrence_matrix_temp = np.zeros((91, 91))
    label_occurrences_temp = np.zeros(91)
    voc_to_coco = [5, 2, 16, 9, 44, 6, 3, 17, 62, 21, 67, 18, 19, 4, 1, 64, 20, 63, 7, 72]
    total_instances_vc = total_instances_coco + total_instances_voc

    for l1 in voc_to_coco:
        label_occurrences_temp[l1-1] = label_occurrences_voc[voc_to_coco.index(l1)]
        for l2 in voc_to_coco:
            co_occurrence_matrix_temp[l1-1, l2-1] = co_occurrence_matrix_voc[voc_to_coco.index(l1),
                                                                             voc_to_coco.index(l2)
                                                                             ]

    co_occurrence_matrix_vc = co_occurrence_matrix_coco + co_occurrence_matrix_temp
    label_occurrences_vc = label_occurrences_coco + label_occurrences_temp

    return co_occurrence_matrix_vc, label_occurrences_vc, total_instances_vc


def convert_91_to_80(S):
    """
    Set the 11 irrelevant COCO classes to 0 in the matrix
    :param S: the 91x91 matrix
    :return S: Sparse 91x91 matrix
    """
    leftouts = [11, 25, 28, 29, 44, 65, 67, 68, 70, 82, 90]

    for i in range(S.shape[0]):
        for j in range(S.shape[1]):
            if i in leftouts:
                S[i,j] = 0
            if j in leftouts:
                S[i,j] = 0

    return S


if __name__ == '__main__':


    # load in the datasets
    batch_size = 1
    workers = 2
    np.set_printoptions(threshold=sys.maxsize)  # don't truncate printing

    path_project = os.path.abspath(os.path.join(__file__, "../.."))
    voc_path = os.path.join(path_project, "Data raw/VOC2007")
    output_folder = os.path.join(path_project, "Datasets")
    create_data_lists(voc07_path=voc_path, output_folder=output_folder)

    # Load in all datasets.
    voc_train_dataset = VocDataset(output_folder, split='train')
    voc_train_dataloader = torch.utils.data.DataLoader(
        voc_train_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=voc_train_dataset.collate_fn,
        num_workers=workers,
        pin_memory=True
        )

    voc_validation_dataset = VocDataset(output_folder, split='validation')
    voc_validation_dataloader = torch.utils.data.DataLoader(
        voc_validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=voc_validation_dataset.collate_fn,
        num_workers=workers,
        pin_memory=True
        )

    coco_train_imgFile = os.path.join(path_project, "Data raw/COCO2014/train2014")
    coco_train_annFile = os.path.join(path_project, "Data raw/COCO2014/annotations/instances_train2014.json")
    coco_train_dataset = CocoDetection(root=coco_train_imgFile, annFile=coco_train_annFile)
    coco_validation_imgFile = os.path.join(path_project, "Data raw/COCO2014/val2014")
    coco_validation_annFile = os.path.join(path_project, "Data raw/COCO2014/annotations/instances_val2014.json")
    coco_validation_dataset = CocoDetection(root=coco_validation_imgFile, annFile=coco_validation_annFile)
    coco_combo_dataset = coco_validation_dataset + coco_train_dataset

    coco_minival_1k, coco_minival_4k, coco_trainset = torch.utils.data.random_split(
                                                        coco_combo_dataset,
                                                        [1000, 4000, 118287],
                                                        generator=torch.Generator().manual_seed(42)
                                                        )

    generator = torch.Generator().initial_seed()  # Set manual seed back to default random seed

    coco_train_dataloader = torch.utils.data.DataLoader(
        coco_trainset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        collate_fn=CocoDetection.collate_fn
    )
    coco_validation_dataloader = torch.utils.data.DataLoader(
        coco_minival_1k,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        collate_fn=CocoDetection.collate_fn
    )


    # VOC_only_all (VOA):
    # All images from VOC trainloader and VOC validationloader.

    # Initialize variables.
    num_classes = 20
    co_occurrence_matrix_VOA = np.zeros((num_classes, num_classes))
    label_occurrences_VOA = np.zeros(num_classes)
    total_instances_VOA = 0
    all_images = True      # If False, only 250 images are used

    # Store intermediate results in _int variables.  They are used later for combining datasets
    label_occurrences_VOA_int, total_instances_VOA_int, co_occurrence_matrix_VOA_int = \
        count_a_dataset(voc_train_dataloader, label_occurrences_VOA, total_instances_VOA,
                        co_occurrence_matrix_VOA, num_classes, all_images)

    # Use _int variables as input for second dataset, to get final results.
    label_occurrences_VOA, total_instances_VOA, co_occurrence_matrix_VOA = \
        count_a_dataset(voc_validation_dataloader, label_occurrences_VOA_int, total_instances_VOA_int,
                        co_occurrence_matrix_VOA_int, num_classes, all_images)

    # VOC_only_500 (VO5):
    # 250 images from VOC trainloader 250 images from VOC validationloader.

    num_classes = 20
    co_occurrence_matrix_VO5 = np.zeros((num_classes, num_classes))
    label_occurrences_VO5 = np.zeros(num_classes)
    total_instances_VO5 = 0
    all_images = False

    label_occurrences_VO5_int, total_instances_VO5_int, co_occurrence_matrix_VO5_int = \
        count_a_dataset(voc_train_dataloader, label_occurrences_VO5, total_instances_VO5,
                        co_occurrence_matrix_VO5, num_classes, all_images)

    label_occurrences_VO5, total_instances_VO5, co_occurrence_matrix_VO5 = \
        count_a_dataset(voc_validation_dataloader, label_occurrences_VO5_int, total_instances_VO5_int,
                        co_occurrence_matrix_VO5_int, num_classes, all_images)

    # COCO_only_all (COA):
    # All images from COCO trainloader and COCO validationloader.

    num_classes = 91
    co_occurrence_matrix_COA = np.zeros((num_classes, num_classes))
    label_occurrences_COA = np.zeros(num_classes)
    total_instances_COA = 0
    all_images = True

    label_occurrences_COA_int, total_instances_COA_int, co_occurrence_matrix_COA_int = \
        count_a_dataset(coco_train_dataloader, label_occurrences_COA, total_instances_COA,
                        co_occurrence_matrix_COA, num_classes, all_images)

    label_occurrences_COA, total_instances_COA, co_occurrence_matrix_COA = \
        count_a_dataset(coco_validation_dataloader, label_occurrences_COA_int, total_instances_COA_int,
                        co_occurrence_matrix_COA_int, num_classes, all_images)

    # COCO_only_500 (CO5):
    # 250 images from COCO trainloader 250 images from COCO validationloader.

    num_classes = 91
    co_occurrence_matrix_CO5 = np.zeros((num_classes, num_classes))
    label_occurrences_CO5 = np.zeros(num_classes)
    total_instances_CO5 = 0
    all_images = False

    label_occurrences_CO5_int, total_instances_CO5_int, co_occurrence_matrix_CO5_int = \
        count_a_dataset(coco_train_dataloader, label_occurrences_CO5, total_instances_CO5,
                        co_occurrence_matrix_CO5, num_classes, all_images)

    label_occurrences_CO5, total_instances_CO5, co_occurrence_matrix_CO5 = \
        count_a_dataset(coco_validation_dataloader, label_occurrences_CO5_int, total_instances_CO5_int,
                        co_occurrence_matrix_CO5_int, num_classes, all_images)

    # Matrices to be stored:

    # KF-All-VOC (FAV)
    # All images from VOC trainloader and all images from COCO trainloader (adapted to VOC).

    num_classes = 20

    co_occurrence_matrix_FAV, label_occurrences_FAV, total_instances_FAV = \
        convert_coco_to_voc(co_occurrence_matrix_VOA_int, co_occurrence_matrix_COA_int, label_occurrences_VOA_int,
                            label_occurrences_COA_int, total_instances_VOA_int, total_instances_COA_int)

    S_FAV = consistency_matrix(num_classes, label_occurrences_FAV, total_instances_FAV, co_occurrence_matrix_FAV)

    print("\nKF-All-VOC \n")
    print("co_occurrence_matrix: \n", co_occurrence_matrix_FAV.astype(int).shape)
    print("non-zero units in co_occurrence_matrix: \n", np.count_nonzero(co_occurrence_matrix_FAV.astype(int)))
    print("label_occurrences: ", label_occurrences_FAV)
    print("label_occurrences: ", np.sum(label_occurrences_FAV))
    print("total_instances: ", total_instances_FAV)
    print("consistency matrix: \n", S_FAV.shape)
    print("non-zero units in consistency matrix: \n", np.count_nonzero(S_FAV))

    KF_All_VOC_info = {}
    KF_All_VOC_info['co_occurrence_matrix'] = co_occurrence_matrix_FAV
    KF_All_VOC_info['label_occurrences'] = label_occurrences_FAV
    KF_All_VOC_info['total_instances'] = [total_instances_FAV]
    KF_All_VOC_info['S'] = S_FAV

    # KF-500-VOC (F5V):
    # 250 images from VOC trainloader and 250 images from COCO trainloader (adapted to VOC).

    num_classes = 20

    co_occurrence_matrix_F5V, label_occurrences_F5V, total_instances_F5V = \
        convert_coco_to_voc(co_occurrence_matrix_VO5_int, co_occurrence_matrix_CO5_int, label_occurrences_VO5_int,
                            label_occurrences_CO5_int, total_instances_VO5_int, total_instances_CO5_int)

    S_F5V = consistency_matrix(num_classes, label_occurrences_F5V, total_instances_F5V, co_occurrence_matrix_F5V)

    print("\nKF-500-VOC \n")
    print("co_occurrence_matrix: \n", co_occurrence_matrix_F5V.astype(int).shape)
    print("non-zero units in co_occurrence_matrix: \n", np.count_nonzero(co_occurrence_matrix_F5V.astype(int)))
    print("label_occurrences: ", label_occurrences_F5V)
    print("label_occurrences: ", np.sum(label_occurrences_F5V))
    print("total_instances: ", total_instances_F5V)
    print("consistency matrix: \n", S_F5V.shape)
    print("non-zero units in consistency matrix: \n", np.count_nonzero(S_F5V))

    KF_500_VOC_info = {}
    KF_500_VOC_info['co_occurrence_matrix'] = co_occurrence_matrix_F5V
    KF_500_VOC_info['label_occurrences'] = label_occurrences_F5V
    KF_500_VOC_info['total_instances'] = [total_instances_F5V]
    KF_500_VOC_info['S'] = S_F5V

    # KF-All-COCO (FAC)
    # All images from COCO trainloader and all images from VOC trainloader (adapted to COCO).

    num_classes = 91

    co_occurrence_matrix_FAC, label_occurrences_FAC, total_instances_FAC = \
        convert_voc_to_coco(co_occurrence_matrix_VOA_int, co_occurrence_matrix_COA_int, label_occurrences_VOA_int,
                            label_occurrences_COA_int, total_instances_VOA_int, total_instances_COA_int)

    S_FAC = consistency_matrix(num_classes, label_occurrences_FAC, total_instances_FAC, co_occurrence_matrix_FAC)
    S_FAC = convert_91_to_80(S_FAC)  # if you want the full 91 concepts, comment this line

    print("\nKF-All-COCO \n")
    print("co_occurrence_matrix: \n", co_occurrence_matrix_FAC.astype(int).shape)
    print("non-zero units in co_occurrence_matrix: \n", np.count_nonzero(co_occurrence_matrix_FAC.astype(int)))
    print("label_occurrences: ", label_occurrences_FAC)
    print("label_occurrences: ", np.sum(label_occurrences_FAC))
    print("total_instances: ", total_instances_FAC)
    print("consistency matrix: \n", S_FAC.shape)
    print("non-zero units in consistency matrix: \n", np.count_nonzero(S_FAC))

    KF_All_COCO_info = {}
    KF_All_COCO_info['co_occurrence_matrix'] = co_occurrence_matrix_FAC
    KF_All_COCO_info['label_occurrences'] = label_occurrences_FAC
    KF_All_COCO_info['total_instances'] = [total_instances_FAC]
    KF_All_COCO_info['S'] = S_FAC

    # KF-500-COCO (F5C):
    # 250 images from COCO trainloader and 250 images from VOC trainloader (adapted to COCO).

    num_classes = 91

    co_occurrence_matrix_F5C, label_occurrences_F5C, total_instances_F5C = \
        convert_voc_to_coco(co_occurrence_matrix_VO5_int, co_occurrence_matrix_CO5_int, label_occurrences_VO5_int,
                            label_occurrences_CO5_int, total_instances_VO5_int, total_instances_CO5_int)

    S_F5C = consistency_matrix(num_classes, label_occurrences_F5C, total_instances_F5C, co_occurrence_matrix_F5C)
    S_F5C = convert_91_to_80(S_F5C)  # if you want the full 91 concepts, comment this line

    print("\nKF-500-COCO \n")
    print("co_occurrence_matrix: \n", co_occurrence_matrix_F5C.astype(int).shape)
    print("non-zero units in co_occurrence_matrix: \n", np.count_nonzero(co_occurrence_matrix_F5C.astype(int)))
    print("label_occurrences: ", label_occurrences_F5C)
    print("label_occurrences: ", np.sum(label_occurrences_F5C))
    print("total_instances: ", total_instances_F5C)
    print("consistency matrix: \n", S_F5C.shape)
    print("non-zero units in consistency matrix: \n", np.count_nonzero(S_F5C))

    KF_500_COCO_info = {}
    KF_500_COCO_info['co_occurrence_matrix'] = co_occurrence_matrix_F5C
    KF_500_COCO_info['label_occurrences'] = label_occurrences_F5C
    KF_500_COCO_info['total_instances'] = [total_instances_F5C]
    KF_500_COCO_info['S'] = S_F5C

    # Store all dictionaries in another dictionary.
    info = {}
    info['KF_All_VOC_info'] = KF_All_VOC_info
    info['KF_500_VOC_info'] = KF_500_VOC_info
    info['KF_All_COCO_info'] = KF_All_COCO_info
    info['KF_500_COCO_info'] = KF_500_COCO_info

    # Save all the info to a JSON file
    with open(os.path.join(path_project, 'Semantic Consistency/Stored matrices/CM_freq_info.json'), 'w') as j:
        json.dump(info, j, cls=NumpyEncoder)

