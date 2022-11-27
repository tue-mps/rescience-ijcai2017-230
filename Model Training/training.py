import torchvision
from Datasets.datasets import VocDataset, CocoDetection
from Utils.plotting import plot_annotated_data_samples
from Utils.dataloading import *
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.backbone_utils import BackboneWithFPN, resnet_fpn_backbone

"""
Running this file will train the models as described in the paper. Select settype to choose between VOC and COCO
dataset. The model checkpoint will be stored as checkpoint.pth. To use a newly trained model for testing the results,
the file should be renamed to a logical name, and this should be accounted for in the testing file. 
"""

modeltype = 'resnet50' # 'resnet50', 'resnet18' or 'vgg16'
settype = 'coco'  # 'voc' or 'coco'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':

    batch_size = 2
    batch_size_test = 1
    workers = 2

    path_project = os.path.abspath(os.path.join(__file__, "../.."))
    print("path_project: ", path_project)

    # PASCAL VOC DATASET
    voc_path = os.path.join(path_project, "Data raw/VOC2007")
    output_folder = os.path.join(path_project, "Datasets")
    create_data_lists(voc07_path=voc_path, output_folder=output_folder)

    voc_train_dataset = VocDataset(output_folder, split='train')
    voc_train_dataloader = torch.utils.data.DataLoader(
        voc_train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=voc_train_dataset.collate_fn,
        num_workers=workers,
        pin_memory=True
        )

    voc_validation_dataset = VocDataset(output_folder, split='validation')
    voc_validation_dataloader = torch.utils.data.DataLoader(
        voc_validation_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=voc_validation_dataset.collate_fn,
        num_workers=workers,
        pin_memory=True
        )

    voc_test_dataset = VocDataset(output_folder, split='test')
    voc_test_dataloader = torch.utils.data.DataLoader(
        voc_test_dataset,
        batch_size=batch_size_test,
        shuffle=True,
        collate_fn=voc_test_dataset.collate_fn,
        num_workers=workers,
        pin_memory=True
        )

    # examples = enumerate(voc_train_dataloader)
    # i, (images, targets) = next(examples)
    # plot_annotated_data_samples(images, targets, "Annotated sample images from the VOC training dataset", 'voc')

    # COCO DATASET

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

    generator = torch.Generator().initial_seed()    # reset generator to random instead of seed 42

    coco_train_dataloader = torch.utils.data.DataLoader(
        coco_trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=True,
        collate_fn=CocoDetection.collate_fn
        )

    coco_validation_dataloader = torch.utils.data.DataLoader(
        coco_minival_1k,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=True,
        collate_fn=CocoDetection.collate_fn
        )

    coco_test_dataloader = torch.utils.data.DataLoader(
        coco_minival_4k,
        batch_size=batch_size_test,
        shuffle=True,
        num_workers=workers,
        pin_memory=True,
        collate_fn=CocoDetection.collate_fn
        )

    # examples = enumerate(coco_train_dataloader)
    # l, (images, targets) = next(examples)
    # plot_annotated_data_samples(images, targets, "Annotated sample images from the COCO train dataset", 'coco')

    print("\nDATASET LENGTHS: ")
    print("PASCAL training set: \t", voc_train_dataset.__len__(), " Images")
    print("PASCAL validation set: \t", voc_validation_dataset.__len__(), " Images")
    print("PASCAL test set: \t\t", voc_test_dataset.__len__(), " Images")
    print("COCO training set: \t\t", coco_trainset.__len__(), " Images")
    print("COCO validation set: \t", coco_minival_1k.__len__(), " Images")
    print("COCO test set: \t\t\t", coco_minival_4k.__len__(), " Images")


def train_epoch(train_loader, model, optimizer, epoch):
    """
    Iterate over the batches in the training set data loader, perform a forward and backward pass.
    :param train_loader: (pytorch) data_loader containing the training data
    :param model: the model. make sure it is in training mode
    :param optimizer: the optimizer
    :param epoch: counter that keeps track on which epoch we're on (for printing purposes)
    Source: https://blog.francium.tech/object-detection-with-faster-rcnn-bc2e4295bf49 (for the loss)
    Source: https://github.com/pytorch/vision/blob/master/references/detection/engine.py (for the loss)
    """
    file_path = os.path.join(path_project, "Model Training/Trained models/checkpoint.pth")
    losses_list = []
    model.train()
    for i, (images, targets) in enumerate(train_loader):

        # Move to default device.
        images = [im.to(device) for im in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # Some (1021) images in COCO contain no objects at all.  These are filtered out in the data loader.
        # In the rare occasion that both items from the batch contain such an empty image, the batch is
        # ignored altogether, as the model doesn't accept empty batch.
        if len(targets) == 0:
            continue

        # Forward prop.
        output = model(images, targets)

        # losses = sum(loss for loss in output[0].values())
        losses = sum(loss for loss in output.values())

        # Backward prop.
        optimizer.zero_grad()
        losses.backward()

        # Update model.
        optimizer.step()

        # Store losses, in case we would want to plot them, only used for printing now.
        # losses_list.append(losses.item())

        # Print status after every 100 iterations.
        if i % 100 == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss}\t'.format(epoch, i, len(train_loader), loss=losses))

    # Save the model after every epoch (the checkpoint file gets updated each epoch).
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': losses_list,
        }, file_path)
    print("\nModel saved.")


#### TRAINING ####

if settype == 'voc':
    epochs = 48
    epochs_dcy = 40
    num_classes = 21
    train_loader = voc_train_dataloader
elif settype == 'coco':
    epochs = 8
    epochs_dcy = 6
    num_classes = 92
    train_loader = coco_train_dataloader
else:
    print("selected wrong settype, please select 'voc' or 'coco' ")


lr = 1e-3
lr_dcy = 1e-4
momentum = 0.9
weight_decay = 5e-4

if modeltype == 'resnet50':
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
                       box_roi_pool=roi_pooler)
elif modeltype == 'resnet18':
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
                       box_roi_pool=roi_pooler)
elif modeltype == 'vgg16':
    backbone_vgg = torchvision.models.vgg16(pretrained=True).features
    out_channels = 512
    in_channels_list = [128, 256, 512, 512]
    return_layers = {'9': '0', '16': '1', '23': '2', '30': '3'}
    backbone = BackboneWithFPN(backbone_vgg, return_layers, in_channels_list, out_channels)
    backbone.out_channels = 512

    # anchor_generator = AnchorGenerator(sizes=((128,), (256,), (512,)),
    #                                    aspect_ratios=((0.5, 1.0, 2.0),(0.5, 1.0, 2.0),(0.5, 1.0, 2.0)))


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
                       box_roi_pool=roi_pooler)

else:
    print("select a valid model type")


optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
model.to(device)

model.train()

for e in range(epochs):
    print('training epoch number: ', e + 1)

    if e >= epochs_dcy:
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=lr_dcy,
                                    momentum=momentum,
                                    weight_decay=weight_decay
                                    )

    train_epoch(train_loader, model, optimizer, e)


