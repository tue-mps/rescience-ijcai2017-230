
Code for replicating the paper "Object Detection Meets Knowledge Graphs".

# Description

`datasets.py` contains the dataset classes which are made such that both VOC and COCO datasets are processed in the same way.

`training.py` is used to train the Faster R-CNN model for both the VOC and COCO datasets.

`frequency_based.py` is used to extract the semantic consistency by using the frequency based method. The resulting matrices for VOC and COCO are stored in a JSON file so that they can be re-used.

`knowledge_graph_prep` is used to load in the data from the knowledge graph assertion lists and filters them to the english subgraph and filters out negative relations. The resulting graph is converted to integers and both are stored in a new CSV file.

`knowledge_graph_based.py` is used to extract the semantic consistency from the filtered knowledge graph. The resulting matrices for VOC and COCO are stored in a JSON file so that they can be re-used.

`results_voc` and `results_coco` are used to output the mAP and recall (per class and averaged) for the configurations (including matrix type) which can be changed at the top of the files.

`dataloading`, `metrics`, `plotting` and `testing` in Utils contain several functions used to support the main files. 

More detailed descriptions are included in the top of each of the files themselves.


# Data-preparation
For this application, we use the PASCAL VOC 2007 dataset and the COCO2014 dataset (for object detection) and the ConceptNet 5.7.0 and 5.5.0 knowledge graphs.

## The easy way

The easiest way to gather the datasets is to download it from #adddataurlhere.

after cloning the repository, unzip the 'raw data' file in your project folder such that your structure looks as follows:

add the trained model (.pth files) into the Trained Models folder also as follows:

* Project
  * **Data raw**
    * COCO2014
      *	Annotations
      *	Image_info_test2014
      *	Test2014
      *	Train2014
      *	Val2014
    * ConceptNet
      *	Assertions55
      *	Assertions57
    * VOC2007
      *	Annotations
      *	ImageSets
      *	JPEGImages
      *	SegmentationClass
      *	SegmentationObject
  *	Datasets
  *	Model Training
    *	Trained Models
         * **coco-FRCNN-8e.pth**
         * **voc-FRCNN-48e.pth**
  *	Results
  *	Semantic Consistency
  *	Utils

## The difficult way

### VOC:

The VOC datasets can be downloaded at

[http://host.robots.ox.ac.uk/pascal/VOC/voc2007/index.html](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/index.html)

under the header ‘development kit’:

Download the training/validation data (450MB tar file)

and under the header ‘test data’:

Download the annotated test data (430MB tar file)

After unpacking the files, it is important to copy the files within Annotations, ImageSets and JPEGImages into one ‘superfolder’, such that there is only 1 folder containing all information for training, validation and testing. So in the Annotations folder there will be all xml files, in the JPEGImages folder there will be all images. In the ImageSets folder, in the Layout folder there will be all (test, train, trainval and val) text files. In the ImageSets folder, in the Main folder will also be all text files of the train test and validation sets. And the same goes for the ImageSets, Segmenation folder.

### COCO:
The COCO datasets can be downloaded at

[https://cocodataset.org/#download](https://cocodataset.org/#download)

download the following folders:

- 2014 Train images [83K/13GB]
- 2014 Val images [41K/6GB]
- 2014 Test images [41K/6GB]
- 2014 Train/Val annotations [241MB]
- 2014 Testing Image info [1MB]

Unpack these folders and add them together in a COCO2014 folder.

### ConceptNet

The assertion lists (knowledge graphs) can be downloaded at

[https://github.com/commonsense/conceptnet5/wiki/Downloads](https://github.com/commonsense/conceptnet5/wiki/Downloads)

for this approach versions 5.7.0 and 5.5.0 are used.

### structure

make sure the raw data folder is structured correctly:

* Project
  * **Data raw**
    * COCO2014
      *	Annotations
      *	Image_info_test2014
      *	Test2014
      *	Train2014
      *	Val2014
    * ConceptNet
      *	Assertions55
      *	Assertions57
    * VOC2007
      *	Annotations
      *	ImageSets
      *	JPEGImages
      *	SegmentationClass
      *	SegmentationObject
  *	Datasets
  *	Model Training
    *	Trained Models
         * **coco-FRCNN-8e.pth**
         * **voc-FRCNN-48e.pth**
  *	Results
  *	Semantic Consistency
  *	Utils


# Usage

Clone the repository using `git clone repositorylink`

Create a correct environment (update this)

Make sure the raw data is added correctly

The easiest is to download the trained models, otherwise `training.py` in the Model Training folders should be run first.

To obtain the semantic consistency matrix for the frequency based method, `frequency_based.py` in the semantic consistency folder should be run. This will store the matrix information in the folder. For the knowledge graph based method, first `knowledge_graph_prep.py` should be run. this will filter the raw data graph. Next `knowledge_graph_based.py` will produce and store the matrix. Since the matrices are stored, these have only to be run once.

To recreate the results from the paper, you can run either or both `results_voc` and `results_coco` in the results folder. In the top of the file you can change the configurations to select the correct matrix method to use and then this will output the mAP and recall.

# Summary of Reproducibility

### Scope of Reproducibility

'Object Detection meets Knowledge Graphs' by Fang et al. describes a framework which integrates external knowledge such as knowledge graphs into object detection. They apply two different approaches to quantify background knowledge as semantic consistency. An existing object detection algorithm is re-optimized with this knowledge to get updated knowledge-aware detections. The authors claim that this framework can be applied to any existing object detection algorithm and that this approach can increase recall, while maintaining mean Average Precision (mAP). In this report, the framework is implemented and the experiments are conducted as described in the paper, such that the claims can be validated.

### Methodology

The authors describe a framework where a frequency based approach and a knowledge graph based approach are used to determine semantic consistency. A knowledge-aware re-optimization function updates the detections of an existing object detection algorithm. Both the baseline and its re-optimized correlates are evaluated on two benchmark datasets, namely PASCAL VOC 2007 and MS COCO 2014. The replication of the experiments was completed using the information in the paper and clarifications of the author, as no source code was available. The framework was implemented in PyTorch and evaluated on the same benchmark datasets.

### Results

We were able to successfully implement the framework from scratch and replicate the experiments supporting the claim that this approach can be used to re-optimize an existing object detection algorithm. The frequency based approach to extract the semantic consistency showed an increase in recall, while maintaining mAP, confirming the second claim of the authors. The knowledge graph based approach, shows a negative (decrease in mAP) result with respect to the baseline. This could be due to the use of a different object detection model. Therefore we concluded that the benefits of knowledge-aware re-optimization are model-specific and cannot be used blindly for every object detector.

### What was easy

The methodology was well described and easy to understand conceptually.

### What was difficult

There was no source code available, which made it difficult to understand some technicalities of the implementation. The authors failed to mention a number of crucial details and assumptions of this implementation in the paper, which are essential for reproducing the methodology without making fundamental assumptions.

### Contact with the authors

We contacted the authors to elaborate on missing details, however no contact was found with the contact-information on the paper. Fortunately, a different email address of Yuan Fang was found online, to which he did respond fast and with clear explanations, for which we would like to express our gratitude.
