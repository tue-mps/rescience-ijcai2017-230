
Code for replicating the paper "Object Detection Meets Knowledge Graphs".

# Description

`datasets.py` contains the dataset classes which are made such that both VOC and COCO datasets are processed in the same way.

`training.py` is used to train the Faster R-CNN model for both the VOC and COCO datasets.

`frequency_based.py` is used to extract the semantic consistency by using the frequency based method. The resulting matrices for VOC and COCO are stored in a JSON file so that they can be re-used.

`knowledge_graph_prep` is used to load in the data from the knowledge graph assertion lists and filters them to the english subgraph and filters out negative relations. The resulting graph is converted to integers and both are stored in a new CSV file.

`knowledge_graph_based.py` is used to extract the semantic consistency from the filtered knowledge graph. The resulting matrices for VOC and COCO are stored in a JSON file so that they can be re-used.

`results_voc` and `results_coco` are used to output the mAP and recall (per class and averaged) for the configurations (including matrix type) which can be changed at the top of the files.

`results_voc_multiple_runs` and `results_coco_multiple_runs` can be run to compute all results used in the paper in one single run

`dataloading`, `metrics`, `plotting` and `testing` in Utils contain several functions used to support the main files. 

More detailed descriptions are included in the top of each of the files themselves.

# Installation

We first need to clone this repository, and the random walk with restart repository (which can be found at [https://github.com/jinhongjung/pyrwr](https://github.com/jinhongjung/pyrwr)):

`git clone https://github.com/tue-mps/rescience-ijcai2017-230.git`

`cd "rescience-ijcai2017-230"`

`git clone https://github.com/jinhongjung/pyrwr.git`

Next we need to create the correct Environment from our requirements file and the RWR requirements. (if using Anaconda prompt):

(if you want to create a new environment for this project, I suggest using python 3.7):
`conda create -n odmkg_env python=3.7`

`conda activate odmkg_env`

`pip3 install -r requirements.txt`

`cd pyrwr`

`pip3 install -r requirements.txt`

`python setup.py install`

`cd ..`


# Data-preparation
For this application, we use the PASCAL VOC 2007 dataset and the COCO2014 dataset (for object detection) and the ConceptNet 5.7.0 and 5.5.0 knowledge graphs.

## The easy way

The easiest way to gather the datasets is to download it from [https://zenodo.org/record/5554349#.YWAWFppByiN](https://zenodo.org/record/5554349#.YWAWFppByiN). There are 3 files, 2 .pth files that contain the trained models and 1 large zip file (28.7 GB) which contains the raw data. Download all 3 of them.

after cloning the repository, unzip the 'raw data' file (without the extra folder) in your project folder such that your structure looks as follows:

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
    *	Trained models
         * **coco-FRCNN-resnet50.pth**
         * **coco-FRCNN-vgg16.pth**
         * **voc-FRCNN-resnet50.pth**
         * **voc-FRCNN-resnet18.pth**
         * **voc-FRCNN-vgg16.pth**
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
    *	Trained models
         * **coco-FRCNN-resnet50.pth**
         * **coco-FRCNN-vgg16.pth**
         * **voc-FRCNN-resnet50.pth**
         * **voc-FRCNN-resnet18.pth**
         * **voc-FRCNN-vgg16.pth**
  *	Results
  *	Semantic Consistency
  *	Utils


# Usage

Once the repositories are installed, the environment is correct and the raw data and trained models are put in the correct locations, the code can be used.

**It is recommended to use GPU (cuda), as cpu only has not been developed/tested for.**

To recreate the results from the paper you can just run the initial configuration which includes the ConceptNet consistency matrix.
You can run the script from the prompt as follows:

`python -m Results.results_voc_multiple_runs` or
`python -m Results.results_coco_multiple_runs`

Or if you want to select specific configurations, you can use:

`python -m Results.results_voc` or
`python -m Results.results_coco`

If you want to train a model for yourself, `training.py` in the Model Training folders should be run first.

To (re-)obtain the semantic consistency matrix for the frequency based method, `frequency_based.py` in the semantic consistency folder should be run. This will store the matrix information in the folder. For the knowledge graph based method, first `knowledge_graph_prep.py` should be run. this will filter the raw data graph. Next `knowledge_graph_based.py` will produce and store the matrix. Since the matrices are stored, these have only to be run once. However, they are included in the files, so if you just want to reproduce the results from the paper, you don't have to run these.

To recreate the results from the paper, you can run either or both `results_voc` and `results_coco` in the Results folder. In the top of the file you can change the configurations to select the correct matrix method to use and then this will output the mAP and recall.


## configurations

results_voc and results_coco start with a couple of configuration lines. Initially for the VOC set they are as follows:

```model_type = 'resnet50'       # choose between 'resnet50', 'resnet18' or 'vgg16'
matrix_type = 'KG-CNet-55-VOC'   # choose between 'KF-All-VOC', 'KF-500-VOC', 'KG-CNet-57-VOC' or 'KG-CNet-55-VOC'
detections_per_image = 500       # the maximum number of detected objects per image
num_iterations = 10              # number of iterations to calculate p_hat
box_score_threshold = 1e-5       # minimum score for a bounding box to be kept as detection (default = 0.05)
bk = 5                           # number of neighbouring bounding boxes to consider for p_hat
lk = 5                           # number of largest semantic consistent classes to consider for p_hat
epsilon = 0.9                    # trade-off parameter for traditional detections and knowledge aware detections
topk = 100                       # maximum number of detections to be considered for metrics (recall@k / mAP@k)
```
They can be changed, and by running the script again, different results will be computed.

# Summary of Reproducibility

### Scope of Reproducibility

'Object Detection meets Knowledge Graphs' by Fang et al. describes a framework which integrates external knowledge such as knowledge graphs into object detection. They apply two different approaches to quantify background knowledge as semantic consistency. An existing object detection algorithm is re-optimized with this knowledge to get updated knowledge-aware detections. The authors claim that this framework can be applied to any existing object detection algorithm and that this approach can increase recall, while maintaining mean Average Precision (mAP). In this report, the framework is implemented and the experiments are conducted as described in the paper, such that the claims can be validated.

### Methodology

The authors describe a framework where a frequency based approach and a knowledge graph based approach are used to determine semantic consistency. A knowledge-aware re-optimization function updates the detections of an existing object detection algorithm. Both the baseline and its re-optimized correlates are evaluated on two benchmark datasets, namely PASCAL VOC 2007 and MS COCO 2014. The replication of the experiments was completed using the information in the paper and clarifications of the author, as no source code was available. The framework was implemented in PyTorch and evaluated on the same benchmark datasets.

### Results

We were able to successfully implement the framework from scratch as described.
We have bench‐marked the developed framework on two datasets and replicated the
results of all matrices as described. The claim of the authors can not be
confirmed for either of the described approaches. The results either showed an increase
of recall at the cost of a decrease in mAP, or a maintained mAP, without an improvement
in recall. Three different backbone models show similar behavior after re‐optimization,
concluding that the knowledge‐aware re‐optimization does not benefit object detection
algorithms.

### What was easy

The methodology was well described and easy to understand conceptually.

### What was difficult

There was no source code available, which made it difficult to understand some technicalities of the implementation. The authors failed to mention a number of crucial details and assumptions of this implementation in the paper, which are essential for reproducing the methodology without making fundamental assumptions.

### Contact with the authors

We contacted the authors to elaborate on missing details, however no contact was found with the contact-information on the paper. Fortunately, a different email address of Yuan Fang was found online, to which he did respond fast and with clear explanations, for which we would like to express our gratitude.
