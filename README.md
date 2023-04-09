# Training Mask R-CNN on custom dataset using pytorch
This repository contains code for training a Mask R-CNN model on a custom dataset using PyTorch. Mask R-CNN is a powerful deep learning model that can be used for both object detection and instance segmentation. This project serves as a practical demonstration of how to train a Mask R-CNN model on a custom dataset using PyTorch, with a focus on building a person classifier.

One way to save time and resources when building a Mask RCNN model is to use a pre-trained model. In this repository, we will use a pre-trained model with a ResNet-50-FPN backbone provided by torchvision. This model has already undergone extensive training on the COCO dataset, allowing it to learn generalizable features from a large dataset. Fine-tuning this pre-trained model to suit a specific task can significantly reduce the amount of time and resources required compared to training a new model from scratch. In this case, We only need to modify the box predictor layer and the mask predictor layer to fit the custom dataset.

![GITHUB](https://duck00036-public-images.s3.ap-northeast-1.amazonaws.com/output.jpg)

# Installation
To run this project, you will need to install the following packages:

* PyTorch
* torchvision
* NumPy
* matplotlib
* Pillow
* OpenCV
* tqdm
* wget
* pycocotools
* pycocotools-windows (for windows user)
* requests

You can install these packages using pip:
```
pip install torch torchvision numpy matplotlib Pillow opencv-python tqdm wget pycocotools requests
```

# Dataset
Before you can start training the model, you will need to prepare your custom dataset. The dataset should be organized in the following directory structure:
```
dataset/
    images_train/
        data1.jpg
        data2.jpg
        data3.jpg
        ...
    annotations_train/
        data1.npy
        data2.npy
        data3.npy
        ...
```

Each image should have a corresponding NPY file with the same name, containing the annotations for that image. The NPY files should contain the following information for each object in the image:

* **label** -> **int** : object class
* **box** -> **np.array(shape=(4, ))** : bounding box coordinates (xmin, ymin, xmax, ymax)
* **mask** -> **np.array(shape=(width, height))** : segmentation mask

The final NPY file should be structured as follows:
```py
{
    'boxes': np.array([box_1, box_2, ...]), 
    'labels': np.array([label_1, label_2, ...]),  
    'masks': np.array([mask_1, mask_2, ...])
}
```
where each element of the boxes, labels, and masks arrays corresponds to a single object in the image, and the length of each array is the number of objects in the image.

In this repository, I have chosen to work with the person data from the COCO dataset. To utilize the person images and annotations from the COCO dataset, we need to download the entire dataset first. Once downloaded, we should extract the images and annotations for persons and organize them into the structure mentioned above. The specific steps for doing so can be found in the [data_prepare.ipynb](data_prepare.ipynb) notebook. However, if you are working with a different custom dataset, the specific steps may vary depending on the structure and format of your dataset.

# Usage
## Training
To train your own custom dataset, you should first clone this repository
```
git clone https://github.com/duck00036/Training-Mask-RCNN-on-custom-dataset-using-pytorch.git
```
Delete the .gitkeep file in **dataset/images_train** and **dataset/annotations_train**, then put your own images and annotations into these two folder.

Run the script:
```
python train.py
```
and the training will start with default parameters.

Or your can use [training_demo.ipynb](training_demo.ipynb) notebook to train your model step by step.

### default parameters

* **Class Number** : 1 + 1(background)
* **Batch Size** : 1
* **Optimizer** : SGD (momentum = 0.9, weight_decay = 0.0005)
* **Learning Rate** : 0.0001
* **lr scheduler** : No
* **Epoch Number** : 20
* **Saving Frequency** : every 5 epoch

Of course, you can tweak it as much as you want by editing the code to make the model fit your data better.

## Person classifier
To use the trained person classifier, you should first clone this repository
```
git clone https://github.com/duck00036/Training-Mask-RCNN-on-custom-dataset-using-pytorch.git
```
Download the trained model [here](https://duck00036-public-images.s3.ap-northeast-1.amazonaws.com/person_classifier.pt) in the repository, and put the images to be classified into the **input_image** folder

Run the script:
```
python eval.py
```
and the result will be saved in the **output_image** folder.

# About trained person classifier model


