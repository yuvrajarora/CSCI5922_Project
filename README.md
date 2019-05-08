# CSCI5922_Project
Neural Networks and Deep Learning Course Project - CSCI 5922

### Project Title: Deep Learning for Cell Segmentation in Time-lapse Microscopy
##### Team Members: Shemal Lalaji, Swathi Upadhyaya, Yuvraj Arora 

#### Objective: 

The goal of our project was to build Neural Network Models to segment moving cells in a real 2D time-lapse microscopy videos of cells along with computer generated 2D video sequences simulating whole cells moving in realistic environments. The evaluation method used for the models is segmentation accuracy.

#### Dataset:

From the vast dataset available in the Cell Tracking Challenge, we chose the Fluo-N2DH-SIM+ dataset. This dataset consists of simulated nuclei of HL60 cells stained with Hoescht. The video is recorded over 29 minutes to study the cell dynamics of various cells. The benchmark for the segmentation evaluation methodology is 80.7 % for this dataset.


#### Model Architecture:

##### 1. U-Net:

![Fig.1 U-Net Model](https://github.com/yuvrajarora/CSCI5922_Project/blob/master/Assets/Unet-Model.png)

U-Net is built on Fully Convolutional Network. It is modified and extended in a way such that it works with very few training images and yields more precise segmentation. The network aims to classify each pixel. This network takes a raw input image and outputs a segmentation mask. A class label is assigned to each pixel. This architecture consists of two main parts: Contraction Path and Expansion Path. We end up creating multiple feature maps and the network is able to learn complex patterns with these feature maps. The Contraction path helps to localize high resolution features and the Expansion Path increases the resolution of the output by upsampling and combining features from the contraction path.


##### 2. U-Net with Convolution LSTM Block - 

![Fig.2 U-Net C-LSTM Model](https://github.com/yuvrajarora/CSCI5922_Project/blob/master/Assets/Unet-CLSTM.png)

This network incorporates Convolution LSTM (C-LSTM) Block into the U-Net architecture. This network allows considering past cell appearances at multiple scales by holding their compact representations in the C-LSTM memory units. Applying the CLSTM on multiple scales is essential for cell microscopy sequences since the frame to frame differences might be at different scales, depending on cells' dynamics. The network is fully convolutional and, therefore, can be used with any image size during both training and testing.

##### 3. VGG Net with Skip - 

![Fig.3 U-Net C-LSTM Model](https://github.com/yuvrajarora/CSCI5922_Project/blob/master/Assets/VGG-Net.png)

VGG Net shows a improvement on the classification accuracy and generalization capability on our model. Along with that using skip and Relu  allows us to improve the performance of the models and segment cells properly to view and  refine the spatial precision of the output.

#### Results:

##### Original Image

![Fig.4 Original Image](https://github.com/yuvrajarora/CSCI5922_Project/blob/master/Assets/Original_Img.png)

##### Masks generated:

###### U-Net mask
![](https://github.com/yuvrajarora/CSCI5922_Project/blob/master/Assets/U-Net_Mask.png)
###### U-Net CSLTM mask
![](https://github.com/yuvrajarora/CSCI5922_Project/blob/master/Assets/ConvLSTM_Mask.png)
###### VGG Net mask
![](https://github.com/yuvrajarora/CSCI5922_Project/blob/master/Assets/VGG-Net_Mask.png)

