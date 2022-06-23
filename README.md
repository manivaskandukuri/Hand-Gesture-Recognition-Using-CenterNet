# Hand-Gesture-Recognition-using-CenterNet

## About the project:
Hand Gestures is one of the interesting areas of Computer Vision. It has numerous real world applications like communication with the specially abled people (deaf and mute), Human-Computer-Interactio(HCI), robotics, etc. Hand gesture recognition involves detection of the hand in the image or video, followed by recognizing meaning conveyed by the hand pose. In this project, we used CenterNet based object detection method to localize the hand portion and recognize the hand gesture.


## About CenterNet:
CenterNet is a CNN based on keypoint based approach. It is a single state detector and doesn't use any anchor boxes for detection. This make CenterNet state of art performance over the existing single stage approaches like Yolo, Single Shot detector (SSD), RetinaNet etc. For localizing objects, CenterNet predicts the Center point of the object and the width and height of the bounding box surrounding the object.

![image](https://user-images.githubusercontent.com/83395271/167784242-60fcee2b-1937-42c6-bd45-363382a7163d.png)
![image](https://user-images.githubusercontent.com/83395271/167784287-4abecffc-b996-4b83-90d1-757279ac8135.png)


The Network architecture of CenterNet consists of Encoder and Decoder modules followed by 3 parallel branches. Correspondingly, 3 feature maps are obtained at the output. The first heatmap is used to predict the Center keypoint, the second one is used to predict the width and height of bounding box and the third one is used to predict the width and height offsets to better localize the objects. Encoder is made of Resnet-101 which downsamples the input features and decoder consists of transpose convolutions for upsampling. All the three output feature maps are decoded to regress and classify the objects in the image.

![image](https://user-images.githubusercontent.com/83395271/167829434-0cf658e8-a566-4655-a7ea-e1ca13189a38.png)



## About the Dataset:
OUHANDS dataset is used for this project. It consists of 10 classes (which are labelled as A, B, C, D, E, F, H, I, J, K). This dataset consists of 2000 train images, 500 validation images and 500 test images. Each image is of 480x640 resolution.


## Performance metric:
Mean F1-score is used as the performance metric for evaluation.


## Training:
In this project, CenterNet is trained for 70 epochs at a learning rate of 1e-4 with batch size of 2. On the validation data, we obtained a Mean F1-score of 0.80.


## Results on test images:
On the test data, we obtained a Mean F1-score of 82.37%. Following are some of the results.

!<img src="https://user-images.githubusercontent.com/83395271/167828331-9948b2cc-22be-48bd-99ad-b78d2dd33fe6.png" width="320" height="240">
!<img src="https://user-images.githubusercontent.com/83395271/167828375-cabd1ff4-4a13-4e4f-b06f-c730943d6e4a.png" width="320" height="240">
!<img src="https://user-images.githubusercontent.com/83395271/167828418-c4bb9935-7866-4bbd-b83b-47e8c5220dab.png" width="320" height="240">
!<img src="https://user-images.githubusercontent.com/83395271/167828453-33c06519-8982-47dd-9e90-fa0f7d792c30.png" width="320" height="240">
!<img src="https://user-images.githubusercontent.com/83395271/167828481-4c965845-fb60-4a33-9887-663d30ebaebe.png" width="320" height="240">
!<img src="https://user-images.githubusercontent.com/83395271/167828500-c16b6c69-e43b-4cee-b6a2-53be9cb09021.png" width="320" height="240">
!<img src="https://user-images.githubusercontent.com/83395271/167828599-25ea7db7-e9d1-4066-bec4-4c8eefe5a8de.png" width="320" height="240">
!<img src="https://user-images.githubusercontent.com/83395271/167828662-12d04b31-8007-4231-b08a-d19898d4dbc2.png" width="320" height="240">
!<img src="https://user-images.githubusercontent.com/83395271/167828707-eb8a073b-7951-467f-add1-8f36b4841005.png" width="320" height="240">
!<img src="https://user-images.githubusercontent.com/83395271/167828752-eddbd414-7981-4c4a-98c5-556980c7c392.png" width="320" height="240">




## Dual Attention Network:
In order to improve the performance of CenterNet, Dual Attention Network (DA-Net) is added between Encoder and Decoder modules. DA-Net consists of Position and Channel Attention modules connected in parallel. The Position Attention module captures spatial relation between the pixels whereas the Channel Attention module captures the relation between the channels.

By adding DA-Net, the Mean F1-score on the test data improved from 82.37% to 84.40%.
