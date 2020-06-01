# 1 Identify Handwritten Digits by Manipulator

In this task we evaluate the performance of the model provided in **MNISTModel.mat**. Before we can feed images to the model we need to make sure that the images are in a compatible format and usable for the model. To ensure this we first convert every image to grayscale. Then we resize the smaller dimension of the image to 28px and scale the larger dimension accordingly. In the end the take the 28 x 28 px center crop of the image. 

By applying this preprocessing to every image we ensure that every matrix, fed to the model, has the desired size of 28 x 28.

In total we feed 800 self-hand-written numbers to the model. The numbers are written in different colors and different sizes. Most of the time the model misclassifies the digits. The class it assigns the most is the "1".

![Task1 Confusion Matrix](./task1_confusion_matrix.png)

Regarding the results shown in the confusion matrix above, we assume that the provided model is not capable for real-world uses-cases.



# 2 Transfer Learning in Ground Robot

Uses the google net model pretrained on [ImageNet](http://www.image-net.org) data [GoogLeNet](https://arxiv.org/abs/1409.4842)

# 3 Semantic Segmentation in Aerial Robot

