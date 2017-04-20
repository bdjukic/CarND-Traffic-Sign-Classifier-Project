# Recognizing traffic signs with deep neural network

I have recently enrolled into [Udacity's Self-driving nano degree program](https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013) and it has been a roller-coaster ride since the first lessons. If you're curios about the content of the first term in the program, you can get more details [here](https://medium.com/self-driving-cars/term-1-in-depth-on-udacitys-self-driving-car-curriculum-ffcf46af0c08#.ssjww0l3e).

Second project in the first term is building [deep neural network](https://en.wikipedia.org/wiki/Deep_learning) (in this particular case [Convolutional neural network](https://en.wikipedia.org/wiki/Convolutional_neural_network)) from scratch which is capable of recognizing traffic signs.

The deep neural network is supposed to be trained with existing real world data from [German roads](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset).

Once the training is done, we are supposed to throw at it few traffic signs from Germany that we can find just by googling for them. The deep neural network should gives us back correct predictions for the traffic signs it has not seen before.

Let's get a bit deeper (pun intended) into the steps how is this supposed to be done.

### Dataset Summary & Exploration

The traffic signs data set you get from Udacity is fairly large (at least by my standards since I have not had a chance to work with similar problems before). Here's a quick summary of the data which will be used to train deep neural network:

1. Training set size: **34799**
2. Validation set size: **4410**
3. Testing set size: **12630**
4. Image data shape: **32x32x3**
5. Number of traffic sign classes: **43**

This is a sample from the data set:

![alt tag](https://github.com/bdjukic/CarND-Traffic-Sign-Classifier-Project/raw/master/md_images/1.png)

When looking at the distribution of the training set, we can notice that the data is not distributed ideally:
Distribution of traffic signs in the training set. X axis represents traffic sign class. Y axis represents the total number of samples.

![alt tag](https://github.com/bdjukic/CarND-Traffic-Sign-Classifier-Project/raw/master/md_images/2.png)

This effectively means that our deep neural network will be better at recognizing certain traffic sign types than others. In a real world project we would aim for a better distribution of data samples which would result in better predictions.

### Dataset Preprocessing

Another important aspect of the successful training and prediction is dataset preprocessing. For this particular exercise, I've decided to on two techniques:

1. Applying grayscale
2. Equalizing the histogram of a grayscale image (normalizing brightness and increasing the contrast of the image)

Both of these methods can be found in OpenCV library.

### Model Architecture

As I've already mentioned, I will be using Convolutional Neural Network (CNN) for our model and training. The concrete CNN model which will be used is [LeNet](http://yann.lecun.com/exdb/lenet/).

![alt tag](https://github.com/bdjukic/CarND-Traffic-Sign-Classifier-Project/raw/master/md_images/3.png)

This industry proven model takes 32x32x1 image as an input. It then takes that input through first convolutional layer (C1, 6 feature maps, 28x28) which is followed by pooling function (S2, 6 feature maps, 14x14). Output from this step is taken to another convolutional layer (C3, 16 feature maps, 10x10) which is also followed with another pooling function (S4, 16 feature maps, 5x5). Finally, we end up with three fully connected layers (which includes our output layer as well).

There definitely other options I could have picked as an architecture for the model (check out [Pierre Sermanet and Yann LeCun](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf) paper for example), but LeNet gave me quite a decent accuracy right out of the box.

### Model training

The model was able to get to validation set accuracy of **0.952**after 45 epochs and with batch size of **128**. Learning rate of the model was set to **0.01**.

Test set came back with the accuracy of **0.922**.

I've noticed how validation data set size impacted the actual accuracy of the model. The more data I fed into the model, the better starting point of the accuracy was and also the model was able to get better final accuracy numbers.

### Test Model on New Images

The final piece of the puzzle was to test the model against the sample of German traffic signs which can found on the Internet. I've picked up five random traffic signs, preprocessed them the same way the original training data was preprocessed and let the model decide which traffic sign it's being shown:

![alt tag](https://github.com/bdjukic/CarND-Traffic-Sign-Classifier-Project/raw/master/md_images/4.png) 

By looking closer at our training data, we can notice the number of data points for the first test image (Speed limit (100km/h)) is quite low. This could have been fixed by doing some of the augmentation techniques on the data set which would increase the number of samples and in the end the accuracy of the model. Lack of good training data for this traffic sign will result in bad prediction (further down).

Second image had some background noise which might be challenging for the model as well.

The rest of the test images should not cause any problems during the classification.

The final accuracy rate was **80%** which I think it's really good result given that I have not done any data augmentation (rotation, translation, zoom, flips, and/or color perturbation, basically data generation). Data augmentation would improve the accuracy of the model. I'll keep this on my mind in any upcoming similar project.

### Model Certainty — Softmax Probabilities

As mentioned, the accuracy rate was 80%. Out of our 5 traffic signs, first example (Speed limit (100km/h)) was not recognized properly (it was marked as Speed limit (120km/h) with really high probability rate 9.99999881e-01). This would probably result in a speeding ticket for the self driving car on German road :).

All other traffic signs were detected properly with a really high probability rate.

### Wrapping up

This has been a great introduction to the deep neural networks on a very concrete problem.

The whole solution can be found in my forked [Github repo](https://github.com/bdjukic/CarND-Traffic-Sign-Classifier-Project).
