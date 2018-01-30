# **Behavioral Cloning Writeup**

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[trainloss]: ./examples/loss.png "Model Training Loss"
[track1_issue1]: ./examples/track1_issue1.png "Track 1 issue 1"
[track1_issue2]: ./examples/track1_issue2.png "Track 1 issue 2"
[track2_issue]: ./examples/track2_issue.png "Track 2 issue"
[saliency]: ./examples/saliency.png "Saliency Track 2"
[attention]: ./examples/attention.png "Attention Track 2"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

The model is created in createModel function (model.py lines 114-157).

My model consists of a convolution neural network with 3x3 filter sizes and depths between 32 and 256. The model includes RELU layers to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer. Each Convolution2D layer (except the first) is preceded by a BatchNormalization layer, to account for covariate shift.

Before passed to convolution layers, input images are cropped from the top (sky) and bottom (car hood), to remove unnecessary image information, and reduce the network parameter space.

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting. I chose to use dropout layers only on top of fully-connected layers.

The model was trained and validated on different data sets to ensure that the model was not overfitting (function openTelemetryTrainValidFromSingleDataset, code line 108). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 187).

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

In order to process large amounts of training data, I chose to use python generators with Keras, instead of loading training/validation sets in memory.

Instead of writing separate generators for the training and validation phases, I parameterized the generator, so it uses a fixed batch size (passed as a command line parameter), yet the training specialization of the generator performs data augmentation live, as it loads data from disk, using the following rules:

1. For this simulation, the rule of the road tells to stay on track, such that not a single wheel goes outside of the road. Hence it is safe to assume there is no left/right-hand side driving, and we can train a symmetrical model, which will lean towards the road axis. For each input sample consisting of a pair (image, steering_angle), I augment the training set by a horizontally-flipped version of the image, and the negated steering_angle. This doubles the training set and allows to save time during manual driving.

2. Each training sample comes with side views from the camera, and the lecture notes conveniently suggest the angular distance between the cameras' focal axes (0.2 in parameter space). So I additionally triple the training set by using all three images instead of the front view, and subtracting/adding the angular delta to get the steering angles for side views.

After the generators were in place, the initial model architecture was chosen to connect all individual pixel inputs to the output neuron, thus presenting a fully-connected layer.

Without hesitation this architecture was complemented with all the intuitions discussed in the lecture notes, inspired by LeNet architecture:
- the input cropping layer
- 5 groups of
    - BatchNormalization (except the first) layer to compensate for covariate shift
    - Convolution2d layer with filters 3x3 and doubling depths from 32 to 256
    - MaxPooling2D layer to shrink the image size by the factor of 2 each step
- Followed by 2 groups of
    - BatchNormalization layer (covariate shift)
    - Dense layer
    - Dropout layer, useful to prevent overfitting
- The single output regression neuron, taking values +1 to steer right, -1 to steer left

This model architecture was successfully trained on both AWS and laptop with GPU. The latter became possible due to special attention to the training set and the way to curate it.

Curiously enough, training for 4 epochs was enough after the training set became representative of the task. Loss did not improve after more epochs.

The model is using MSE loss (due to the regression nature of the task), 20% validation set. Here is visualization of the function during training.

![][trainloss]

NB: This model worked well immediately, and later I noticed that it is quite large for the task it is solving. The model proposed in NVIDIA whitepaper ("End to End Learning for Self-Driving Cars") has much smaller number of convolutional filters, yet it is capable to perform in real car driving situations, which suggests it can also handle the provided simulator without issues. Instead of further optimizing for the network size, I chose to experiment with attention maps, discussed in the last section.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture was chosen as a series of groups with convolutions and fully-connected layers, as described in the previous section. Given this structure, it was easy to introduce 2 fully connected layers (as in most prominent yet minimalistic architectures), and the number of convolutional groups is dictated by the input image size, and roughly equals ceil(log(min_image_dims)/log(maxpoolstride)) in case of same padding, and a couple units less in case of valid padding.

| Layer              | Description |
|:------------------:|:---------------------------------------------:|
| Input              | 160x320x3 RGB image |
| Cropping2D         | 57 pixels from top, 25 - bottom, and 1 pixel left and right, output 78x318x3 |
| Lambda 1/127.5-0.5 | Input normalization layer |
| Convolution 3x3    | 1x1 stride, valid padding, merged RELU activation, output 76x316x16 |
| Max pooling        | 2x2 stride, output 38x158x16 |
| Batch norm         | Also, removed the addition of 'b' variables from every layer, following batch norm |
| Convolution 3x3    | 1x1 stride, valid padding, merged RELU activation, output 36x156x32 |
| Max pooling        | 2x2 stride, output 18x78x32 |
| Batch norm         | |
| Convolution 3x3    | 1x1 stride, valid padding, merged RELU activation, output 16x76x64 |
| Max pooling        | 2x2 stride, output 8x38x64 |
| Batch norm         | |
| Convolution 3x3    | 1x1 stride, valid padding, merged RELU activation, output 6x36x128 |
| Max pooling        | 2x2 stride, output 3x18x128 |
| Batch norm         | |
| Convolution 3x3    | 1x1 stride, valid padding, merged RELU activation, output 1x16x256 |
| Flatten            | output 4092 |
| Batch norm         | |
| Fully connected    | Input 4092, output 256 |
| Dropout            | With keep_prob=0.5 |
| Batch norm         | |
| Fully connected    | Input 256, output 32 |
| Dropout            | With keep_prob=0.5 |
| Batch norm         | |
| Fully connected    | Input 32, output 1 |

#### 3. Creation of the Training Set & Training Process

I chose an active learning approach to train the model, because it allows to incrementally collect data and track how the network performs, as it learns. It also helps to identify when an addition to the training set introduces a regression, which can be seen as a very early overfitting prevention.

The training process consisted of two tactics:
- keep driving perfectly in the center of the road (braking can be used to react to the road conditions quicker). This data goes into the training set unaltered
- driving with swings left and right. This data requires initial filtration - I played the sequence in the Preview app and removed the parts, where the car went away from the desired central position, and only kept the parts when it converges to the center.

After the initial training set collection and training, the car went flawlessly a few smooth curves, and the bridge, until it reached a junction on the track one, where the main road splits into the main road, which curves to the left, and the dirt road straight ahead.

![][track1_issue1]

Apparently, the lack of training data with such split caused the network to think it is ok to proceed without steering, resulting in car going off the main road. Including this place into the training set helped the network to adapt to this new situation.

The second time it went off road was the very next sharp bend to the right. Reason to that was the lack of sharp bends in the training set before.

![][track1_issue2]

Again, inclusion of this sharp bend helped to pass lap one without issues. Video file of the lap: track1.mp4

I proceeded with trying the network on track two. Unlike track one, here are the key differences:
- present lane marks, which can be ignored this time around
- the road bends sharp and changes uphill and downhill direction very often

Running the network, trained purely on the data from track one, on the second track, resulted in an instant failure due to the presence of lane marks. This was solved by collecting 3-4 turns on the track two. After retraining, the car was capable of following the lane line and fitting into sharp bends of the road without issues, while still being capable to perform on track one. The last time the network failed in the simulator was the place of track two, where two roads bordering with each other.

![][track2_issue]

This situation indicates that the network relies on a variety of image zones in making decisions when not to steer. This case made me check the approaches to visualization of network attention. Video file of the lap: track2.mp4

### Brief analysis of the trained model

The failures of the network during the active learning phase gave me an idea of what a network considers as important clues, and this understanding was often surprising. The search for the methods of network performance analysis led to the concept of saliency maps (where each pixel contribution to the output is evaluated per given sample), and attention maps, which are more coarse, yet provide higher-level information about which areas of the input image leaned the network's decision to steer left or right.

These concepts were conveniently implemented in `keras-vis` package, which I used to produce the following visualizations, and the video files (track2_saliency.mp4 and track2_attention.mp4)

Saliency maps appear a bit noisy as in the example below, and they don't seem to provide much high level clues, instead they highlight the road lanes and borders, which is already a good sign that the network learned to steer from this information, and not the bushes on the side of the road.

![][saliency]

Attention maps are generated from the middle convolution or pooling layers, and highlight coarse zones of the image, which contributed to increasing the output value, which corresponds to the decision to steer right. To visualize steering left, the framework needs to be told to use negative gradients. The video file track2_attention.mp4 and the following image visualize attention maps when network decides to steer left (top part of video) or right (bottom part of video).

![][attention]

End of report
