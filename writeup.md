# **Traffic Sign Recognition**

## Introduction

---

The steps of this traffic sign recognition project are:

* load, Explore, summarize and visualize the data set
* Design, train and test a model architecture starting from LeNet as a baseline
* Use the model to make predictions on new images
* Analyze the Softmax probabilities of the new images
* Summarize the results


[//]: # (Image References)
[categories_preview]: ./output/images_preview.jpg "images/categories preview"
[train_dist]: ./output/classes_distribution.png "train dataset distribution"
[training_session]: ./output/training_session.png "training session"

[image2]: ./examples/grayscale.jpg "Grayscaling"

[augmentation]: ./output/augmentation_generation/preview_0.jpg "Random Noise and slight rotation"
[augmentation_dist]: ./output/classes_distribution_with_augmented.png "Classes distribution with augmented"

[models_comparison]: ./output/models_comparison.png "models comparison"
[top_5_softmax_probs]: ./output/top_5_softmax_probs.png "top 5 probs"

[top_misclassified]: ./output/misclassified/preview_0.jpg "top misclassified"

[categories_preview_1]: ./output/previews/preview_0.jpg "categories preview. 1"
[categories_preview_2]: ./output/previews/preview_1.jpg "categories preview. 2"
[categories_preview_3]: ./output/previews/preview_2.jpg "categories preview. 3"
[categories_preview_4]: ./output/previews/preview_3.jpg "categories preview. 4"
[categories_preview_5]: ./output/previews/preview_4.jpg "categories preview. 5"


[activation_preview_13_first_cov2d]: ./output/activation/preview_0_cov2d_13.jpg "Activation preview class 13"
[activation_preview_17_first_cov2d]: ./output/activation/preview_0_cov2d_17.jpg "Activation preview class 17"
[activation_preview_13_second_cov2d]: ./output/activation/preview_1_cov2d_13.jpg "Activation preview class 13"
[activation_preview_17_second_cov2d]: ./output/activation/preview_1_cov2d_17.jpg "Activation preview class 17"

[selected_model_train]: ./output/selected_model_train.png "selected model (train arch.)"
[selected_model_pred]:  ./output/selected_model_pred.png "selected model (pred arch.)"

[ext_images]: ./output/ext_new/preview_0.jpg "Traffic Sign 1"


## Rubric Points

ref. [rubric points](https://review.udacity.com/#!/rubrics/481/view)

## Data Set Summary & Exploration

### 1. Summary statistics of the traffic signs data set:

* The size of the training set: 34799
* The size of the validation set: 4410
* The size of the test set: 12630
* The shape of a traffic sign image: (32, 32, 3)
* The number of unique classes/categories in the data set is 43


### 2. Images preview

Here is a preview showing 5 randomly chosen example for each of the 43 traffic sign classes.

![Images per category preview ][categories_preview_1]
![Images per category preview ][categories_preview_2]
![Images per category preview ][categories_preview_3]
![Images per category preview ][categories_preview_4]
![Images per category preview ][categories_preview_5]


### 3. Exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It shows the distribution of images over classes/categories.  We can observe that the classes are not equally represented within the dataset (skewed distribution): while the first 8 most represented classes have more than 1.5k examples per categories. The least represented classes ones have less than 250 examples (6+ time smaller size than the most represented ones).

![train data set distribution ][train_dist]

## Design and Test a Model Architecture



### 1. Image pre-processing

Since some of the color features might be useful for the recognition narrowing the categories down (especially the Red, blue, and yellow), combined with the colored nature of the images (in the dataset ). Starting with LeNet on the color channels sounded natural.

As preprocessing, an image resizing to 32x32 layer as well as a rescaling is added to the data.

While it is redundant for the initial/problem dataset already within (32x32) image size, the first layer is useful for predicting external images of different sizes, or including different images batches to the training set.

The second layer, namely rescaling approximates the normalization (the image data should be normalized so that the data has mean zero and equal variance).


### 2. Data augmentation

As the classes are not balanced  (skewed distribution of sample per class/category), the categories with few examples in the training set are re-sampled then augmented with random noise and small rotation. For each of the classes with a number of samples fewer than `category_size_threshold` set `1000` augmented examples are generated to reach at least `category_size_threshold` per category for the training dataset.

Here is the output of the augmentation pipeline on a data sample. It highlights the difference between the original data set and the augmented data set.

![augmentation][augmentation]

In one variation of the model the augmentation layer is always applied (with added augmentation layers): all the training data, are transformed (images are augmented on the fly). This strategy is compared/combined to a preprocessing adding augmented data to the original ones, which does not the categories with a high number of examples above the `category_size_threshold` set `1000`.

![dist. after augmentation][augmentation_dist]


### 3. Models Architecture Modeling approach

Starting from  LeNet-color, we tested several architectures/variations.
LeNet on color images was chosen as a baseline `tsc_baseline` (tsc -> traffic sign classifier). In a second model/variant a `BatchNormalization` layer was added before each of the `MaxPooling` layers (optionally) as well as an optional `Dropout` the model is called `tsc_lenet_batch-norm_dropout`.

To the previous model `tsc_lenet_batch-norm_dropout,  we added an augmentation layer. The augmentation layer is only active for training data. The resulting model is named `tsc_lenet_always_augment`; The idea is to give more robustness with a random rotation and random Gaussian noise applied to the original training data ( including the already created with a similar augmentation).
Nevertheless, this augmentation is a non-deterministic. In the last three `tsc_lenet_more_filters`, `tsc_lenet_7x7_more_filters` and `tsc_lenet_11x11_more_filters`, we combine more filters (number) per convolution layer with gradually larger filter size.

```python
logits = False # append a Softmax activation ! set to true to keep the logits output

networks = {
        "tsc_baseline": LeNet(categ_nbr=n_classes, logits=logits,
                              preprocessing_layers = preprocessing_layers,
                              name="tsc_baseline",
                              batch_norm=False,
                              dropout=0)
                            ,
        "tsc_lenet_batch-norm_dropout": LeNet(categ_nbr=n_classes, logits=logits,
                               preprocessing_layers = preprocessing_layers,
                               name="tsc_lenet_batch-norm_dropout"),
        "tsc_lenet_always_augment": LeNet(categ_nbr=n_classes, logits=logits,
                                         preprocessing_layers = preprocessing_layers,
                                         augmentation_layers = [
                                            RandomRotation(.01),
                                            RandomTranslation(.05, .05),
                                            RandomZoom(height_factor=(-0.1, 0.1)),
                                            RandomContrast(.1)
                                         ],
                                         name="tsc_lenet_always_augment"),

            "tsc_lenet_more_filters":
                LeNet(categ_nbr=n_classes, logits=logits,
                      preprocessing_layers = preprocessing_layers,

                      conv_layers_config = {
                            1: ConvLayerConfig(filters=12, kernel_size=(5, 5)),
                            2: ConvLayerConfig(filters=32, kernel_size=(5, 5)),
                            },
                       name="tsc_lenet_more_filters",
                     ),
                "tsc_lenet_7x7_more_filters":
                LeNet(categ_nbr=n_classes, logits=logits,
                      preprocessing_layers = preprocessing_layers,

                      conv_layers_config = {
                            1: ConvLayerConfig(filters=12, kernel_size=(7, 7)),
                            2: ConvLayerConfig(filters=32, kernel_size=(7, 7)),
                            },
                       name="tsc_lenet_7x7_more_filters",
                     ),
                    "tsc_lenet_11x11_more_filters":
                LeNet(categ_nbr=n_classes, logits=logits,
                      preprocessing_layers = preprocessing_layers,

                      conv_layers_config = {
                            1: ConvLayerConfig(filters=12, kernel_size=(11, 11)),
                            2: ConvLayerConfig(filters=32, kernel_size=(7, 7)),
                            },
                       name="tsc_lenet_11x11_more_filters",
                     )

           }
```

### 4. Model training

Adam optimizer is used to train the models `Adam(learning_rate=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)` with a learning rate of `0.002`. The models are trained for `50+` epochs and with a batch size equal to `256` and learning rate of `0.002`.

An EarlyStopping mechanism is implemented to restore the weight from the best performing model through epochs.

```python
early_stopping_callback = tf.keras.callbacks.EarlyStopping(
    monitor="val_accuracy",
    min_delta=0,
    patience=10,
    verbose=1,
    mode="auto",
    baseline=None,
    restore_best_weights=True,
)
```


### 5. Best Model Results (based on the validation dataset)

Below a comparison (perf. visualization) between tested model/arch. based on the accuracy on validation dataset.

![models comparison][models_comparison]


| Model | acc. orig. train| acc. valid| acc. test
|:--:|:--:|:--:|:--:|
| tsc_baseline |0.996|<p style="background:red; opacity=0.3"> 0.9117913832199547 </p>|0.905
| tsc_lenet_batch-norm_dropout |0.998|<p style="background:green; opacity=0.1"> 0.943764172335601 </p>|0.930
| tsc_lenet_always_augment |0.993|<p style="background:red; opacity=0.3"> 0.9297052154195011 </p>|0.915
| tsc_lenet_more_filters |1.000|<p style="background:green; opacity=0.1"> 0.9523809523809523 </p>|0.932
| tsc_lenet_7x7_more_filters |0.994|<p style="background:green; opacity=0.1"> 0.9532879818594104 </p>|0.930
| tsc_lenet_11x11_more_filters |0.994|<p style="background:green; opacity=0.1"> 0.946031746031746 </p>|0.926

The final/best model results, according to the valid. set accuracy is  : tsc_lenet_7x7_more_filters

(model when predicting)

![selected model pred][selected_model_pred]


## Model and challenges evaluation

To examine the reasons and root cause of the misclassification we extract here the top 10 misclassified categories with a sample visualization.

The `Speed limit (30km/h) (class 1)` exposes challenging light condition, some noise and a high intra-classes similarity between the Speed limit classes with a high resemblance between 50km/h, 70km/h, and 80km/h where a right classification relies on one digit recognition or more precisely a sub-region of the very left digit pixels. All of the misclassifications are within the speed limit ones.

Interestingly even with the low contrast shown by the `Speed limit (100km/h) (class 7)` the errors (misclassification) remain within the speed limit categories.

The light condition seems to be the main challenging element for the `Road work (class 25)` and `(id: 40) Roundabout mandatory`.

Fo the `Dangerous curve to the right (class 20)` the background context or in other words the proportion/scale of the object of interest in comparison to the image size makes the recognition tougher: fewer pixels represent the object of interest and consequently poor details. While the `Double curve (class 21)`.

For the pedestrian `Pedestrians (class 27)` the misclassifications are interestingly related to visually very similar categories namely `General caution`

![top misclassified][top_misclassified]

Classification errors detailed stats:

(id: 16) Vehicles over 3.5 metric tons prohibited, 25 incorrectly classified:
* as End of no passing -> 22 time(s)!
* as No passing -> 3 time(s)!

(id: 25) Road work, 20 incorrectly classified:
* as General caution -> 2 time(s)!
* as Right-of-way at the next intersection -> 5 time(s)!
* as Beware of ice/snow -> 4 time(s)!
* as Wild animals crossing -> 8 time(s)!
* as Speed limit (50km/h) -> 1 time(s)!

(id: 7) Speed limit (100km/h), 16 incorrectly classified:
* as Speed limit (120km/h) -> 16 time(s)!

(id: 41) End of no passing, 15 incorrectly classified:
* as End of all speed and passing limits -> 14 time(s)!
* as End of speed limit (80km/h) -> 1 time(s)!

(id: 21) Double curve, 14 incorrectly classified:
* as Road narrows on the right -> 4 time(s)!
* as Bumpy road -> 2 time(s)!
* as Right-of-way at the next intersection -> 4 time(s)!
* as Slippery road -> 4 time(s)!

(id: 20) Dangerous curve to the right, 14 incorrectly classified:
* as Right-of-way at the next intersection -> 3 time(s)!
* as Bicycles crossing -> 2 time(s)!
* as Road work -> 2 time(s)!
* as Priority road -> 2 time(s)!
* as General caution -> 1 time(s)!
* as Slippery road -> 2 time(s)!
* as Children crossing -> 1 time(s)!
* as Road narrows on the right -> 1 time(s)!

(id: 40) Roundabout mandatory, 13 incorrectly classified:
* as Speed limit (30km/h) -> 12 time(s)!
* as End of no passing -> 1 time(s)!

(id: 27) Pedestrians, 8 incorrectly classified:
* as General caution -> 8 time(s)!

(id: 30) Beware of ice/snow, 7 incorrectly classified:
* as Dangerous curve to the right -> 2 time(s)!
* as Right-of-way at the next intersection -> 1 time(s)!
* as Road work -> 2 time(s)!
* as Double curve -> 2 time(s)!

(id: 1) Speed limit (30km/h), 6 incorrectly classified:
* as Speed limit (50km/h) -> 3 time(s)!
* as Speed limit (70km/h) -> 2 time(s)!
* as Speed limit (80km/h) -> 1 time(s)!



## Test a Model on New Images

### 1. The New images quality/difficulty


To extend the testing with new images here are 19 german traffic signs collected from the internet mainly using street view, around Munich City (Germany). Some of the images include a bit more surrounding context than the ones in the training and validation data sets: Remark the white rectangle below the `Turn right ahead` examples. Also, the cropping was not perfectly square which results in distorted images when resized to the input size of the designed network. This distortion might be the most challenging. Also for two of the `Ahead only` sample, there is a bit of occlusion with tree leaves.

![new external images][ext_images]

### 2. New images prediction

Here are the results of the prediction:

| Image | Ground truth | Prediction |
|:-----:|:------------:|:----------:|
| ./examples/test_sample/13_1.png | 13, Yield|  <p style="background:green; opacity=0.3"> 13,  Yield </p>|
| ./examples/test_sample/13_2.png | 13, Yield|  <p style="background:green; opacity=0.3"> 13,  Yield </p>|
| ./examples/test_sample/13_3.png | 13, Yield|  <p style="background:green; opacity=0.3"> 13,  Yield </p>|
| ./examples/test_sample/17_1.png | 17, No entry|  <p style="background:green; opacity=0.3"> 17,  No entry </p>|
| ./examples/test_sample/17_2.png | 17, No entry|  <p style="background:green; opacity=0.3"> 17,  No entry </p>|
| ./examples/test_sample/18_1.png | 18, General caution|  <p style="background:red; opacity=0.3"> 11, Right-of-way at the next intersection </p>|
| ./examples/test_sample/18_2.png | 18, General caution|  <p style="background:green; opacity=0.3"> 18,  General caution </p>|
| ./examples/test_sample/2_1.png | 2, Speed limit (50km/h)|  <p style="background:green; opacity=0.3"> 2,  Speed limit (50km/h) </p>|
| ./examples/test_sample/33_1.png | 33, Turn right ahead|  <p style="background:red; opacity=0.3"> 40, Roundabout mandatory </p>|
| ./examples/test_sample/33_2.png | 33, Turn right ahead|  <p style="background:green; opacity=0.3"> 33,  Turn right ahead </p>|
| ./examples/test_sample/35_1.png | 35, Ahead only|  <p style="background:red; opacity=0.3"> 37, Go straight or left </p>|
| ./examples/test_sample/35_2.png | 35, Ahead only|  <p style="background:red; opacity=0.3"> 40, Roundabout mandatory </p>|
| ./examples/test_sample/35_3.png | 35, Ahead only|  <p style="background:green; opacity=0.3"> 35,  Ahead only </p>|
| ./examples/test_sample/35_4.png | 35, Ahead only|  <p style="background:green; opacity=0.3"> 35,  Ahead only </p>|
| ./examples/test_sample/38_1.png | 38, Keep right|  <p style="background:green; opacity=0.3"> 38,  Keep right </p>|
| ./examples/test_sample/38_2.png | 38, Keep right|  <p style="background:green; opacity=0.3"> 38,  Keep right </p>|
| ./examples/test_sample/3_1.png | 3, Speed limit (60km/h)|  <p style="background:green; opacity=0.3"> 3,  Speed limit (60km/h) </p>|
| ./examples/test_sample/3_2.png | 3, Speed limit (60km/h)|  <p style="background:red; opacity=0.3"> 2, Speed limit (50km/h) </p>|
| ./examples/test_sample/3_3.png | 3, Speed limit (60km/h)|  <p style="background:green; opacity=0.3"> 3,  Speed limit (60km/h) </p>|

The model was able to correctly guess 14 of the 19 traffic signs, which gives an accuracy of 0.74, with 100% success for some of the classes: Yield, No entry

### 3. Model certainty (using Softmax output)

Here we provide the top 5 softmax probabilities for each image along with the sign type of each probability.

![top 5 softmax probabilities][top_5_softmax_probs]

Interestingly for the image `test_sample/35_2.png` ) (Ahead only) the uncertainty was higher and the correct class was in the top 5 (with a considerably high probability).


## Visualizing the Neural Network activation

Below ia a visualization for the feature map for the first and second convolution filters activation for two classes: `Yield (class 13)` and `No entry (class 17)` with respectively 3 and 2 samples.


At the output of the first convolution layer we can remark some filters that are more specialized in edges extraction (FeatureMap 0 to 3) while other are focused on blobs (segmenting region) interestingly extracting the context around the traffic sign more than the inside (FeatureMap 5, 8, 10).

![activation preview][activation_preview_13_first_cov2d]
![activation preview][activation_preview_17_first_cov2d]

For the second convolution layer the activations seem to be more localized and sparse in the filter space, but interestingly consistent within the same class.

![activation preview][activation_preview_13_second_cov2d]
![activation preview][activation_preview_17_second_cov2d]


