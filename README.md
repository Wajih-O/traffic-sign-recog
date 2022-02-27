# Traffic Sign Recognition
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

[models_comparison]: ./output/models_comparison.png "models comparison"
[top_misclassified]: ./output/misclassified/preview_0.jpg "top misclassified"

Overview
---

This project covers a training and validation of a model so it can classify traffic sign images using the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset).

After the model is trained we will test it with German traffic signs found on the web.

starter code: [Ipython notebook](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb).

**This version/repo includes updating Tensorflow to 2.7+ and a more flexible/suitable implementation of the `LeNet` architecture enabling architecture variation**



## Requirements

The models are trained using Tensorflow-GPU docker container (supporting GPU passthrough). Nevertheless, the
project works seamlessly with only the CPU. All the needed dependencies are part of the `Requirements.txt`.

## Glimpse:

### Modeling
Starting from  LeNet-color, we tested several architectures/variations: LeNet on color images was chosen as a baseline `tsc_baseline` (tsc -> traffic sign classifier) from which we develop variation that reaches the targeted accuracy `0.93`

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


Below a comparison (perf. visualization) between tested model/arch. based on the accuracy on validation dataset.

![models comparison][models_comparison]


### Challenges evaluation

To examine the reasons and root cause of the misclassification
we extract here the top 10 misclassified categories with a sample visualization.
Light condition, noise, object vs background (surrounding context size) and intra-class similarity are the main challenges.

![top misclassified][top_misclassified]



Check `writeup.md` and `Traffic_Sign_Classifier.ipynb` for more details
