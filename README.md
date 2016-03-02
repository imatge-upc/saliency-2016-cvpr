# Shallow and Deep Convolutional Networks for Saliency Prediction
## Paper accepted at [2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR)](http://cvpr2016.thecvf.com/) 


| ![Junting Pan][JuntingPan-photo]  | ![Kevin McGuinness][KevinMcGuinness-photo]  | ![Elisa Sayrol][ElisaSayrol-photo]  | ![Noel O'Connor][NoelOConnor-photo]  | ![Xavier Giro-i-Nieto][XavierGiro-photo]  |
|:-:|:-:|:-:|:-:|:-:|
| Junting Pan (*)  | [Kevin McGuinness](KevinMcGuinness-web) (*)   |  [Elisa Sayrol](Elisa Sayrol-web) | [Noel O'Connor](NoelOConnor-web)   | [Xavier Giro-i-Nieto](XavierGiro-web)   |
| [ETSETB TelecomBCN](etsetb-web)  | [Insight Centre for Data Analytics](insight-web)   |  [Image Processing Group]((gpi-web)) | [Insight Centre for Data Analytics](insight-web)   | [Image Processing Group](gpi-web)   |
| [Universitat Politecnica de Catalunya (UPC)](upc-web)  | [Dublin City University (DCU)](dcu-web)   |  [Universitat Politecnica de Catalunya (UPC)](upc-web) | [Dublin City University (DCU)](dcu-web) | [Universitat Politecnica de Catalunya (UPC)](upc-web)   |

(*) Equal contribution

[KevinMcGuinness-web]: https://www.insight-centre.org/users/kevin-mcguinness
[ElisaSayrol-web]: https://imatge.upc.edu/web/people/elisa-sayrol
[NoelOConnor-web]: https://www.insight-centre.org/users/noel-oconnor
[XavierGiro-web]: https://imatge.upc.edu/web/people/xavier-giro

[etsetb-web]: https://www.etsetb.upc.edu/en/
[gpi-web]: https://imatge.upc.edu/web/
[insight-web]: https://www.insight-centre.org/
[upc-web]: http://www.upc.edu/?set_language=en
[dcu-web]: http://www.dcu.ie/


[JuntingPan-photo]: https://github.com/imatge-upc/saliency-2016-cvpr/blob/master/authors/JuntingPan.jpg "Junting Pan"
[KevinMcGuinness-photo]: https://github.com/imatge-upc/saliency-2016-cvpr/blob/master/authors/KevinMcGuinness.jpg "Kevin McGuinness"
[ElisaSayrol-photo]: https://github.com/imatge-upc/saliency-2016-cvpr/blob/master/authors/ElisaSayrol.jpg "Elisa Sayrol"
[NoelOConnor-photo]: https://github.com/imatge-upc/saliency-2016-cvpr/blob/master/authors/NoelOConnor.jpg "Noel O'Connor"
[XavierGiro-photo]: https://github.com/imatge-upc/saliency-2016-cvpr/blob/master/authors/XavierGiro.jpg "Xavier Giro-i-Nieto"


#### Trained convnets

The two convnets presented in our work can be downloaded from the links provided below each respective figure:

| Shallow ConvNet (JuntingNet)  |  Deep ConvNet (SalNet) |
|:-:|:-:|
|  Figure | Figure  |
| [Lasagne Model (2.5 GB)](shallow-model)  | [Caffe Model](deep-model) [Caffe Prototxt](deep-prototxt)  |

[shallow-model]: https://imatge.upc.edu/web/sites/default/files/resources/1720/saliency/2016-cvpr/shallow_net.pickle
[deep-model]: https://imatge.upc.edu/web/sites/default/files/resources/1720/saliency/2016-cvpr/deep_net_model.caffemodel
[deep-prototxt]: https://imatge.upc.edu/web/sites/default/files/resources/1720/saliency/2016-cvpr/deep_net_deploy.prototxt


Our previous winning shallow models for the [LSUN Saliency Prediction Challenge 2015](http://lsun.cs.princeton.edu/#saliency) are described in [this preprint](https://imatge.upc.edu/web/publications/end-end-convolutional-network-saliency-prediction) and available from [this other site](https://imatge.upc.edu/web/resources/end-end-convolutional-networks-saliency-prediction-software). That work was also part of Junting Pan's bachelor thesis at [UPC TelecomBCN school](https://www.etsetb.upc.edu/en/) in June 2015, which report, slides and video are available [here](https://imatge.upc.edu/web/publications/visual-saliency-prediction-using-deep-learning-techniques).


### Datasets

#### Training
As explained in our paper, our networks were trained on the training and validation data provided by [SALICON](http://salicon.net/).

#### Test
Three different dataset were used for test:
* Test partition of [SALICON](http://salicon.net/) dataset.
* Test partition of [iSUN](http://vision.princeton.edu/projects/2014/iSUN/) dataset.
* [MIT300](http://saliency.mit.edu/datasets.html).

A collection of links to the SALICON and iSUN datasets is available from the [LSUN Challenge site](http://lsun.cs.princeton.edu/#saliency).

### Software frameworks

Our paper presents two different convolutional neural networks trained with different frameworks. For this reason, different instructions and source code folders are provided.

#### Shallow Network (alias JuntingNet)

The shallow network is implemented in [Lasagne](https://github.com/Lasagne/Lasagne), which at its time is developed over [Theano](http://deeplearning.net/software/theano/).
To install required version of Lasagne and all the remaining dependencies, you should run this [pip](https://pip.pypa.io/en/stable/) command.

```
pip install -r https://raw.githubusercontent.com/dnouri/kfkd-tutorial/master/requirements.txt
```

This requirements file was provided by [Daniel Nouri](http://danielnouri.org/notes/2014/12/17/using-convolutional-neural-nets-to-detect-facial-keypoints-tutorial/).

### Deep Network (alias SalNet)

The deep network was developed over [Caffe](http://caffe.berkeleyvision.org/) by [Berkeley Vision and Learning Center (BVLC)](http://bvlc.eecs.berkeley.edu/). You will need to follow [these instructions](http://caffe.berkeleyvision.org/installation.html) to install Caffe.

