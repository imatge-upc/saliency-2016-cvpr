# Shallow and Deep Convolutional Networks for Saliency Prediction
## Paper accepted at [2016 IEEE Conference onComputer Vision and Pattern Recognition (CVPR)](http://cvpr2016.thecvf.com/) 


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


Our paper presents two different convolutional neural networks trained with different frameworks. For this reason, different instructions and source code folders are provided.

### Shallow Network (alias JuntingNet)

#### Software framework

This network is implemented in [Lasagne](https://github.com/Lasagne/Lasagne), which at its time is developed over [Theano](http://deeplearning.net/software/theano/).
To install required version of Lasagne and all the remaining dependencies, you should run this [pip](https://pip.pypa.io/en/stable/) command.

```
pip install -r https://raw.githubusercontent.com/dnouri/kfkd-tutorial/master/requirements.txt
```

This requirements file was provided by [Daniel Nouri](http://danielnouri.org/notes/2014/12/17/using-convolutional-neural-nets-to-detect-facial-keypoints-tutorial/).

#### Trained convnets

As explained in our paper, our shallow network was trained with two different datasets to generate the winning entries in the [LSUN Saliency Prediction Challenge 2015](http://lsun.cs.princeton.edu/#saliency):
* [Shallow ConvNet](https://imatge.upc.edu/web/sites/default/files/resources/1720/saliency/models/JuntingNet_iSUN.pickle) trained with [iSUN](http://vision.princeton.edu/projects/2014/iSUN/) training and validation data - 819 MB
* [Shallow ConvNet](https://imatge.upc.edu/web/sites/default/files/resources/1720/saliency/models/JuntingNet_SALICON.pickle) trained with [SALICON](http://salicon.net/) training and validation data - 2.5 GB 

Links to these datasets are available from the [LSUN Challenge site](http://lsun.cs.princeton.edu/#saliency).

### Deep Network (alias SalNet)

#### Software framework

The deep network is developed over [Caffe](http://caffe.berkeleyvision.org/) by [Berkeley Vision and Learning Center (BVLC)](http://bvlc.eecs.berkeley.edu/). You will need to follow [these instructions](http://caffe.berkeleyvision.org/installation.html) to install Caffe.

#### Trained convnet

The two files that define our deep convnet over Caffe are the following:



This model was trained only with SALICON data.
