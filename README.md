# Shallow and Deep Convolutional Networks for Saliency Prediction

|  ![CVPR 2016 logo][logo-cvpr] | Paper accepted at [2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR)](http://cvpr2016.thecvf.com/)   |
|:-:|---|

[logo-cvpr]: https://raw.githubusercontent.com/imatge-upc/saliency-2016-cvpr/master/logos/cvpr2016.jpg "CVPR 2016 logo"

| ![Junting Pan][JuntingPan-photo]  | ![Kevin McGuinness][KevinMcGuinness-photo]  | ![Elisa Sayrol][ElisaSayrol-photo]  | ![Noel O'Connor][NoelOConnor-photo]  | ![Xavier Giro-i-Nieto][XavierGiro-photo]  |
|:-:|:-:|:-:|:-:|:-:|
| [Junting Pan][JuntingPan-web] (*)| [Kevin McGuinness][KevinMcGuinness-web] (*)|  [Elisa Sayrol][ElisaSayrol-web] | [Noel O'Connor][NoelOConnor-web]  | [Xavier Giro-i-Nieto][XavierGiro-web]|

(*) Equal contribution

[JuntingPan-web]: https://junting.github.io/
[KevinMcGuinness-web]: https://www.insight-centre.org/users/kevin-mcguinness
[ElisaSayrol-web]: https://imatge.upc.edu/web/people/elisa-sayrol
[NoelOConnor-web]: https://www.insight-centre.org/users/noel-oconnor
[XavierGiro-web]: https://imatge.upc.edu/web/people/xavier-giro

[JuntingPan-photo]: https://raw.githubusercontent.com/imatge-upc/saliency-2016-cvpr/master/authors/JuntingPan.jpg "Junting Pan"
[KevinMcGuinness-photo]: https://raw.githubusercontent.com/imatge-upc/saliency-2016-cvpr/master/authors/KevinMcGuinness.jpg "Kevin McGuinness"
[ElisaSayrol-photo]: https://raw.githubusercontent.com/imatge-upc/saliency-2016-cvpr/master/authors/ElisaSayrol.jpg "Elisa Sayrol"
[NoelOConnor-photo]: https://raw.githubusercontent.com/imatge-upc/saliency-2016-cvpr/master/authors/NoelOConnor.jpg "Noel O'Connor"
[XavierGiro-photo]: https://raw.githubusercontent.com/imatge-upc/saliency-2016-cvpr/master/authors/XavierGiro.jpg "Xavier Giro-i-Nieto"

A joint collaboration between:

| ![logo-insight] | ![logo-dcu] | ![logo-upc] | ![logo-etsetb] | ![logo-gpi] | 
|:-:|:-:|:-:|:-:|:-:|
| [Insight Centre for Data Analytics][insight-web] | [Dublin City University (DCU)][dcu-web]  |[Universitat Politecnica de Catalunya (UPC)][upc-web]   | [UPC ETSETB TelecomBCN][etsetb-web]  | [UPC Image Processing Group][gpi-web] | 

[insight-web]: https://www.insight-centre.org/ 
[dcu-web]: http://www.dcu.ie/
[upc-web]: http://www.upc.edu/?set_language=en 
[etsetb-web]: https://www.etsetb.upc.edu/en/ 
[gpi-web]: https://imatge.upc.edu/web/ 


[logo-insight]: https://raw.githubusercontent.com/imatge-upc/saliency-2016-cvpr/master/logos/insight.jpg "Insight Centre for Data Analytics"
[logo-dcu]: https://raw.githubusercontent.com/imatge-upc/saliency-2016-cvpr/master/logos/dcu.png "Dublin City University"
[logo-upc]: https://raw.githubusercontent.com/imatge-upc/saliency-2016-cvpr/master/logos/upc.jpg "Universitat Politecnica de Catalunya"
[logo-etsetb]: https://raw.githubusercontent.com/imatge-upc/saliency-2016-cvpr/master/logos/etsetb.png "ETSETB TelecomBCN"
[logo-gpi]: https://raw.githubusercontent.com/imatge-upc/saliency-2016-cvpr/master/logos/gpi.png "UPC Image Processing Group"


## Abstract

The prediction of salient areas in images has been traditionally addressed with hand-crafted features based on neuroscience principles. This paper, however, addresses the problem with a completely data-driven approach by training a convolutional neural network (convnet). The learning process is formulated as a minimization of a loss function that measures the Euclidean distance of the predicted saliency map with the provided ground truth. The recent publication of large datasets of saliency prediction has provided enough data to train end-to-end architectures that are both fast and accurate. Two designs are proposed: a shallow convnet trained from scratch, and a another deeper solution whose first three layers are adapted from another network trained for classification.
To the authors knowledge, these are the first end-to-end CNNs trained and tested for the purpose of saliency prediction

## Publication

[Our paper](http://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Pan_Shallow_and_Deep_CVPR_2016_paper.pdf) is open published thanks to the Computer Science Foundation. An [arXiv pre-print](http://arxiv.org/abs/1603.00845) is also available. 

![Image of the paper](https://raw.githubusercontent.com/imatge-upc/saliency-2016-cvpr/master/figs/paper.jpg)

Please cite with the following Bibtex code:

````
@InProceedings{Pan_2016_CVPR,
author = {Pan, Junting and Sayrol, Elisa and Giro-i-Nieto, Xavier and McGuinness, Kevin and O'Connor, Noel E.},
title = {Shallow and Deep Convolutional Networks for Saliency Prediction},
booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2016}
}
````

You may also want to refer to our publication with the more human-friendly Chicago style:

*Junting Pan, Kevin McGuinness, Elisa Sayrol, Noel E. O'Connor, and Xavier Giro-i-Nieto. "Shallow and Deep Convolutional Networks for Saliency Prediction." In Proceedings of the IEEE International Conference on Computer Vision (CVPR). 2016.*

## Models

The two convnets presented in our work can be downloaded from the links provided below each respective figure:

| Shallow ConvNet (aka JuntingNet)  |  Deep ConvNet (aka SalNet) |
|:-:|:-:|
|  ![shallow-fig] | ![deep-fig]  |
| [[Lasagne Model (2.5 GB)]](https://imatge.upc.edu/web/sites/default/files/resources/1720/saliency/2016-cvpr/shallow_net.pickle)  | [[Caffe Model (99 MB)]](https://imatge.upc.edu/web/sites/default/files/resources/1720/saliency/2016-cvpr/deep_net_model.caffemodel) [[Caffe Prototxt]](https://imatge.upc.edu/web/sites/default/files/resources/1720/saliency/2016-cvpr/deep_net_deploy.prototxt)  |

[shallow-fig]: https://raw.githubusercontent.com/imatge-upc/saliency-2016-cvpr/master/figs/shallow.png "Shallow convnet architecture"
[deep-fig]: https://raw.githubusercontent.com/imatge-upc/saliency-2016-cvpr/master/figs/deep.png "Deep convnet architecture"

[shallow-model]: https://imatge.upc.edu/web/sites/default/files/resources/1720/saliency/2016-cvpr/shallow_net.pickle
[deep-model]: https://imatge.upc.edu/web/sites/default/files/resources/1720/saliency/2016-cvpr/deep_net_model.caffemodel
[deep-prototxt]: https://imatge.upc.edu/web/sites/default/files/resources/1720/saliency/2016-cvpr/deep_net_deploy.prototxt


Our previous winning shallow models for the [LSUN Saliency Prediction Challenge 2015](http://lsun.cs.princeton.edu/#saliency) are described in [this preprint](https://imatge.upc.edu/web/publications/end-end-convolutional-network-saliency-prediction) and available from [this other site](https://imatge.upc.edu/web/resources/end-end-convolutional-networks-saliency-prediction-software). That work was also part of Junting Pan's bachelor thesis at [UPC TelecomBCN school](https://www.etsetb.upc.edu/en/) in June 2015, which report, slides and video are available [here](https://imatge.upc.edu/web/publications/visual-saliency-prediction-using-deep-learning-techniques).

## Visual Results

![Qualitative saliency predictions](https://raw.githubusercontent.com/imatge-upc/saliency-2016-cvpr/master/figs/qualitative.jpg)


## Datasets

### Training
As explained in our paper, our networks were trained on the training and validation data provided by [SALICON](http://salicon.net/).

### Test
Three different dataset were used for test:
* Test partition of [SALICON](http://salicon.net/) dataset.
* Test partition of [iSUN](http://vision.princeton.edu/projects/2014/iSUN/) dataset.
* [MIT300](http://saliency.mit.edu/datasets.html).

A collection of links to the SALICON and iSUN datasets is available from the [LSUN Challenge site](http://lsun.cs.princeton.edu/#saliency).

## Software frameworks

Our paper presents two different convolutional neural networks trained with different frameworks. For this reason, different instructions and source code folders are provided.

### Shallow Network on Lasagne

The shallow network is implemented in [Lasagne](https://github.com/Lasagne/Lasagne), which at its time is developed over [Theano](http://deeplearning.net/software/theano/).
To install required version of Lasagne and all the remaining dependencies, you should run this [pip](https://pip.pypa.io/en/stable/) command.

```
pip install -r https://raw.githubusercontent.com/imatge-upc/saliency-2016-cvpr/master/shallow/requirements.txt
```

This requirements file was provided by [Daniel Nouri](http://danielnouri.org/notes/2014/12/17/using-convolutional-neural-nets-to-detect-facial-keypoints-tutorial/).

### Deep Network on Caffe

The deep network was developed over [Caffe](http://caffe.berkeleyvision.org/) by [Berkeley Vision and Learning Center (BVLC)](http://bvlc.eecs.berkeley.edu/). You will need to follow [these instructions](http://caffe.berkeleyvision.org/installation.html) to install Caffe.

## Posterior work

If you were interested in this work, you may want to also check our posterior work, [SalGAN](https://imatge-upc.github.io/saliency-salgan-2017/), which offers a better performance.

## Acknowledgements

We would like to especially thank Albert Gil Moreno and Josep Pujal from our technical support team at the Image Processing Group at the UPC.

| ![AlbertGil-photo]  | ![JosepPujal-photo]  |
|:-:|:-:|
| [Albert Gil](AlbertGil-web)  |  [Josep Pujal](JosepPujal-web) |

[AlbertGil-photo]: https://raw.githubusercontent.com/imatge-upc/saliency-2016-cvpr/master/authors/AlbertGil.jpg "Albert Gil"
[JosepPujal-photo]: https://raw.githubusercontent.com/imatge-upc/saliency-2016-cvpr/master/authors/JosepPujal.jpg "Josep Pujal"

[AlbertGil-web]: https://imatge.upc.edu/web/people/albert-gil-moreno
[JosepPujal-web]: https://imatge.upc.edu/web/people/josep-pujal

|   |   |
|:--|:-:|
|  We gratefully acknowledge the support of [NVIDIA Corporation](http://www.nvidia.com/content/global/global.php) with the donation of the GeoForce GTX [Titan Z](http://www.nvidia.com/gtx-700-graphics-cards/gtx-titan-z/) and [Titan X](http://www.geforce.com/hardware/desktop-gpus/geforce-gtx-titan-x) used in this work. |  ![logo-nvidia] |
|  The Image ProcessingGroup at the UPC is a [SGR14 Consolidated Research Group](https://imatge.upc.edu/web/projects/sgr14-image-and-video-processing-group) recognized and sponsored by the Catalan Government (Generalitat de Catalunya) through its [AGAUR](http://agaur.gencat.cat/en/inici/index.html) office. |  ![logo-catalonia] |
|  This work has been developed in the framework of the project [BigGraph TEC2013-43935-R](https://imatge.upc.edu/web/projects/biggraph-heterogeneous-information-and-graph-signal-processing-big-data-era-application), funded by the Spanish Ministerio de Economía y Competitividad and the European Regional Development Fund (ERDF).  | ![logo-spain] | 
|  This publication has emanated from research conducted with the financial support of Science Foundation Ireland (SFI) under grant number SFI/12/RC/2289. |  ![logo-ireland] |

[logo-nvidia]: https://raw.githubusercontent.com/imatge-upc/saliency-2016-cvpr/master/logos/nvidia.jpg "Logo of NVidia"
[logo-catalonia]: https://raw.githubusercontent.com/imatge-upc/saliency-2016-cvpr/master/logos/generalitat.jpg "Logo of Catalan government"
[logo-spain]: https://raw.githubusercontent.com/imatge-upc/saliency-2016-cvpr/master/logos/MEyC.png "Logo of Spanish government"
[logo-ireland]: https://raw.githubusercontent.com/imatge-upc/saliency-2016-cvpr/master/logos/sfi.png "Logo of Science Foundation Ireland"

## Contact

If you have any general doubt about our work or code which may be of interest for other researchers, please use the [public issues section](https://github.com/imatge-upc/saliency-2016-cvpr/issues) on this github repo. Alternatively, drop us an e-mail at <mailto:xavier.giro@upc.edu>.

<!---
Javascript code to enable Google Analytics
-->

<!---
<script>
-->
<!---
(function(i,s,o,g,r,a,m){i['GoogleAnalyticsObject']=r;i[r]=i[r]||function(){
(i[r].q=i[r].q||[]).push(arguments)},i[r].l=1*new Date();a=s.createElement(o),
m=s.getElementsByTagName(o)[0];a.async=1;a.src=g;m.parentNode.insertBefore(a,m)
})(window,document,'script','//www.google-analytics.com/analytics.js','ga');
-->
<!---
ga('create', 'UA-7678045-3', 'auto');
ga('send', 'pageview');
-->
<!---
</script>
-->
