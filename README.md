# mlrf

Mlrf is a small library made for a class on Machine Learning for Shape recognition.  
It provides simple classes to extract features from and classify images.
The libary was made to work specifically on the CIFAR-10 dataset.

## Instructions

To use the library, use the following steps:
-    Download this folder from it's [GitHub page](https://github.com/MRIBOUR/MLRF)
-    Download the CIFAR-10 dataset from this [Link](http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz)
-    Place the dataset in a data folder in the project on the same level as this file (path should be: mlrf/data/cifar-10-batches-py/)
-    You should now create a python environment in the git folder to use the library, and execute the command 'pip install .' inside it. If you are using linux, you can use the command 'make create_environment' to execute this step.
-    Once you have your python environment, you can open the tutorial.ipynb in the notebook folder to get instructions on how to use the library.
-    You also have multiple python scripts in the scripts folder to quickly train a model and test it's accuracy on a feature extractor.

## Project Organization

```
├── Makefile
├── README.md
├── requirements.txt
├── setup.cfg
├── setup.py
├── mlrf
│   ├── __init__.py
│   ├── cifar_utils
│   │   ├── cifar_utils.py
│   │   └── __init__.py
│   ├── feature_extraction
│   │   ├── BoVWFE.py
│   │   ├── FeatureExtractor.py
│   │   ├── FlattenFE.py
│   │   ├── HoGFE.py
│   │   └── __init__.py
│   └── models
│       ├── BaseModel.py
│       ├── __init__.py
│       ├── KMeansModel.py
│       ├── LogisticRegressionModel.py
│       └── SVCModel.py
├── models
├── notebooks
│   └── tutorial.ipynb
└── scripts
    ├── bovw_kmeans.py
    ├── bovw_logreg.py
    ├── bovw_svc.py
    ├── flatten_kmeans.py
    ├── flatten_logreg.py
    ├── flatten_svc.py
    ├── hog_kmeans.py
    ├── hog_logreg.py
    └── hog_svc.py
```

--------

