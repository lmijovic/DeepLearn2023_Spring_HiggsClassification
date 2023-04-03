# Deep learning & the Higgs boson: classification with fully connected and adversarial networks

[introductory/intermediate]

## Summary

The Higgs boson is a fundamental particle, responsible for the origin of mass. How is machine learning used to investigate it's properties in the data by the Large Hadron Collider at CERN? These lectures challenge you to find out. You will be faced with a tricky Higgs classification problem, in which the classification has to be independent of a specific property of the events. Our goal will be to solve this problem with an adversarial neural network, an architecture closely related to generative adversarial networks. Upon completion, you will have gained an overview of machine learning challenges in particle physics, and acquired skills to solve non-standard machine learning problems in Keras and Tensorflow. 

## Software 

We will work in Python3 and Tensorflow2. Please install a working environment with the packages needed to run the tutorials: https://github.com/lmijovic/DeepLearn2023_Spring_HiggsClassification/blob/master/code/requirements.txt . You can use this file to create a conda environment as follows:


```

conda create --name DeepLearn23 --file requirements.txt 


```


## Before the lecture 

### Download the input data 

Please aim to have ~ 1GB of free disk-space available, and download the input data:
 
https://cern.ch/dl23data

The data is derived from ATLAS H->yy CERN OpenData sample DOI:10.7483/OPENDATA.ATLAS.B5BJ.3SGS.

### Test your environment 

(1) Clone this directory: 

```

git clone https://github.com/lmijovic/DeepLearn2023_Spring_HiggsClassification.git 

```

(2) Make sure your conda environment is activated:

```

conda activate DeepLearn23

```

(3) run the test: 

```

cd DeepLearn2023_Spring_HiggsClassification/code/test
python test.py 

```

If you get no errors (ignore tensorflow warnings), and you get non-empty output files: clf_results.csv, ANN_results.csv, all is working fine.

## Backup: binder

The classification requires a large amount of data, and should therefore be done on the local machine to get good results. 

Backup, binder running on small statistics: 

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/lmijovic/test_binder_dl23/HEAD)


## Syllabus 

- Lecture1: The Higgs boson and event classification
After this lecture, you will be able to perform event classification with fully connected neural network and evaluate classification performance using Keras API.

- Lecture2: Solving the background sculpting challenge
This lecture will provide you with hands-on knowledge of manipulating neural networks in Tensorflow.

- Lecture3: Putting it all together: event classification with adversarial neural network
Finally, you will create an adversarial network, and compare it's classification performance to the fully connected network.

## Links

- Lecture1: 
* Please upload results to: https://cern.ch/dl23lect1
* Please upload any feedback to: https://cern.ch/l1feed

- Lecture2:

- Lecture3: 


## References 

[1] Introduction to the Higgs boson: CERN, https://home.cern/science/physics/higgs-boson/how

[2] Gilles Louppe, Michael Kagan, Kyle Cranmer, Learning to Pivot with Adversarial Networks, https://arxiv.org/abs/1611.01046

[3] Chase Shimmin et al, Decorrelated Jet Substructure Tagging using Adversarial Neural Networks, https://arxiv.org/abs/1703.03507

[4] Andeas Sogaard's notebook on Adversarial neural networks: https://github.com/asogaard/ep2mlf/tree/master/01-adversarial

## Credits

Most of the code and ideas for these lectures are thanks to my wonderful colleagues and students: Dr. Andreas Sogaard, Dr. Emily Takeva, Keira Farmer, Jenifer Curran, Stefan Katsarov, James Cranmer. 
