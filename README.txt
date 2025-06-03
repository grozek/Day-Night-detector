README for outside_detection program

The program trains a model to recognize whether an image shows day or night.

It is given a set of over 750 images, each with a appropiate label - 'day' or 'night'. 
The dataset is imbalanced - there are 599 images of 'day'. 

Model not only is trained, but the results of the training are presented using show_images() function
that compares the predicted label with the target label, and creates a colored rectangle for
each correctly (green) and incorrectly (red) assigned label. Every tested image is represented
by such small rectangle, and then visualised with rectangles of all other images. This makes
the visualisation of accuracy very clear, especially when it comes to the improvement over the
course of training.

The code is divided into distinct files: 
- dataset.py: import, split and transform data into dataloaders
- model.py: setup the learning network
- train.py: train and test the dataset
- showImage.py: create the visalizaitons of accuracy of the predictions


UPDATES:


03/06/2025
Classification works with around 99% accuracy. Beginning to implement the webpage that faciliates the classification.