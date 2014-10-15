Text Recognition in Natural Images in Python  
========

This repository contains the code for the OCR Project I'm working on as part of **Data Science Retreat** (Berlin).

The idea is to be able to get as input an image (i.e. picture taken with phone) from a user and process it in order to return the text contained in it.

The file you should look at first is **main.py**. This file instantiates classes and calls appropriate classes methods.

**Class UserData** (contained in userimageski.py) is instantiated passing to the constructor the image filename to be processed.

**Class Cifar** (contained in cifar.py) is instantiated passing a config file (with all the parameters) to the constructor. Specifically cifar-config.py, which is used in the merge-with-cifar method inside OcrData class in order to build the dataset to perform the text/no-text classification. The data mentioned inside cifar-config.py was too big to be uploaded on Github but it is available on [CIFAR-10 Kaggle Competition](http://www.kaggle.com/c/cifar-10/data).

**Class OcrData** (contained in data.py) is instantiated passing a config file (with all the parameters) to the constructor. Specifically ocr-config.py and text-config.py are both used in two different contextes. The first one is called to perform the machine learning pipeline on character images. The second one is called only once inside the merge-with-cifar method inside OcrData class in order to build the dataset to perform the text/no-text classification. 

The data used for this project is the **Chars74K dataset** which can be found [here](http://www.ee.surrey.ac.uk/CVSSP/demos/chars74k/).

A complete explanation of the work can be found on my [website](http://francescopochetti.com/portfoliodata-science-machine-learning/).

------------------------------------------------------------------------------------

Ideas for improvement:

1. Get rid of nested rectangles in object detection. Solves the problem of detecting a circle (classified as an 'o') inside an 'a'.

2. Manually labeling objects containing or not containing text. It is possible to add a wait_for_key during the object detection phase and as soon as a rectangle is identified manually specify if it's text or not. For example a tree may be miscassified as text and then classified as a T. Manual detection is very time consuming and before diving into that it is necessary to analyze the pipeline and be sure that it is worth doing it.

3. Introduce as a final step a 'Guess Missing Text Phase' to correct little mistakes. For example if in the end we should detect the word 'house' but we identify 'hous', well of course that's a house! 

4. Implement Neural Network and Deep Learning.
