import os
import cPickle
import numpy as np
from skimage.io import imread
from skimage.transform import resize
from glob import glob
import string
from matplotlib import pyplot as plt
import sys
from random import seed, sample
from pprint import pprint
from datetime import datetime
from sklearn.base import BaseEstimator
from skimage.feature import hog
from skimage import color
from sklearn import cross_validation
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix
from nolearn.convnet import ConvNetFeatures
from sklearn.metrics import accuracy_score
from cifar import Cifar

class OcrData():
    """
    class in charge of creating and dealing with image objects.
    The goal of this class is to return clean data which is ready for a Machine Learning Pipeline.
    total images = 78905
    WINDOWS:
    -- folder_labels='D:\CharacterProject\ImageTree'
    -- folder_data='D:\CharacterProject'
    LINUX:
    -- folder_labels='/media/francesco/Francesco/CharacterProject/ImageTree'
    -- folder_data='/media/francesco/Francesco/CharacterProject'
    """
    
    def __init__(self, config):
        """
        builds the constructor by reading the config.py file and initializing parameters.
        It also automatically loads the images and in case splits the data into train and test set. 
        """
        self.config = self._load_config(config)
        self.folder_labels = self.config['folder_labels']
        self.folder_data = self.config['folder_data']
        self.verbose = self.config['verbose']
        self.img_size = self.config['img_size']
        self.limit = self.config['limit']
        self.pickle_data = self.config['pickle_data']
        self.from_pickle = self.config['from_pickle']
        self.automatic_split = self.config['automatic_split']
        self.plot_evaluation = self.config['plot_evaluation']
        self.split = self.config['percentage_of_test_set']
        self.cross_val_models = self.set_models()
        self.load()
        if self.automatic_split:
            self.split_train_test()
        
        
#######################################################################################################################

    def _load_config(self, filename):
        """
        Reads a config.py file and returns the python dictionary with all parameters
        """
        return eval(open(filename).read())        
        
########################################################################################################################        
        
    def getRelativePath(self):
        """
        Fetches the relative path of all the images and returns them into a list.
        The paths will be used to load the images by the load method.
        Images fetched from the 3 available datasets:
        - Img
        - Fnt
        - Hnd
        """
        mfiles = [os.path.join(self.folder_labels,mfile) for mfile in glob(os.path.join(self.folder_labels,'*.m'))]

        self.images = []
        
        for mfile in mfiles:
            m = open(mfile, "r")
            lines = m.readlines()
            for index, line in enumerate(lines):
                if line.startswith('list.ALLnames'):
                    start_index = index
                    start_image = line[18:].strip()[:-1 ]
                    if 'Img' in mfile:
                        self.images.append(os.path.join(*(['Englishimg','Img'] + start_image.split('/'))))
                    elif 'Fnt' in mfile:
                        self.images.append(os.path.join(*(['EnglishFnt','Fnt'] + start_image.split('/'))))
                    elif 'Hnd' in mfile:
                        self.images.append(os.path.join(*(['EnglishHnd','Hnd'] + start_image.split('/'))))
                elif line.startswith('list.classlabels'):
                    end_index = index - 1
            if 'Img' in mfile:
                self.images += [os.path.join(*(['Englishimg','Img'] + line.strip()[1:-1].split('/'))) for line in lines[start_index+1:end_index]]
            elif 'Fnt' in mfile:
                self.images += [os.path.join(*(['EnglishFnt','Fnt'] + line.strip()[1:-1].split('/'))) for line in lines[start_index+1:end_index]]
            elif 'Hnd' in mfile:
                self.images += [os.path.join(*(['EnglishHnd','Hnd'] + line.strip()[1:-1].split('/'))) for line in lines[start_index+1:end_index]]
            m.close()
        
        if self.verbose:
            print 'Found {} images.'.format(len(self.images))
    
        return self.images
    
###################################################################################################################################################    
    
    def getLabels(self):
        """
        Fetches the labels of all the images and returns them into a list.
        Once loaded the images will be labeled accordingly.
        Images fetched from the 3 available datasets:
        - Img
        - Fnt
        - Hnd     
        In the original dataset there are 62 classes:
        - [0-9] --> 10
        - [A-Z] --> 26
        - [a-z] --> 26
        For the sake of simplicity lowercase is considered the same as uppercase and now we have 36 classes:
        - [0-9] --> 10
        - [(a == A)-(z == Z)]
        """
        mfiles = [os.path.join(self.folder_labels,mfile) for mfile in glob(os.path.join(self.folder_labels,'*.m'))]

        self.labels = []        
 
        for mfile in mfiles:
            m = open(mfile, "r")
            lines = m.readlines()
            for index, line in enumerate(lines):
                if line.startswith('list.ALLlabels'):
                    start_index = index
                    start_label = line[18:].strip()[:-1 ]
                    self.labels.append(start_label)
                    
                elif line.startswith('list.ALLnames'):
                    end_index = index - 1
            self.labels += [line.strip()[:-1] for line in lines[start_index+1:end_index]]
            m.close()

        keys = range(1,63)
        values = map(str, range(10)) + list(string.ascii_lowercase) + list(string.ascii_lowercase) 
        
        classes = dict(zip(keys, values))
        self.labels = map(lambda x: classes[int(x)], self.labels)
        
        if self.verbose:
            print 'Found {} labels.'.format(len(self.labels))        
           
        return self.labels 
 
##################################################################################################################################################       
        
    def load(self):
        """
        if from_pickle == False the load method gets the relative paths of the images and their labels,
        zips them together, loads the images in greyscale, resizes them to img_size, flattens them, shuffles randomly 
        the loaded data and returns the following dictionary:
         - images --> shuffled (M x N) images
         - data --> matrix of flattened images (n_images x (M x N))
         - target --> labels of each image 
        if from_pickle == True and pickle_data == 'path/to/pickle/dictionary' the load method simply 
        returns the same dictionary as before previously loaded and saved.
        """
        
        if self.from_pickle:
            try:
                with open(os.path.join(self.folder_data,self.pickle_data), 'rb') as fin:
                    self.ocr = cPickle.load(fin)
                    if self.limit==0:
                        pass
                    else:
                        self.ocr = {
                                   'images': self.ocr['images'][:self.limit],
                                   'data': self.ocr['data'][:self.limit],
                                   'target': self.ocr['target'][:self.limit]
                                   }
                    if self.verbose:
                        print 'Loaded {} images each {} pixels'.format(self.ocr['images'].shape[0], self.img_size)
                    return self.ocr
                
            except IOError:
                print 'You have not provided a .pickle file to load data from!'
                sys.exit(0)
        else:
            image_paths = self.getRelativePath()
            image_labels = self.getLabels()
            
            if self.limit == 0:
                complete = zip(image_paths, image_labels)
            else:
                complete = zip(image_paths[:self.limit], image_labels[:self.limit])
            n_images = len(complete)
            im = np.zeros((n_images,) + self.img_size)        
            labels = []
            i=0
            
            for couple in complete:
                image = imread(os.path.join(self.folder_data, couple[0] + '.png'), as_grey=True)
                sh = image.shape
                if ((sh[0]*sh[1]) >= (self.img_size[0]*self.img_size[1])):
                    im[i] = resize(image, self.img_size)
                    i+=1
                    labels.append(couple[1])          
            im = im[:len(labels)]   
            
            seed(10)
            k = sample(range(len(im)), len(im))
            im_shuf = im[k]
            labels_shuf = np.array(labels)[k]
                
            if self.verbose:
                print 'Loaded {} images each {} pixels'.format(len(labels), self.img_size)
            
            self.ocr = {            
                 'images': im_shuf,
                 'data': im_shuf.reshape((im_shuf.shape[0], -1)), # / 255.0
                 'target': labels_shuf
                 }
            
            now = str(datetime.now()).replace(':','-')   
            fname_out = 'images-{}-{}-{}.pickle'.format(len(labels), self.img_size, now)
            full_name = os.path.join(self.folder_data,fname_out)
            with open(full_name, 'wb') as fout:
                cPickle.dump(self.ocr, fout, -1)
                
            return self.ocr

###########################################################################################################################################

    def split_train_test(self):
        """
        given the dictionary returned by the load method it returns two datasets: 
        - train set --> (1-self.split)% of originally loaded data
        - test set --> self.split% of originally loaded data
        """ 

        seed(10)
        total = len(self.ocr['target'])
        population = range(total)
        if self.split==0:
            self.images_train = self.ocr['images']
            self.data_train = self.ocr['data']
            self.labels_train = self.ocr['target']
            
            return self.images_train, self.data_train, self.labels_train
        else:
            k = int(np.floor(total * self.split))
            test = sample(population, k)
            train = [i for i in population if i not in test]
            
            self.images_train = self.ocr['images'][train]
            self.data_train = self.ocr['data'][train]
            self.labels_train = self.ocr['target'][train]
            
            self.images_test = self.ocr['images'][test]
            self.data_test = self.ocr['data'][test]
            self.labels_test = self.ocr['target'][test]
    
            return self.images_train, self.data_train, self.labels_train, self.images_test, self.data_test, self.labels_test
        
###########################################################################################################################################

    def plot_some(self):
        """
        plots 100 images with relative label randomly picked from loaded data.
        """
        n_images = self.ocr['images'].shape[0]
    
        fig = plt.figure(figsize=(12, 12))
        fig.subplots_adjust(
            left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)
    
        for i, j in enumerate(np.random.choice(n_images, 100)):
            ax = fig.add_subplot(10, 10, i + 1, xticks=[], yticks=[])
            ax.imshow(self.ocr['images'][j], cmap="Greys_r")
            ax.text(2, 7, str(self.ocr['target'][j]), fontsize=25, color='red')
        plt.show()       

#########################################################################################################################################

    def set_models(self):
        """
        sets the ML Algorithms + Parameters which will be used during CV.
        Returns a dictionary which is ready to be taken as input by GridSearchCV.
        """

        models = {
            'linearsvc': (
                LinearSVC(),
                {'C':  list(np.arange(0.01,1.5,0.01))}, 
                ),

            'linearsvc-hog': (
                Pipeline([
                    ('hog', HOGFeatures(
                        orientations=2,
                        pixels_per_cell=(2, 2),
                        cells_per_block=(2, 2),
                        size = self.img_size
                        )), ('clf', LinearSVC(C=1.0))]),

                {
                    'hog__orientations': [2, 4, 5, 10],
                    'hog__pixels_per_cell': [(2,2), (4,4), (5,5)],
                    'hog__cells_per_block': [(2,2), (4,4), (5,5)],
                    'clf__C': [0.01, 0.05, 0.1, 0.5, 1, 1.5, 2, 5, 10],
                    },
                ),
            }

        return models


####################################################################################################################################

    def perform_grid_search_cv(self, model_name):
        """
        given a labeled train set (X_train, y_train) and a model_name among the
        ones set by the set_models method, returns the best model out of all
        parameters combinations using the specified algorithm
        """
        if not self.automatic_split:
            print 'Before performing any ML you should split your data!'
            print 'Change to True the automatic_split in the config file.'
            sys.exit(0)
            
        model, param_grid = self.cross_val_models[model_name]
        
        print 'Model: ', model_name
        print 'Parameters: ', param_grid
        print 'Train set shape: ', self.data_train.shape
        print 'Target shape: ', self.labels_train.shape
        
        gs = GridSearchCV(model, param_grid, n_jobs=-1, cv=3, verbose=4)
        gs.fit(self.data_train, self.labels_train)
     
        pprint(sorted(gs.grid_scores_, key=lambda x: -x.mean_validation_score))
 
        now = str(datetime.now()).replace(':','-')   
        fname_out = '{}-{}.pickle'.format(model_name, now)
        full_name = os.path.join(self.folder_data,fname_out)
 
        with open(full_name, 'wb') as fout:
            cPickle.dump(gs, fout, -1)
     
        print "Saved model to {}".format(full_name)

###############################################################################################################################

    def perform_convnet(self):
        """
        trains a model on data using pre-trained NN to extract features and then using SVM with linear kernel.
        """
        n_images = self.images_train.shape[0]
        print 'Preparing to turn {} to RGB.'.format(n_images)
        size = (self.img_size[0], self.img_size[1], 3)
        colored = np.zeros((n_images,) + size)
        for i in range(n_images):
            colored[i] =  color.gray2rgb(self.images_train[i])
        print 'Turned {} images to {} shape.'.format(colored.shape[0], size)
        for c in [0.01, 0.1, 1, 2, 10]:
            print 'Fitting Pipeline (NN + SVC) C=', c
            clf = Pipeline([
                            ('convnet', ConvNetFeatures(
                                        pretrained_params='/home/francesco/BigData/Kaggle/CatsDogs/imagenet.decafnet.epoch90',
                                        pretrained_meta='/home/francesco/BigData/Kaggle/CatsDogs/imagenet.decafnet.meta',
                                        )), 
                            ('clf', LinearSVC(C=c))])
            scores = cross_validation.cross_val_score(clf, colored, self.labels_train, cv=5, scoring='accuracy') 
            print("Accuracy C=%0.3f : %0.2f (+/- %0.2f)" % (c, scores.mean(), scores.std() * 2))     

###############################################################################################################################

    def generate_best_hog_model(self):
        """
        given the best parameters out of grid search returns best model on all train set using
        Pipeline(hog + linearsvc).  
        """
        
        clf = Pipeline([('hog', HOGFeatures(orientations=10, pixels_per_cell=(5,5), cells_per_block=(2,2), size = self.img_size)), 
                        ('clf', LinearSVC(C=2.0))])        

        clf.fit(self.data_train, self.labels_train)
        y_pred = clf.predict(self.data_train)
        
        print 'Accuracy on train set: ', accuracy_score(self.labels_train, y_pred)

        now = str(datetime.now()).replace(':','-')   
        fname_out = 'linearsvc-hog-fulltrain-{}.pickle'.format(now)
        full_name = os.path.join(self.folder_data,fname_out)
 
        with open(full_name, 'wb') as fout:
            cPickle.dump(clf, fout, -1)
     
        print "Saved model to {}".format(full_name)        
        
################################################################################################################################

    def evaluate(self, model_filename):
        """
        Evaluates best model out of CV on test set
        """
        if not self.automatic_split:
            print 'Before performing any ML you should split your data!'
            print 'Change to True the automatic_split in the config file.'
            sys.exit(0)
            
        if self.split==0:
            print 'The percentage_of_test_set in the config.py is set to 0.'
            print 'Thus you do not have a test set to evaluate your model on.'
            sys.exit(0)
                               
        with open(model_filename, 'rb') as fin:
            model = cPickle.load(fin)
     
        y_pred = model.predict(self.data_test)
        print 'Test set shape: ', self.data_test.shape
        print 'Target shape: ', self.labels_test.shape
        print 'Accuracy on train set: ', accuracy_score(self.labels_train, model.predict(self.data_train))
        print 'Accuracy on test set: ', accuracy_score(self.labels_test, y_pred)

        if self.plot_evaluation:      
            target_names = sorted(np.unique(self.labels_test))
            n_images = self.data_test.shape[0]    
            fig = plt.figure(figsize=(6, 6))
            fig.subplots_adjust(
                left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)
         
            for i, j in enumerate(np.random.choice(n_images, 64)):
                ax = fig.add_subplot(8, 8, i + 1, xticks=[], yticks=[])
                ax.imshow(self.images_test[j], cmap="Greys_r")
                predicted = model.predict(np.array([self.data_test[j]]))[0]
                if predicted == self.labels_test[j]:
                    color = 'black'
                else:
                    color = 'red'
                ax.text(2, 7, predicted, fontsize=25, color=color)  
            plt.show()      
            
            cm = confusion_matrix(self.labels_test, y_pred)
            plt.matshow(cm)
            plt.colorbar()
            plt.ylabel('True label')
            plt.xlabel('Predicted label')
            plt.xticks(range(len(target_names)), target_names, rotation='vertical')
            plt.yticks(range(len(target_names)), target_names)

########################################################################################################################

    def merge_with_cifar(self):
        """
        merges ocr data with cifar data and relabels in order to perform binary classification.
        This method is in charge of generating a unique data set merging 50000 images containing text (from the OCR data set)
        and 50000 images not containing text (from the CIFAR-10 data set).
        """ 
        cifar = Cifar('/home/francesco/Dropbox/DSR/OCR/cifar-config.py')
        
        text = OcrData('/home/francesco/Dropbox/DSR/OCR/text-config.py')
        
        text.ocr['target'][:] = 1

        total = 100000
        seed(10)
        k = sample(range(total), total)
        
        cifar_plus_text = {
                           'images': np.concatenate((cifar.cif['images'], text.ocr['images'][:50000]))[k],
                           'data': np.concatenate((cifar.cif['data'], text.ocr['data'][:50000]))[k],
                           'target': np.concatenate((cifar.cif['target'], text.ocr['target'][:50000]))[k]
                           }
 
        now = str(datetime.now()).replace(':','-')   
        fname_out = 'images-{}-{}-{}.pickle'.format(cifar_plus_text['target'].shape[0], self.img_size, now)
        full_name = os.path.join(self.folder_data,fname_out)
        with open(full_name, 'wb') as fout:
            cPickle.dump(cifar_plus_text, fout, -1)
            
        return cifar_plus_text
        


#################################################################################################################################
#################################################################################################################################
#################################################################################################################################

class HOGFeatures(BaseEstimator):
    """
    Defining class with fit/transform interface necessary for the Scikit-learn Pipeline.
    This class implements the Histogram Of Gradients, which is a technique commonly used to 
    extract relevant features from images (object detection for example) and then pass them to a classifier. 
    """
    def __init__(self, 
                 size,
                 orientations=8, 
                 pixels_per_cell=(10, 10),
                 cells_per_block=(1, 1)):
        
        super(HOGFeatures, self).__init__()
        self.orientations = orientations
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block
        self.size = size

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.reshape((X.shape[0], self.size[0], self.size[1]))
        result = []
        for image in X:
            #image = rgb2gray(image)
            features = hog(
                image,
                orientations=self.orientations,
                pixels_per_cell=self.pixels_per_cell,
                cells_per_block=self.cells_per_block,
                )
            result.append(features)
        return np.array(result)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    