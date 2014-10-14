import os
import re
import sys
import cPickle
import numpy as np
from skimage.io import imread
from datetime import datetime
from skimage.transform import resize
from matplotlib import pyplot as plt

class Cifar():
    """
    this class deals with images not containing text.
    An instance of this class is created into the merge_with_cifar method of OCR class
    in order merge cifar data with text data.
    """
    
    def __init__(self, config):
        """
        initialize the instance picking parameters from a config.py file.
        """
        self.config = self._load_config(config)
        self.img_size = self.config['img_size']
        self.folder = self.config['folder']
        self.verbose = self.config['verbose']
        self.from_pickle = self.config['from_pickle']
        self.pickle_data = self.config['pickle_data']
        self.load()

########################################################################################################################
        
    def _load_config(self, filename):
        """
        Reads a config.py file and returns the dictionary with all parameters
        """
        return eval(open(filename).read())        

#########################################################################################################################

    def load(self):
        """
        loads cifar data into python dictionary.
        """

        if self.from_pickle:
            try:
                with open(os.path.join(self.folder,self.pickle_data), 'rb') as fin:
                    self.cif = cPickle.load(fin)
                    if self.verbose:
                        print 'Loaded {} images each {} pixels'.format(self.cif['images'].shape[0], self.img_size)
                    return self.cif
                
            except IOError:
                print 'You have not provided a .pickle file to load data from!'
                sys.exit(0)
        else:
    
            filenames = [os.path.join(self.folder,f) for f in os.listdir(self.folder) if re.match(r'[0-9]+.*\.png', f)]
            n_images = len(filenames)
            target = [0]*n_images
            images = np.zeros((n_images,) + self.img_size)
        
            for index, filename in enumerate(filenames):
                image = imread(filename, as_grey=True)
                image = resize(image, self.img_size)
                images[index] = image
        
            if self.verbose:
                print "Loaded {} images.".format(n_images)
        
            self.cif = {
                'images': images,
                'data': images.reshape((images.shape[0], -1)),# / 255.0,
                'target': np.array(target),
                }
            
            now = str(datetime.now()).replace(':','-')   
            fname_out = 'images-{}-{}-{}.pickle'.format(len(target), self.img_size, now)
            full_name = os.path.join(self.folder,fname_out)
            with open(full_name, 'wb') as fout:
                cPickle.dump(self.cif, fout, -1)
                        
            return self.cif

########################################################################################################################

    def plot_some(self):
        """
        plots 100 images with relative label randomly picked from loaded data.
        """
        n_images = self.cif['images'].shape[0]
    
        fig = plt.figure(figsize=(12, 12))
        fig.subplots_adjust(
            left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)
    
        for i, j in enumerate(np.random.choice(n_images, 100)):
            ax = fig.add_subplot(10, 10, i + 1, xticks=[], yticks=[])
            ax.imshow(self.cif['images'][j], cmap="Greys_r")
            ax.text(2, 7, str(self.cif['target'][j]), fontsize=25, color='red')
        plt.show() 
