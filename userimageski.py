import numpy as np
from skimage.io import imread
from skimage.filter import threshold_otsu
from skimage.transform import resize
import cPickle
from matplotlib import pyplot as plt
from skimage.morphology import closing, square
from skimage.measure import regionprops
from skimage import restoration
from skimage import measure
from skimage.color import label2rgb
import matplotlib.patches as mpatches

class UserData():
    """
    class in charge of dealing with User Image input.
    the methods provided are finalized to process the image and return 
    the text contained in it.
    """
    
    def __init__(self, image_file):
        """
        reads the image provided by the user as grey scale and preprocesses it.
        """
        self.image = imread(image_file, as_grey=True)
        self.preprocess_image()
    
#############################################################################################################

    def preprocess_image(self):
        """
        Denoises and increases contrast. 
        """
        image = restoration.denoise_tv_chambolle(self.image, weight=0.1)
        thresh = threshold_otsu(image)
        self.bw = closing(image > thresh, square(2))
        self.cleared = self.bw.copy()
        return self.cleared 
    
############################################################################################################

    def get_text_candidates(self):
        """
        identifies objects in the image. Gets contours, draws rectangles around them
        and saves the rectangles as individual images.
        """
        label_image = measure.label(self.cleared)   
        borders = np.logical_xor(self.bw, self.cleared)
        label_image[borders] = -1
        
        
        coordinates = []
        i=0
        
        for region in regionprops(label_image):
            if region.area > 10:
                minr, minc, maxr, maxc = region.bbox
                margin = 3
                minr, minc, maxr, maxc = minr-margin, minc-margin, maxr+margin, maxc+margin
                roi = self.image[minr:maxr, minc:maxc]
                if roi.shape[0]*roi.shape[1] == 0:
                    continue
                else:
                    if i==0:
                        samples = resize(roi, (20,20))
                        coordinates.append(region.bbox)
                        i+=1
                    elif i==1:
                        roismall = resize(roi, (20,20))
                        samples = np.concatenate((samples[None,:,:], roismall[None,:,:]), axis=0)
                        coordinates.append(region.bbox)
                        i+=1
                    else:
                        roismall = resize(roi, (20,20))
                        samples = np.concatenate((samples[:,:,:], roismall[None,:,:]), axis=0)
                        coordinates.append(region.bbox)
        
        self.candidates = {
                    'fullscale': samples,          
                    'flattened': samples.reshape((samples.shape[0], -1)),
                    'coordinates': np.array(coordinates)
                    }
        
        print 'Images After Contour Detection'
        print 'Fullscale: ', self.candidates['fullscale'].shape
        print 'Flattened: ', self.candidates['flattened'].shape
        print 'Contour Coordinates: ', self.candidates['coordinates'].shape
        print '============================================================'
        
        return self.candidates 
    
##########################################################################################################################

    def select_text_among_candidates(self, model_filename2):
        """
        it takes as argument a pickle model and predicts whether the detected objects
        contain text or not. 
        """
        with open(model_filename2, 'rb') as fin:
            model = cPickle.load(fin)
            
        is_text = model.predict(self.candidates['flattened'])
        
        self.to_be_classified = {
                                 'fullscale': self.candidates['fullscale'][is_text == '1'],
                                 'flattened': self.candidates['flattened'][is_text == '1'],
                                 'coordinates': self.candidates['coordinates'][is_text == '1']
                                 }

        print 'Images After Text Detection'
        print 'Fullscale: ', self.to_be_classified['fullscale'].shape
        print 'Flattened: ', self.to_be_classified['flattened'].shape
        print 'Contour Coordinates: ', self.to_be_classified['coordinates'].shape
        print 'Rectangles Identified as NOT containing Text '+str(self.candidates['coordinates'].shape[0]-self.to_be_classified['coordinates'].shape[0])+' out of '+str(self.candidates['coordinates'].shape[0])
        print '============================================================'
        
               
        return self.to_be_classified
    
####################################################################################################

    def classify_text(self, model_filename36):
        """
        it takes as argument a pickle model and predicts character
        """
        with open(model_filename36, 'rb') as fin:
            model = cPickle.load(fin)
            
        which_text = model.predict(self.to_be_classified['flattened'])
        
        self.which_text = {
                                 'fullscale': self.to_be_classified['fullscale'],
                                 'flattened': self.to_be_classified['flattened'],
                                 'coordinates': self.to_be_classified['coordinates'],
                                 'predicted_char': which_text
                                 }     

        return self.which_text

############################################################################################################################

    def realign_text(self):
        """
        processes the classified characters and reorders them in a 2D space 
        generating a matplotlib image. 
        """
        max_maxrow = max(self.which_text['coordinates'][:,2])
        min_mincol = min(self.which_text['coordinates'][:,1])
        subtract_max = np.array([max_maxrow, min_mincol, max_maxrow, min_mincol]) 
        flip_coord = np.array([-1, 1, -1, 1])
        
        coordinates = (self.which_text['coordinates'] - subtract_max) * flip_coord
        
        ymax = max(coordinates[:,0])
        xmax = max(coordinates[:,3])
        
        coordinates = [list(coordinate) for coordinate in coordinates]
        predicted = [list(letter) for letter in self.which_text['predicted_char']]
        to_realign = zip(coordinates, predicted)
        
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for char in to_realign:
            ax.text(char[0][1], char[0][2], char[1][0], size=16)
        ax.set_ylim(-10,ymax+10)
        ax.set_xlim(-10,xmax+10)  
       
       
        #### CODE ADDED MANUALLY TO UNDERLINE CORRECT TEXT OUTPUT. JUST FOR THE PURPOSE OF
        #### CLEARNESS AND ONLY APPLICABLE TO THE EXAMPLE IMAGE lao.jpg     
        ############################################################################################
        ############################################################################################
        ax.broken_barh([(80, 80), (175, 150), (370, 40)] , (185, 35), facecolors='blue', alpha = 0.5)
        ax.broken_barh([(-5, 205), (225, 120), (355, 160)] , (120, 35), facecolors='blue', alpha = 0.5)
        ax.broken_barh([(35, 95), (150, 20), (190, 135), (370, 65)] , (50, 40), facecolors='blue', alpha = 0.5)
        ax.broken_barh([(440, 95)] , (-2, 22), facecolors='blue', alpha = 0.5)
        ############################################################################################
        ############################################################################################
        
        plt.show()
 
############################################################################################################################

    def plot_to_check(self, what_to_plot, title):
        """
        plots images at several steps of the whole pipeline, just to check output.
        what_to_plot is the name of the dictionary to be plotted
        """
        n_images = what_to_plot['fullscale'].shape[0]
        
        fig = plt.figure(figsize=(12, 12))

        if n_images <=100:
            if n_images < 100:
                total = range(n_images)
            elif n_images == 100:
                total = range(100)
           
            for i in total:
                ax = fig.add_subplot(10, 10, i + 1, xticks=[], yticks=[])
                ax.imshow(what_to_plot['fullscale'][i], cmap="Greys_r")  
                if 'predicted_char' in what_to_plot:
                    ax.text(-6, 8, str(what_to_plot['predicted_char'][i]), fontsize=22, color='red')
            plt.suptitle(title, fontsize=20)
            plt.show()  
        else:
            total = list(np.random.choice(n_images, 100)) 
            for i, j in enumerate(total):
                ax = fig.add_subplot(10, 10, i + 1, xticks=[], yticks=[])
                ax.imshow(what_to_plot['fullscale'][j], cmap="Greys_r")  
                if 'predicted_char' in what_to_plot:
                    ax.text(-6, 8, str(what_to_plot['predicted_char'][j]), fontsize=22, color='red')
            plt.suptitle(title, fontsize=20)
            plt.show()   
        
############################################################################################################################
  
    def plot_preprocessed_image(self):
        """
        plots pre-processed image. The plotted image is the same as obtained at the end
        of the get_text_candidates method.
        """
        image = restoration.denoise_tv_chambolle(self.image, weight=0.1)
        thresh = threshold_otsu(image)
        bw = closing(image > thresh, square(2))
        cleared = bw.copy()
        
        label_image = measure.label(cleared)
        borders = np.logical_xor(bw, cleared)
       
        label_image[borders] = -1
        image_label_overlay = label2rgb(label_image, image=image)
        
        fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(12, 12))
        ax.imshow(image_label_overlay)
        
        for region in regionprops(label_image):
            if region.area < 10:
                continue
        
            minr, minc, maxr, maxc = region.bbox
            rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                      fill=False, edgecolor='red', linewidth=2)
            ax.add_patch(rect)
        
        plt.show()       
        
            
        
        
        
        
    