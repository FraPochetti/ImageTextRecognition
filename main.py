from userimageski import UserData

if __name__ == '__main__':
   
    ##### the following code includes all the steps to get from a raw image to a prediction.
    ##### the working code is the uncommented one. 
    ##### the two pickle models which are passed as argument to the select_text_among_candidates
    ##### and classify_text methods are obviously the result of a previously implemented pipeline.
    ##### just for the purpose of clearness below the code is provided. 
    ##### I want to emphasize that the commented code is the one necessary to get the models trained.
    
    # creates instance of class and loads image    
    user = UserData('lao.jpg')
    # plots preprocessed imae 
    user.plot_preprocessed_image()
    # detects objects in preprocessed image
    candidates = user.get_text_candidates()
    # plots objects detected
    user.plot_to_check(candidates, 'Total Objects Detected')
    # selects objects containing text
    maybe_text = user.select_text_among_candidates('/media/francesco/Francesco/CharacterProject/linearsvc-hog-fulltrain2-90.pickle')
    # plots objects after text detection
    user.plot_to_check(maybe_text, 'Objects Containing Text Detected')
    # classifies single characters
    classified = user.classify_text('/media/francesco/Francesco/CharacterProject/linearsvc-hog-fulltrain36-90.pickle')
    # plots letters after classification 
    user.plot_to_check(classified, 'Single Character Recognition')
    # plots the realigned text
    user.realign_text()
    
##########################################################################################################################    
## MACHINE LEARNING SECTION
##########################################################################################################################    
    #from data import OcrData
    #from cifar import Cifar
    #
    ####################################################################
    ## 1- GENERATE MODEL TO PREDICT WHETHER AN OBJECT CONTAINS TEXT OR NOT
    ####################################################################
    #
    # CREATES AN INSTANCE OF THE CLASS LOADING THE OCR DATA 
    #data = OcrData('/home/francesco/Dropbox/DSR/OCR/ocr-config.py')
    #
    # GENERATES A UNIQUE DATA SET MERGING NON-TEXT WITH TEXT IMAGES
    #data.merge_with_cifar()
    #
    # PERFORMS GRID SEARCH CROSS VALIDATION GETTING BEST MODEL OUT OF PASSED PARAMETERS
    #data.perform_grid_search_cv('linearsvc-hog')
    #
    # TAKES THE PARAMETERS LINKED TO BEST MODEL AND RE-TRAINS THE MODEL ON THE WHOLE TRAIN SET
    #data.generate_best_hog_model()
    #
    # TAKES THE JUST GENERATED MODEL AND EVALUATES IT ON TRAIN SET
    #data.evaluate('/media/francesco/Francesco/CharacterProject/linearsvc-hog-fulltrain2-90.pickle')


    ####################################################################
    ## 2- GENERATE MODEL TO CLASSIFY SINGLE CHARACTERS
    ####################################################################
    #
    # CREATES AN INSTANCE OF THE CLASS LOADING THE OCR DATA 
    #data = OcrData('/home/francesco/Dropbox/DSR/OCR/ocr-config.py')
    #
    # PERFORMS GRID SEARCH CROSS VALIDATION GETTING BEST MODEL OUT OF PASSED PARAMETERS
    #data.perform_grid_search_cv('linearsvc-hog')
    #
    # TAKES THE PARAMETERS LINKED TO BEST MODEL AND RE-TRAINS THE MODEL ON THE WHOLE TRAIN SET
    #data.generate_best_hog_model()
    #
    # TAKES THE JUST GENERATED MODEL AND EVALUATES IT ON TRAIN SET
    #data.evaluate('/media/francesco/Francesco/CharacterProject/linearsvc-hog-fulltrain36-90.pickle')
