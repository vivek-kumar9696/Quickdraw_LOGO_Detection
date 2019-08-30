# LOGO PREDICTION THROUGH DOODLES

This project predicts logos of brands using doodles as input. 
It uses a Webcam to capture video and tracks path of specific coloured objects to draw the doodle on the webcam output window.


Logos used:
* Adidas 
* Apple 
* BMW 
* Citroen 
* FedEx 
* HP 
* Mcdonald's 
* Nike 
* Pepsi 
* Puma

## NEURAL NETWORK MODEL USED

MobileNet CNN architecture was used. The last 18 layers were retrained on the dataset to benefit from transfer learning

## DATASET USED

Flickr-27 dataset was used. To generate more data, the images from flickr-27 were converted to binary images and data augmentation was also applied.

## FILE DESCRIPTIONS

1. data_conv.py: python programme to convert RGB training images to binary images for                      better feature extraction.
2. data_structuring.py: python programme to structure data processed by data_conv.py to                           folder respective of the brands so as to facilitate data                                  augmentation and training.(uses text files extracted from                                 flickr27 datset)
3. mob_logo_model.h5: trained model
4. screen_reader.py: Main application programme. Run this programme to launch application
5. train_logo.ipynb: Notebook for training of model.

![Logo detector through doodle's Demo (Mcdonald's, Nike)](logo_recording(1).gif)
![Logo detector through doodle's Demo (Adidas, Puma)](logo_recording(2).gif)
