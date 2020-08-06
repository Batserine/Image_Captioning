# Image_Captioning

Image Captioning using Deep learning models in Keras. The models were trained on Flickr_8k Dataset using Google Colab.

## Objectives:
1. Prepare photo and text data for training a deep learning model.
2. Design and train a deep learning model.
3. Evaluate the model 
4. Using this model generate caption for new pictures.

## Using word to index procedure
### Steps:

1. Data collection
2. Understanding the data
3. Data Cleaning
4. Loading the training set
5. Data Preprocessing — Images
6. Data Preprocessing — Captions
7. Data Preparation using Generator Function
8. Word Embeddings
9. Model Architecture
10. Inference

## Dataset:
After requesting the dataset from the author's website. I got these two files.
1. Flickr8k_Dataset: Contains 8092 photographs in JPEG format.
2. Flickr8k_text: Contains a number of files containing different sources of descriptions for the photographs.


The dataset has a pre-defined training dataset (6,000 images), development dataset (1,000 images), and test dataset (1,000 images).

## Deployment:
Built a basic web app using Flask. It takes an image as input and generates a caption to it.
![Web app](https://github.com/Batserine/Image_Captioning/blob/master/IC_Deploying/output.png)


## Comments:
1. From the result you can see it's not accurate because model was trained for 5 epochs due to limited GPU time Google colab. 
2. Using Checkpoints can make a difference but it will be updated.
3. Have to try for other techniques like different pretrained models for feature etraction and word to vec for token generation.
4. This is Implemented by understanding the tutorial of Jason Brownlee(Machine learning mastery).

# References:
1. [Flickr_8k Dataset](http://nlp.cs.illinois.edu/HockenmaierGroup/Framing_Image_Description/KCCA.html)
2. [Deep learning caption generation - Jason Brownlee](https://machinelearningmastery.com/develop-a-deep-learning-caption-generation-model-in-python/)
