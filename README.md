# Image_Captioning

Image Captioning using Deep learning models in Keras. The models were trained on Flickr_8k Dataset using Google Colab.

## Objectives:
1. Prepare photo and text data for training a deep learning model.
2. Design and train a deep learning model.
3. Evaluate the model 
4. Using this model generate caption for new pictures.

## Dataset:
After requesting the dataset from the author's website. I got these two files.
1. Flickr8k_Dataset: Contains 8092 photographs in JPEG format.
2. Flickr8k_text: Contains a number of files containing different sources of descriptions for the photographs.


The dataset has a pre-defined training dataset (6,000 images), development dataset (1,000 images), and test dataset (1,000 images).

## Comments:
1. From the result you can see it's not accurate because model was trained for 5 epochs due to limited GPU time Google colab. 
2. Using Checkpoints can make a difference but it will be updated.
3. Have to try for other techniques like different pretrained models for feature etraction and word to vec for token generation.
4. This is Implemented by understanding the tutorial of Jason Brownlee(Machine learning mastery).

# References:
1. [Flickr_8k Dataset](http://nlp.cs.illinois.edu/HockenmaierGroup/Framing_Image_Description/KCCA.html)
2. [Deep learning caption generation - Jason Brownlee](https://machinelearningmastery.com/develop-a-deep-learning-caption-generation-model-in-python/)
