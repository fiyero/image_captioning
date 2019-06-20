# Use Pytorch to create an image captioning model with CNN and seq2seq LSTM
## https://medium.com/@patrickhk/use-pytorch-to-create-an-image-captioning-model-with-cnn-and-seq2seq-lstm-cff1f3ba9583

### Dataset
The COCO dataset is used. I follow udacity and use the year 2014 data, you can download and use more updated COCO dataset and may achieve better result.</br>

### Initialize the COCO API
We can follow the official github repo to learn how to use the COCO API. Input the path of the annotations file then we can visualize the image from dataset.<br/>

![p1](https://cdn-images-1.medium.com/max/800/1*ZVbauT2XYcSjG850a5WS2Q.png)<br/>
![p2](https://cdn-images-1.medium.com/max/800/1*F31mJGdooq9e8xDESuZh5A.png)<br/>
### Preprocess image
Apply some standard preprocessing steps, such as resizing, random cropping, normalizing and flipping…etc<br/>

![p3](https://cdn-images-1.medium.com/max/800/1*Wlg9kOHMaJ_P4KSlxxI6Tg.png)<br/>
### Preprocess captions
The original captions were in text form but our model cant read English directly, we have to convert text into integer first. We can use NLTK (or other NLP method) to do the tokenization and build a mapping dictionary to give each unique text token an integer. We can also apply some threshold to filter away very rare or common word by counting their occurrence in corpus.

For example, a raw text sentence “ I am a boy” will be tokenized into [<start>, ‘i’, ‘am’, ‘a’, ‘boy’,<end>] and eventually become [0, 3, 98, 754, 3, 1].<br/>
  
### SubsetRandomSampler
The length of caption on images are varying but our model require a fixed length input per batch. Usually we will use the padding function in pytorch to pad or truncate to make them same length within mini batch.

Or we can use the SubsetRandomSampler in pytorch to samples elements randomly from a given list of indices. We first generate a random number ≤max length of captions, eg 13. Then we use np.where to get all indices of captions having length =13.<br/>

### Build the encoder
For the image extraction part I didn’t build one from scratch because this is not a very strict image classification task. Instead I apply pre-trained resnet50 because it will save me a lot of time. Remember to remove the last FC layer as we are not doing image classification, we just need to extract feature and connect the feature vector with LSTM in decoder. Remember to freeze the parameter of the resnet50 otherwise you will destroy the trained-weight.<br/>

![p4](https://cdn-images-1.medium.com/max/800/1*vye6KMbiWJBNlgVdc9OxzQ.png)<br/>
### Build the decoder
We have already converted text sentence into integer token, now we add a word embedding layer to increase the representation ability of our model. Don’t forget to concatenate the feature vector(image) and our embedding matrix(captions) to pass into LSTM<br/>


![p5](https://cdn-images-1.medium.com/max/800/1*5bgFMd1RkfJohL1w0GKZSA.png)<br/>
### Hyper parameter
I always use Adam optimizer, my first choice.

Batch size=512, this is the max my GPU can afford.

Learning rate 0.01 to 0.001, depend on number of epoch. I didn’t apply a decaying/annealing learning rate scheduled because I prefer fine tune manually. When I observe the loss is getting flatted, I lower the lr.<br/>
### Result
The actual epoch I trained was over 15+ but I didnt record all the loss data. Since this project is for my own interest only, I don’t plan to re-run again just to get all the tracking. (Notice it is a good practice to keep all the records about the epoch, lr and loss data…etc)<br/>
![p6](https://cdn-images-1.medium.com/max/800/1*BQTGK9qkBG1h8lHWjtneSg.png)<br/>

Don’t forget to apply the same image preprocessing steps to the testing image set. The tesing image will go through the same preprocessing steps and feed into the model to output a token, then we map the integer token with the word2idx dictionary to get back the text token. This token also become the input of our model to predict the next token. It loops until our model read the <stop> token.<br/>
  
![p7](https://cdn-images-1.medium.com/max/800/1*zt9fwzy5Jlvuh9HbQa5-6w.png)<br/>
![p8](https://cdn-images-1.medium.com/max/800/1*1QGSUH9DX_NV9On01fhEFw.png)<br/>
-------------------------------------------------------------------------------------------------------------------------------------
### More about me
[[:pencil:My Medium]](https://medium.com/@patrickhk)<br/>
[[:house_with_garden:My Website]](https://www.fiyeroleung.com/)<br/>
[[:space_invader:	My Github]](https://github.com/fiyero)<br/>
