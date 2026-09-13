## Deep Learning Trio (CNN,Self-Atttention mechanism,NLP) implemented using PyTorch

### Thought process and workflow:
1) CNN: Implementing Convolutional Neural Network using PyTorch taught me the core idea and mechansim behind this architecture which is:
Convolution layer-> ReLU activation -> Pooling and this repeats for certain number of times after which the output is flattened to a 1-D array and passed to a feed forward neural network which can later be used for classification / regression tasks.
##### Important lesson learnt:
This project also helped me understand the importance of data preprocessing where I encountered the channel dimension was missing in the input and should accordingly include it, converting the continuous outputs into discrete classes by multiplying and rounding off, converting the inputs and outputs into datasets and eventually dataloaders suitable for the model.

I got the following baseline results:
Training Accuracy: 71% , Validation Accuracy: 70% , Test Accuracy: 70%

2) Self attention mechanism: I learnt the core idea behind self attention mechanism which is:
   
Breaking sentence into words-> converting words to embeddings -> Calculating the query, key, value for each word -> Using the query and key values to calculation attention scores-> attention scores converted to attention weights-> using the attention weights to obtain the context vectors. 

3) NLP: I learn the core idea behind NLP(Natural language processing) by implementing a text classification model using distilbert which is:

Obtain encodings of your texts by using AutoEncoders -> Prepare a dataset consisting of the encodings and the outputs suitable for the model to process -> Training the model with the prepared dataset.

##### Important lesson learnt:
This project helped me to work with pre trained models, preparing custom dataset suitable for the model to process, using classification report and confusion matrix to interpret the results.

I got the following results:
Test set: 99.3% accuracy,0.97 precision and recall on the spam class.
   


   

