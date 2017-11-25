from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from read_train_tweets import


#model parameters
hiddenLayerActivation = 'relu'
classificationLoss = 'categorical_crossentropy'
classificationActivation = 'sigmoid' # for 2 classes, using sigmoic is fine
netOptimizer= 'adam'
np_epoch = 30
hiddenLaters = 3
embeddingSize = 300
minibatch_size =len(X_train)/25

#create model
model = Sequential()
model.add(Dense(embeddingSize,
                activation=hiddenLayerActivation,
                input_dim=embeddingSize))

for i in range(hiddenLaters):
    model.add(Dense(embeddingSize,activation=hiddenLayerActivation))
model.add(Dense(1, activation= classificationActivation))

#compile
model.compile(loss = classificationLoss, optimizer= netOptimizer,
              metrics=['accuracy'])

#fit the model

model.fit(X_train,Y_train,validation_data=(X_train, Y_train),
          nb_epoch=np_epoch,batch_size = minibatch_size,
          verbose=True)
