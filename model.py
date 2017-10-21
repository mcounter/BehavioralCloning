import numpy as np
import DeepDataEngine as dd
import InitStorage

np.random.seed() # Randomize

InitStorage.createStorage(track_v1 = True, track_v2 = True) # Create data storage if not exists

# Initialize train data engine
data_train = dd.DeepDataEngine('train')
data_train.initStorage()

# Initialize validation data engine
data_valid = dd.DeepDataEngine('valid')
data_valid.initStorage()

in_shape, out_shape = data_train.getInOutShape() # Get model input and output size
train_steps, train_gen = data_train.getGenerator() # Load generator for training
valid_steps, valid_gen = data_valid.getGenerator() # Load generator for validation

print('Input shape: {}'.format(in_shape))
print('Train batches: {}'.format(train_steps))
print('Validation batches: {}'.format(valid_steps))

# Import Keras (just topmost namespace)
import keras

continue_learning = False # Set True to continue learning
continue_learning_new_set = False # Set True to train pre-trained model on new data set

if continue_learning:
    # If continue learning, load existing model
    model = keras.models.load_model('model.h5')

    if continue_learning_new_set:
        best_val_loss = 0
        isFirst = True
    else:
        # If continue learning on same data set, evaluate on validation set to know best loss value already achieved
        best_val_loss = float(model.evaluate_generator(valid_gen, valid_steps))
        print("VALIDATION LOSS: {}".format(best_val_loss))
        print()
        print()

        isFirst = False
else:
    # Model definition - Based on Keras Sequental model
    model = keras.models.Sequential()
    model.add(keras.layers.Lambda(lambda x: ((x / 255.0) - 0.5) * 2.0, input_shape = in_shape)) # 160 x 320, normalization
    model.add(keras.layers.Cropping2D(cropping = ((65, 25), (0, 0)))) # 70 x 320, cropping

    model.add(keras.layers.Conv2D(24, (5,5), padding='valid')) # 66 x 316, 1st convolutional layer
    #model.add(keras.layers.BatchNormalization(axis=-1, scale=False))
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.AvgPool2D(pool_size = (2, 2), strides = (2, 2))) # 33 x 158

    model.add(keras.layers.Conv2D(36, (5,5), padding='valid')) # 29 x 154, 2nd convolutional layer
    #model.add(keras.layers.BatchNormalization(axis=-1, scale=False))
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.AvgPool2D(pool_size = (2, 2), strides = (2, 2))) # 15 x 77

    model.add(keras.layers.Conv2D(48, (5,5), padding='valid')) # 11 x 73, 3rd convolutional layer
    #model.add(keras.layers.BatchNormalization(axis=-1, scale=False))
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.AvgPool2D(pool_size = (2, 2), strides = (2, 2))) # 6 x 37

    model.add(keras.layers.Conv2D(64, (3, 3), padding='valid')) # 4 x 35, 4th convolutional layer
    #model.add(keras.layers.BatchNormalization(axis=-1, scale=False))
    model.add(keras.layers.Activation('relu'))

    model.add(keras.layers.Conv2D(64, (3, 3), padding='valid')) # 2 x 33, 5th convolutional layer
    #model.add(keras.layers.BatchNormalization(axis=-1, scale=False))
    model.add(keras.layers.Activation('relu'))

    model.add(keras.layers.Flatten()), # 4224, Flattening to 1-dimension array

    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(200)) # 1st fully-connected layer
    model.add(keras.layers.Activation('relu'))

    model.add(keras.layers.Dropout(0.35))
    model.add(keras.layers.Dense(150)) # 2nd fully-connected layer
    model.add(keras.layers.Activation('relu'))

    model.add(keras.layers.Dropout(0.25))
    model.add(keras.layers.Dense(50)) # 3rd fully-connected layer
    model.add(keras.layers.Activation('relu'))

    model.add(keras.layers.Dropout(0.10))
    model.add(keras.layers.Dense(out_shape[0])) # Output layer

    model.compile(optimizer='adam', loss='mse') # Compile model to use Adam optimizer and MSE (mean squared error) as loss factor

    best_val_loss = 0
    isFirst = True

#for epoch in range(12): # Used to train with fixed number of epochs
epoch = 0
while True: # Infinit loop, must be manually terminated
    print("EPOCH: {}".format(epoch + 1)) # External epoch management is used to be possible save successfull models

    history = model.fit_generator(train_gen, train_steps, validation_data = valid_gen, validation_steps = valid_steps, epochs=1, verbose=1) # Train and validate model on full training set for single epoch

    val_loss = float(history.history['val_loss'][0]) # Get validation losses
    print("VALIDATION LOSS: {}".format(val_loss))
    
    if isFirst or (val_loss < best_val_loss):
        # Save model if loss factor is deсreased on validation set.
        best_val_loss = val_loss
        isFirst = False

        print("MODEL SAVING...")

        model.save('model.h5')

        print("MODEL IS SAVED!")
    else:
        print("MODEL SAVING IS SKIPPED.")

    print()
    print()

    epoch += 1