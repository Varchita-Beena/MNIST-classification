#############train####################
'''
from PIL import Image
from keras import backend as K
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import Activation, Flatten, Dense, Dropout
from keras.layers.normalization import BatchNormalization
from keras import optimizers
from keras.utils import to_categorical
#from keras.callbacks import EarlyStopping



train_images = 49000
folder = 'Images/train/'
ext = '.png'
image_list = []
for i in range(train_images):
	pos = folder+str(i)+ext
	image = Image.open(pos).convert('L')
	data = np.asarray(image)
	data = data/255.0
	image_list.append(data)
trainX = np.array(image_list)
trainX = trainX.reshape((49000,28,28,1))

labels = pd.read_csv('train.csv')
lable_list = labels['label'].values.tolist()
labelsX = to_categorical(lable_list,10)




model = Sequential()
model.add(Conv2D(32, kernel_size = (3,3), activation = 'relu', input_shape = (28,28,1), data_format = 'channels_last'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, kernel_size = (3,3), activation = 'relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size = (2,2)))
#model.add(Dropout(0.25))

model.add(Conv2D(128, kernel_size = (3,3), activation = 'relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Dropout(0.25))


model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(10, activation='softmax'))
model.compile(optimizer='adam', loss = 'categorical_crossentropy',metrics = ['accuracy'])
#es = EarlyStopping(monitor='loss', mode='min', verbose=1)
#model.fit(trainX, trainy, validation_data=(testX, testy), epochs=4000, verbose=0, callbacks=[es])
model.fit(trainX,labelsX,batch_size = 128, epochs = 20)#, validation_data = (trainX,labelsX))
model.save('train.h5')
#print("Accuracy on test data is : %0.2f"%accuracy(trainX,labelsX,model))

'''
#############test############
from keras.models import load_model
import numpy as np
import pandas as pd
from PIL import Image
import csv

model = load_model('train.h5',compile = True)

test = pd.read_csv('Test_fCbTej3_0j1gHmj.csv')
test_list = test['filename'].values.tolist()
folder = 'Images/test/'
ext = '.png'
test_image_list = []
start_num = 49000
total_images = 21000
end_num = start_num+total_images
for i in range(start_num, end_num):
	pos = folder+str(i)+ext
	print(pos)
	image = Image.open(pos).convert('L')
	data = np.asarray(image)
	data = data/255.0
	test_image_list.append(data)
testX = np.array(test_image_list)
testX = testX.reshape((21000,28,28,1))

predictions = model.predict(testX)
classes = np.argmax(predictions, axis = 1)

classes_list = classes.tolist()
class_index = 0
with open('test.csv','w',newline='') as file:
	for i in range(start_num, end_num):
		pos = folder+str(i)+ext
		writer = csv.writer(file)
		writer.writerow([pos, classes_list[class_index]])
		class_index = class_index + 1
	




