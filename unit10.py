#케라스 호출
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#라이브러리 호출
import numpy as np
import tensorflow as tf

#데이터 호출
dataSet = np.loadtxt("C:/Users/LG1/Desktop/SWU/20SWU/프로젝트종합설계1/AHyunDeepLearning/dataset/ThoraricSurgery.csv", delimiter = ",")

#환자 정보=X, 수술 결과=Y
X = dataSet[:,0:17]
Y = dataSet[:,17]

#딥러닝 모델 설정
model = Sequential()
model.add(Dense(30, input_dim = 17, activation = 'relu'))
model.add(Dense(1, activation = 'sigmoid'))

#딥러닝 실행
model.compile(loss = 'mean_squared_error', optimizer = 'adam', metrics = ['accuracy'])
model.fit(X, Y, epochs = 100, batch_size = 10)