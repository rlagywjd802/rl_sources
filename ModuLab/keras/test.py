from keras.layers import Dense
from keras.models import Sequential

x_train = []
y_train = []

model = Sequential()
model.add(Dense(12, input_dim=5, activation='sigmoid'))
model.add(Dense(30, activation="sigmoid"))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='mse', optimizer='RMSProp')

model.fit(x_train, y_train, epochs=1, batch_size=32)
