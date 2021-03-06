# speech-recognition
# 引入函式
	from preprocessing import *
	import keras
	from keras.models import Sequential
	from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
	from keras.utils import to_categorical

# 載入 data 資料夾的訓練資料，並自動分為『訓練組』及『測試組』(在預處理已把資料轉換為MFCC格式)
	X_train, X_test, y_train, y_test = get_train_test()
	X_train = X_train.reshape(X_train.shape[0], 50,32, 1)
	X_test = X_test.reshape(X_test.shape[0], 50, 32, 1)

# 類別變數轉為one-hot encoding
	y_train_hot = to_categorical(y_train)
	y_test_hot = to_categorical(y_test)
	print("X_train.shape=", X_train.shape)

# 建立簡單的線性執行的模型
	model = Sequential()

	model.add(Conv2D(32, kernel_size=(2, 2), activation='relu', input_shape=(50,32, 1)))
	model.add(Conv2D(32,3,3, activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Conv2D(32,3,3, activation='relu'))
	model.add(Conv2D(32,3,3, activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Conv2D(32,3,3, activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
# Dropout層隨機斷開輸入神經元，用於防止過度擬合，斷開比例:0.25
	model.add(Dropout(0.25))
# Flatten層把多維的輸入一維化，常用在從卷積層到全連接層的過渡。
	model.add(Flatten())
# 全連接層: 128個output
	model.add(Dense(128, activation='relu'))
	model.add(Dropout(0.25))
	model.add(Dense(64, activation='relu'))
	model.add(Dropout(0.25))
# Add output layer
	model.add(Dense(31, activation='softmax'))
# 編譯: 選擇損失函數、優化方法及成效衡量方式
	model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

# 進行訓練, 訓練過程會存在 train_history 變數中
	model.fit(X_train, y_train_hot, batch_size=64, epochs=50, verbose=1, validation_data=(X_test, y_test_hot))


	X_train = X_train.reshape(X_train.shape[0], 50, 32, 1)
	X_test = X_test.reshape(X_test.shape[0], 50, 32, 1)
	score = model.evaluate(X_test, y_test_hot, verbose=1)

# 模型存檔
	from keras.models import load_model
	model.save('ASR.h5')  # creates a HDF5 file 'model.h5'
# 分析
	在此MFCC轉換中第二維度只能取到32個向量，似乎取越多的向量出來精準度就能夠提升
