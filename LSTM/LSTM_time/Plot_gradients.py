from keras.layers.core import Dense, Activation, TimeDistributedDense
from keras.models import Sequential
from keras.layers import LSTM
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import theano
import theano.tensor as T
import sys

def load_data(file_path, n):
	x = []
	y = []
	for i in range(n):
		d = np.load(file_path + str(i) + '.npz')
		a = d['data']
		b = d['lab']
		x = x + a.tolist()
		y = y + b.tolist()
	z = []
	for i in range(len(y)):
		if y[i] == '1':
			z.append(x[i])
	return z

def main():
	max_len = 30
	s = 194
	model = Sequential()
	model.add(LSTM(128, return_sequences=True, input_shape=(max_len, s)))
	model.add(LSTM(64))
	model.add(Dense(2, activation='softmax'))
	model.compile(loss='binary_crossentropy', optimizer='adam')

	print('Loading buyers .....')
	buyers = load_data('../../../Train_data/Train/LSTM_time/LSTM_time', 7)

	print('Plotting gradients ........')
	model.load_weights('model/best_lstm_time_buy_w')
	get_output = theano.function([model.layers[0].input], model.layers[-1].output, allow_input_downcast=True)
	fx = theano.function([model.layers[0].input], T.jacobian(model.layers[-1].output.flatten(), model.layers[0].input), allow_input_downcast=True)

	#idx = np.random.permutation(len(buyers))
	#np.savez_compressed('Permuted_buyers_idx', idx=idx)
	a = np.load('../Permuted_buyers_idx.npz')
	idx = a['idx']
	for i in range(10):
		grad = fx([buyers[idx[i]]])
		fig, ax = plt.subplots()
		heatmap = ax.imshow(np.transpose(grad[0][0]), cmap=plt.cm.hot_r, interpolation='nearest')
		plt.axes().set_aspect(0.5)
		cbar = plt.colorbar(heatmap)
		plt.tight_layout()
		plt.title('Feature Excitation in Temporal View')
		plt.savefig('grad_plots/Gradients_buyers_time' + str(i) + '.png')
		plt.close()

if __name__ == '__main__':
	main()



