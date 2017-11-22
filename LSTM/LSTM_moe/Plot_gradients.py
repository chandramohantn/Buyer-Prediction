from keras.layers.core import Dense, Activation
from keras.models import Model
from keras.layers import LSTM, Input, merge
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
	z = np.array(z)
	return z

def load_item_data(file_path, n, item_vec):
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
	z = np.array(z)
	z = Process_data(z, 30, 64, item_vec)
	return z

def Process_data(x, l, s, item_vec):
	a = [0] * s
	z = []
	for i in range(x.shape[0]):
		m = x[i].shape[0]
		y = []
		if (l-m) >= 0:
			for j in range(l-m):
				y.append(a)
			for j in x[i]:
				y.append(item_vec[j])
		else:
			for j in range(l):
				y.append(item_vec[x[i][j]])
		z.append(y)
	z = np.array(z)
	return z

def main():
	max_len = 30
	s = 15
	a = np.load('../LSTM_category/layer_weights.npz')
	weights = a['wts']
	print('Build expert 1...')
	input1 = Input(shape=(max_len, s))
	m11 = LSTM(128, return_sequences=True, weights=weights[0], trainable=False)(input1)
	m12 = LSTM(64, weights=weights[1], trainable=False)(m11)
	m13 = Dense(2, activation='softmax', weights=weights[2], trainable=False)(m12)

	s = 64
	a = np.load('../LSTM_item/layer_weights.npz')
	weights = a['wts']
	print('Build expert 2...')
	input2 = Input(shape=(max_len, s))
	m21 = LSTM(128, return_sequences=True, weights=weights[0], trainable=False)(input2)
	m22 = LSTM(64, weights=weights[1], trainable=False)(m21)
	m23 = Dense(2, activation='softmax', weights=weights[2], trainable=False)(m22)

	s = 194
	a = np.load('../LSTM_time/layer_weights.npz')
	weights = a['wts']
	print('Build expert 3...')
	input3 = Input(shape=(max_len, s))
	m31 = LSTM(128, return_sequences=True, weights=weights[0], trainable=False)(input3)
	m32 = LSTM(64, weights=weights[1], trainable=False)(m31)
	m33 = Dense(2, activation='softmax', weights=weights[2], trainable=False)(m32)

	stacker = merge([m13, m23, m33], mode='concat')

	s = 273
	print('Build Gating Network ....')
	input123 = Input(shape=(max_len, s))
	g1 = LSTM(128, return_sequences=True)(input123)
	g2 = LSTM(64)(g1)
	g3 = Dense(6, activation='softmax')(g2)

	g4 = merge([stacker, g3], mode='mul')
	g5 = Dense(2, activation='softmax')(g4)

	model = Model(input=[input1, input2, input3, input123], output=[g5])
	model.compile(loss='binary_crossentropy', optimizer='adam')

	a = np.load('../../../Full_data/Word2Vec_vectors.npz')
	item_vec = a['vectors']
	print('Loading buyers .....')
	c_buyers = load_data('../../../Train_data/Train/LSTM_category/LSTM_category', 7)
	i_buyers = load_item_data('../../../Train_data/Train/LSTM_item/LSTM_item', 7, item_vec)
	t_buyers = load_data('../../../Train_data/Train/LSTM_time/LSTM_time', 7)
	buyers = np.concatenate((c_buyers, i_buyers), axis=2)
	buyers = np.concatenate((buyers, t_buyers), axis=2)

	print('Plotting gradients ........')
	model.load_weights('model/best_lstm_moe_buy_w')

	a = np.load('../Permuted_buyers_idx.npz')
	idx = a['idx']
	for i in range(10):
		a = np.reshape(c_buyers[idx[i]], (1, 30, 15))
		b = np.reshape(i_buyers[idx[i]], (1, 30, 64))
		c = np.reshape(t_buyers[idx[i]], (1, 30, 194))
		d = np.reshape(buyers[idx[i]], (1, 30, 273))
		get_output = theano.function([model.layers[0].input, model.layers[1].input, model.layers[2].input, model.layers[3].input], model.layers[-1].output, allow_input_downcast=True)
		fx = theano.function([model.layers[0].input, model.layers[1].input, model.layers[2].input, model.layers[3].input], T.jacobian(model.layers[-1].output.flatten(), model.layers[0].input), allow_input_downcast=True)
		grad1 = fx(a, b, c, d)
		get_output = theano.function([model.layers[0].input, model.layers[1].input, model.layers[2].input, model.layers[3].input], model.layers[-1].output, allow_input_downcast=True)
		fx = theano.function([model.layers[0].input, model.layers[1].input, model.layers[2].input, model.layers[3].input], T.jacobian(model.layers[-1].output.flatten(), model.layers[1].input), allow_input_downcast=True)
		grad2 = fx(a, b, c, d)
		get_output = theano.function([model.layers[0].input, model.layers[1].input, model.layers[2].input, model.layers[3].input], model.layers[-1].output, allow_input_downcast=True)
		fx = theano.function([model.layers[0].input, model.layers[1].input, model.layers[2].input, model.layers[3].input], T.jacobian(model.layers[-1].output.flatten(), model.layers[2].input), allow_input_downcast=True)
		grad3 = fx(a, b, c, d)
		get_output = theano.function([model.layers[0].input, model.layers[1].input, model.layers[2].input, model.layers[3].input], model.layers[-1].output, allow_input_downcast=True)
		fx = theano.function([model.layers[0].input, model.layers[1].input, model.layers[2].input, model.layers[3].input], T.jacobian(model.layers[-1].output.flatten(), model.layers[3].input), allow_input_downcast=True)
		grad4 = fx(a, b, c, d)

		fig, ax = plt.subplots()
		h = ax.imshow(np.transpose(grad1[0][0]), cmap=plt.cm.hot_r, interpolation='nearest')
		c = plt.colorbar(h)
		plt.tight_layout()
		plt.savefig('grad_plots/Gradients_buyers_moe_cat' + str(i) + '.png')
		plt.close()

		fig, ax = plt.subplots()
		h = ax.imshow(np.transpose(grad2[0][0]), cmap=plt.cm.hot_r, interpolation='nearest')
		c = plt.colorbar(h)
		plt.tight_layout()
		plt.savefig('grad_plots/Gradients_buyers_moe_item' + str(i) + '.png')
		plt.close()

		fig, ax = plt.subplots()
		h = ax.imshow(np.transpose(grad3[0][0]), cmap=plt.cm.hot_r, interpolation='nearest')
		plt.axes().set_aspect(0.5)
		c = plt.colorbar(h)
		plt.tight_layout()
		plt.savefig('grad_plots/Gradients_buyers_moe_time' + str(i) + '.png')
		plt.close()
		
		fig, ax = plt.subplots()
		h = ax.imshow(np.transpose(grad4[0][0]), cmap=plt.cm.hot_r, interpolation='nearest')
		plt.axes().set_aspect(0.5)
		cbar = plt.colorbar(h)
		plt.tight_layout()
		plt.savefig('grad_plots/Gradients_buyers_moe_att' + str(i) + '.png')
		plt.close()


if __name__ == '__main__':
	main()
