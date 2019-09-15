import argparse
import os 
import numpy as np
import pandas as pd 
import tensorflow as tf 

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description="HAN's parameters")
	parser.add_argument('--train',type=bool, default=False, help='Training with GPU:0')
	parser.add_argument('--train_1',type=bool, default=False, help='Training with GPU:1')
	parser.add_argument("--test", type=bool, default=False, help="Evaluation phase")
	parser.add_argument('--data_path', type=str, default= "data", help="Path of data pickle file.")
	parser.add_argument('--index_path', type=str, default= "trend_index.npy", help="Trend index path")
	parser.add_argument('--vocab_path', type=str, default= "vocab", help="Vocab Path")
	parser.add_argument('--emb_path', type=str, default= "glove.6B/", help="Vocab Path")
	parser.add_argument('--csv_path', type=str, default= "enron_all.csv", help="Vocab Path")	
	parser.add_argument('--att_dim', type=int, default=100, help='Attention dimension') 
	parser.add_argument('--cell_dim', type=int, default=50) 
	parser.add_argument('--embedding_size', type=int, default=300) 
	parser.add_argument('--batch_size', type=int, default=128, help='Training batch size')
	parser.add_argument('--dropout_rate', type=float, default=0.9)
	parser.add_argument('--epoch', type=int, default=10, help='Training epcohs')
	parser.add_argument('--learning_rate', type=float, default=0.0005, help='Initial learning_rate')
	args = parser.parse_args()
	print(args)

	if args.train: 
		os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
		os.environ["CUDA_VISIBLE_DEVICES"]="0"

		import model as md
		tf.reset_default_graph()
		model = md.HAN_Classifier(args)
		model.train()

		"""
		Matrix = {}
		Matrix['Lambda'] = lamda_list
		Matrix['acc']= acc_list
		Matrix['mse_nn'] = mse_list
		Matrix['mse_lrr'] = mse_lrr_list
		Matrix['mse_krr'] = mse_krr_list
		final = pd.DataFrame(Matrix)
		final.to_csv("hybrid_cpgan_nonlinear.csv", index=False)

		"""

	elif args.train_1 :
		os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
		os.environ["CUDA_VISIBLE_DEVICES"]="1"

		import model as md
		tf.reset_default_graph()
		model = md.HAN_Classifier(args)
		model.train()
		"""
		Matrix = {}
		Matrix['Lambda'] = para
		Matrix['acc']= acc_list
		Matrix['mse_nn'] = mse_list
		Matrix['mse_lrr'] = mse_lrr_list
		Matrix['mse_krr'] = mse_krr_list
		final = pd.DataFrame(Matrix)
		final.to_csv("hybrid_cpgan.csv", index=False)
		"""


	else:
		os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
		os.environ["CUDA_VISIBLE_DEVICES"]="1"
		import model as md
		tf.reset_default_graph()
		model = md.HAN_Classifier(args)
		model.test()