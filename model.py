import os;
import sys;
import numpy as np;
import tensorflow as tf;
import matplotlib.pyplot as plt;

from tqdm import tqdm;
from skimage.io import imread,imsave;
from skimage.transform import resize;
from dataset import *;
from encoder import *;
from decoder import *;

class Model(object):
	def __init__(self,params,phase):
		self.params = params;
		self.phase = phase;
		self.vector_size = params.vector_size;
		self.num_filters = params.num_filters;
		self.batch_size = params.batch_size if phase=='train' else 1;
		self.image_shape = [params.image_shape,params.image_shape,3];
		self.save_dir = os.path.join(params.save_dir,self.params.solver+'/');
		self.global_step = tf.Variable(0,name='global_step',trainable=False);
		self.saver = tf.train.Saver(max_to_keep = 100);
		self.build();

	def build(self):
		print('Building the Model......');
		image_shape = self.image_shape;
		vector_size = self.vector_size;
		images = tf.placeholder(tf.float32,[self.batch_size]+image_shape);
		train = tf.placeholder(tf.bool);
		reuse = False if self.phase =='train' else True;
		encoder = Encoder(self.params,self.phase);
		decoder = Decoder(self.params,self.phase);
		vectors = encoder.run(images,train,reuse);
		mean = vectors[:,:vector_size];
		stddev = vectors[:,vector_size:];
		epsilon = tf.random_normal([self.batch_size,vector_size],0,1);
		samples = mean+epsilon*stddev;
		output = decoder.run(samples,train,reuse);
		vae_loss = 0.5*tf.reduce_sum(tf.square(mean)+tf.square(stddev)-tf.log(tf.square(stddev))-1.0)  ;
		rec_loss = tf.reduce_sum(tf.abs(images-output));
		loss = vae_loss+rec_loss;

		self.images = images;
		self.train = train;
		self.vae_loss = vae_loss;
		self.rec_loss = rec_loss;
		self.loss = loss;
		self.output = output;
		self.mean = mean;
		self.stddev = stddev;
		self.samples = samples;

		if self.params.solver == 'adam':
			solver = tf.train.AdamOptimizer(self.params.learning_rate);
		elif self.params.solver == 'momentum':
			solver = tf.train.MomentumOptimizer(self.params.learning_rate,self.params.momentum);
		elif self.params.solver == 'rmsprop':
			solver = tf.train.RMSPropOptimizer(self.params.learning_rate,self.params.weight_decay,self.params.momentum);
		else:
			solver = tf.train.GradientDescentOptimizer(self.params.learning_rate);

		tensorflow_variables = tf.trainable_variables();
		gradients,_ = tf.clip_by_global_norm(tf.gradients(self.loss,tensorflow_variables),3.0);
		optimizer = solver.apply_gradients(zip(gradients,tensorflow_variables),global_step=self.global_step);
		self.optimizer = optimizer;
		print('Model built......');

	def Train(self,sess,data):
		print('Training the model......');
		epochs = self.params.epochs;
		for epoch in tqdm(list(range(epochs)),desc='Epoch'):
			for i in tqdm(list(range(data.batches)),desc='Batch'): 
				files = data.next_batch();
				images = self.load_images(files);
				global_step,loss,vae_loss,rec_loss,_ = sess.run([self.global_step,self.loss,self.vae_loss,self.rec_loss,self.optimizer],feed_dict={self.images: images,self.train: True});
				print((' Loss = %f vae_loss =%f rec_loss = %f ' %(loss,vae_loss,rec_loss)));
				if(global_step%5000==0):
					output = sess.run(self.output,feed_dict={self.images:images, self.train:False});
					self.save_image(output[0],'train_sample_'+str(global_step));
				if(global_step%self.params.save_period==0):
					self.save(sess);
			data.reset();
		self.save(sess);
		print('Model Trained......');

	def Test(self,sess):
		for i in tqdm(list(range(self.params.test_samples)),desc='Batch'):
			sample = np.random.uniform(-1,1,size=(self.vector_size));
			output = sess.run(self.output,feed_dict={self.samples:sample, self.train:False});
			self.save_image(output,'test_sample_'+str(i+1));
		print('Testing completed......');

	def save(self,sess):
		print(('Saving model to %s...' % self.save_dir));
		self.saver.save(sess,self.save_dir,self.global_step);

	def load(self,sess):
		print('Loading model.....');
		checkpoint = tf.train.get_checkpoint_state(self.save_dir);
		if checkpoint is None:
			print("Error: No saved model found. Please train first...");
			sys.exit(0);
		self.saver.restore(sess, checkpoint.model_checkpoint_path);

	def load_images(self,files):
		images = [];
		image_shape = self.image_shape;
		for image_file in files:
			image = imread(image_file);
			image = resize(image,(image_shape[0],image_shape[1]));
			image = (image-127.5)/127.5;
			images.append(image);
		images = np.array(images,np.float32);
		return images;

	def save_image(self,output,name):
		output = (output*127.5)+127.5;
		if(self.phase=='train'):
			file_name = self.params.train_dir+name+'.png';
		else:
			file_name = self.params.test_dir+name+'.png';
		imsave(file_name,output);
		print('Saving the image %s...',file_name);