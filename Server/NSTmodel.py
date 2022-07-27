import numpy as np
import PIL.Image
import time
import functools

import os
import tensorflow as tf
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)


os.environ['TFHUB_MODEL_LOAD_FORMAT'] = 'COMPRESSED'


#Defining StyleContentModel which gets the content and style representaions
class StyleContentModel(tf.keras.models.Model):
  def __init__(self, style_layers, content_layers,vgg, vgg_layers, gram_matrix):
    super().__init__()
    self.vgg = vgg_layers(style_layers + content_layers,vgg)
    self.style_layers = style_layers
    self.content_layers = content_layers
    self.num_style_layers = len(style_layers)
    self.vgg.trainable = False
    self.gram_matrix = gram_matrix

  def call(self, inputs):
      inputs = inputs*255.0
      preprocessed_input = tf.keras.applications.vgg19.preprocess_input(inputs)
      outputs = self.vgg(preprocessed_input)
      style_outputs, content_outputs = (
          outputs[:self.num_style_layers],
          outputs[self.num_style_layers:]
      )

      style_outputs = [self.gram_matrix(style_output) for style_output in style_outputs]
      content_dict = {content_name:value 
                    for content_name, value 
                    in zip(self.content_layers, content_outputs)}

      style_dict = {style_name:value
                  for style_name, value
                  in zip(self.style_layers, style_outputs)}

      return {'content':content_dict, 'style':style_dict}


#Defining Neural Style Transfer(NST)
class NST:
  def __init__(self):
    #loading VGG CNN
    self.vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
    #Style and Content Image
    self.content_image = None
    self.style_image = None
    #defining style and content layers
    self.content_layers = ['block5_conv2']
    self.style_layers = ['block1_conv1',
                'block2_conv1',
                'block3_conv1',
                'block4_conv1',
                'block5_conv1']
    self.num_content_layers = len(self.content_layers)
    self.num_style_layers = len(self.style_layers)
    #setting optimizer
    self.opt = tf.optimizers.Adam(learning_rate = 0.02, beta_1 = 0.99, epsilon = 1e-1)
    #inintializing hyper-parameters
    self.alpha = 1e-2
    self.beta = 1e4
    self.epochs = 5
    self.steps_per_epoch = 50

 
  
  #Loading Style Image
  def loadStyleImage(self, styleImage):
    self.style_image = styleImage
  #Loading Content Image
  def loadContentImage(self, contentImage):
    self.content_image = contentImage

  #Loading image for processing
  def load_img(self, img):
    max_dim = 512

    img = tf.image.convert_image_dtype(img, tf.float32) 

    shape = tf.cast(tf.shape(img)[:-1], tf.float32)

    long_dim = max(shape)
    
    scale = max_dim / long_dim

    new_shape = tf.cast(shape * scale, tf.int32)
    print(new_shape)

    img = tf.image.resize(img, new_shape)

    img = img[tf.newaxis, :]

    return img
  #setting the loaded images
  def setLoadedImages(self):
    self.content_image = self.load_img(self.content_image)
    self.style_image = self.load_img(self.style_image)

  #convert a tensor to image  
  def tensor_to_image(self, tensor):
    tensor = tensor*255
    tensor = np.array(tensor)
    if np.ndim(tensor) > 3:
       assert tensor.shape[0] == 1
       tensor = tensor[0]
    return PIL.Image.fromarray(tensor.astype("uint8"))
  
  #returns a model in which layers corresponds to some layers of VGG model
  def vgg_layers(self, layer_names, vgg):
    vgg.trainable = False

    outputs = [vgg.get_layer(name).output for name in layer_names]

    model = tf.keras.Model([vgg.input], outputs)

    return model
  
  #returns the gram matrix or the unnormalized covarience
  def gram_matrix(self, input_tensor):
    G_aux= tf.linalg.einsum('xijc,xijd->xcd', input_tensor, input_tensor)
    input_shape = tf.shape(input_tensor)
    return G_aux/(tf.cast(input_shape[1]*input_shape[2],tf.float32))
  
  #extracting the style and content representaions using StyleContentModel
  def Extractor(self):
    self.extractor = StyleContentModel(self.style_layers, self.content_layers, self.vgg, self.vgg_layers, self.gram_matrix)

    self.content_targets = self.extractor(self.content_image)['content']
    self.style_targets = self.extractor(self.style_image)['style']
    self.image = tf.Variable(self.content_image)
  
  #clipping the values with lower bound zero and upper bount one
  def clip_0_1(self,image):
    return tf.clip_by_value(image, clip_value_min = 0.0, clip_value_max = 1.0)
  
  #defining main loss
  def main_loss(self,outputs):
   style_outputs = outputs['style']
   content_outputs = outputs['content']
   style_loss = (self.alpha/self.num_style_layers)*(tf.add_n([tf.reduce_mean((style_outputs[name] - self.style_targets[name])**2) 
         for name in style_outputs.keys()]))
   content_loss = (self.beta/self.num_content_layers)*(tf.add_n([tf.reduce_mean((content_outputs[name]-self.content_targets[name])**2) for name in content_outputs.keys()]))
   return style_loss + content_loss

  # defining training step
  def train_step(self, image):
    with tf.GradientTape() as tape:
     outputs = self.extractor(image)
     loss = self.main_loss(outputs)
    grad = tape.gradient(loss, image)
    self.opt.apply_gradients([(grad, image)])
    image.assign(self.clip_0_1(image))
  
  #returns image after processing 
  def masterCall(self):
    for i in range(self.epochs):
     for j in range(self.steps_per_epoch):
       self.train_step(self.image)
    return self.tensor_to_image(self.image)



