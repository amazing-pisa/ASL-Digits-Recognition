import runtime
import tensorflow as tf
from tensorflow.keras import layers as L


@tf.keras.utils.register_keras_serializable(package='SignRecognition')
class AddCLSToken(L.Layer):
    def __init__(self, hidden_dim, **kwargs):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
    def build(self, input_shape):
        self.cls_token = self.add_weight(shape=(1,1,self.hidden_dim), initializer='zeros', name='cls_token', trainable=True)
    def call(self, inputs):
        return tf.concat([tf.tile(self.cls_token,[tf.shape(inputs)[0],1,1]), inputs], axis=1)
    def get_config(self):
        return {**super().get_config(), 'hidden_dim':self.hidden_dim}


@tf.keras.utils.register_keras_serializable(package='SignRecognition')
class PositionalEmbedding(L.Layer):
    def __init__(self, seq_len, hidden_dim, **kwargs):
        super().__init__(**kwargs)
        self.seq_len, self.hidden_dim = seq_len, hidden_dim
    def build(self, input_shape):
        self.pos_emb = self.add_weight(shape=(1,self.seq_len,self.hidden_dim), initializer='random_normal', name='pos_emb', trainable=True)
    def call(self, inputs):
        return inputs+self.pos_emb
    def get_config(self):
        return {**super().get_config(), 'seq_len':self.seq_len, 'hidden_dim':self.hidden_dim}


def build_model(kind):
    if kind=='cnn':
        return tf.keras.Sequential([L.Input((21,3)),L.Conv1D(64,3,activation='relu',padding='same'),L.BatchNormalization(),L.MaxPooling1D(2),L.Conv1D(128,3,activation='relu',padding='same'),L.BatchNormalization(),L.MaxPooling1D(2),L.Conv1D(256,3,activation='relu',padding='same'),L.BatchNormalization(),L.Flatten(),L.Dense(128,activation='relu'),L.Dropout(.4),L.Dense(64,activation='relu'),L.Dropout(.2),L.Dense(10,activation='softmax')])
    if kind=='fnn':
        return tf.keras.Sequential([L.Input((42,)),L.Dense(32,activation='relu'),L.Dropout(.4),L.Dense(10,activation='relu'),L.Dense(10,activation='softmax')])
    if kind=='vgg':
        image=L.Input((224,224,3),name='image_input')
        base=tf.keras.applications.VGG16(weights=None,include_top=False,input_tensor=image)
        base.trainable=False
        a=L.Dropout(.5)(L.Dense(4096,activation='relu')(L.Flatten()(base.output)))
        points=L.Input((63,),name='landmark_input')
        b=L.Dense(256,activation='relu')(points)
        x=L.Dropout(.5)(L.Concatenate()([a,b]))
        return tf.keras.Model([image,points],L.Dense(10,activation='softmax',name='classifier_output')(x))
    inputs=L.Input((30,212))
    if kind=='stcnn':
        joints=L.Reshape((30,21,4))(L.Lambda(lambda x:x[:,:,:84])(inputs))
        x=L.BatchNormalization()(L.Conv2D(64,(1,3),activation='relu',padding='same')(joints))
        x=L.BatchNormalization()(L.Conv2D(128,(3,1),activation='relu',padding='same')(x))
        x=L.Lambda(lambda x:tf.reduce_mean(x,axis=2))(x)
        x=L.Concatenate()([x,L.Lambda(lambda x:x[:,:,84:])(inputs)])
        x=L.BatchNormalization()(L.Conv1D(128,5,activation='relu',padding='same')(x))
        x=L.BatchNormalization()(L.Conv1D(128,3,activation='relu',padding='same')(x))
        x=L.Concatenate()([L.GlobalAveragePooling1D()(x),L.GlobalMaxPooling1D()(x)])
        x=L.Dropout(.5)(L.BatchNormalization()(L.Dense(128,activation='relu')(x)))
    elif kind=='lite':
        x=L.Dense(128,activation='relu')(inputs)
        x=L.LayerNormalization(epsilon=1e-6)(L.Add()([x,L.MultiHeadAttention(num_heads=2,key_dim=64,dropout=.1)(x,x)]))
        x=L.LayerNormalization(epsilon=1e-6)(L.Add()([x,L.Dropout(.1)(L.Dense(128,activation='relu')(x))]))
        x=L.Dropout(.3)(L.Dense(64,activation='relu')(L.GlobalAveragePooling1D()(x)))
    elif kind=='glt':
        x=PositionalEmbedding(31,768)(AddCLSToken(768)(L.Dense(768)(inputs)))
        import numpy as np
        idx=np.arange(31)
        mask=(abs(idx[:,None]-idx[None,:])<=2)|(idx[:,None]==0)|(idx[None,:]==0)
        attn=L.MultiHeadAttention(num_heads=8,key_dim=96,dropout=.1)(x,x,attention_mask=tf.constant(mask))
        x=L.LayerNormalization(epsilon=1e-6)(x+attn)
        f=L.Dense(768)(L.Dropout(.1)(L.Dense(2048,activation='relu')(x)))
        x=L.LayerNormalization(epsilon=1e-6)(x+f)
        x=L.LayerNormalization(epsilon=1e-6)(L.Lambda(lambda x:x[:,0,:])(x))
    else:
        raise ValueError(kind)
    return tf.keras.Model(inputs,L.Dense(11,activation='softmax')(x))
