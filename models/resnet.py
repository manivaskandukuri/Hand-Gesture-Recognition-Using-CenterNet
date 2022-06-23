from keras_resnet import models as resnet_models
from keras_mobilenet.models import MobileNet
from keras_hourglassnet.models import HourglassNet
from keras_densenet.models import DenseNet161, DenseNet264
from tensorflow.keras.applications.densenet import DenseNet201
from tensorflow.keras.layers import Input, Conv2DTranspose, BatchNormalization, ReLU, Conv2D, Lambda, MaxPooling2D, Dropout
from tensorflow.keras.layers import ZeroPadding2D, Add, Activation, concatenate 
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
import tensorflow.keras.backend as K
import tensorflow as tf
from attention_network.danet import PAM_Module, CAM_Module
from attention_network.cbam import cbam_block
from attention_network.tanet import TripletAttention
from deformable_conv.deform_layer import Offset2D
from deformable_conv.deform_layer_v2 import Offset2D_v2
from losses import loss



def nms(heat, kernel=3):
    hmax = tf.nn.max_pool2d(heat, (kernel, kernel), strides=1, padding='SAME')
    heat = tf.where(tf.equal(hmax, heat), heat, tf.zeros_like(heat))
    return heat


def topk(hm, max_objects=100):
    hm = nms(hm)
    # (b, h * w * c)
    b, h, w, c = tf.shape(hm)[0], tf.shape(hm)[1], tf.shape(hm)[2], tf.shape(hm)[3]
    # hm2 = tf.transpose(hm, (0, 3, 1, 2))
    # hm2 = tf.reshape(hm2, (b, c, -1))
    hm = tf.reshape(hm, (b, -1))
    # (b, k), (b, k)
    scores, indices = tf.nn.top_k(hm, k=max_objects)
    # scores2, indices2 = tf.nn.top_k(hm2, k=max_objects)
    # scores2 = tf.reshape(scores2, (b, -1))
    # topk = tf.nn.top_k(scores2, k=max_objects)
    class_ids = indices % c
    xs = indices // c % w
    ys = indices // c // w
    indices = ys * w + xs
    return scores, indices, class_ids, xs, ys


def evaluate_batch_item(batch_item_detections, num_classes, max_objects_per_class=20, max_objects=100,
                        iou_threshold=0.5, score_threshold=0.1):
    batch_item_detections = tf.boolean_mask(batch_item_detections,
                                            tf.greater(batch_item_detections[:, 4], score_threshold))
    detections_per_class = []
    for cls_id in range(num_classes):
        class_detections = tf.boolean_mask(batch_item_detections, tf.equal(batch_item_detections[:, 5], cls_id))
        nms_keep_indices = tf.image.non_max_suppression(class_detections[:, :4],
                                                        class_detections[:, 4],
                                                        max_objects_per_class,
                                                        iou_threshold=iou_threshold)
        class_detections = tf.gather(class_detections, nms_keep_indices)
        detections_per_class.append(class_detections)

    batch_item_detections = tf.concat(detections_per_class, axis=0)

    def filter():
        nonlocal batch_item_detections
        _, indices = tf.nn.top_k(batch_item_detections[:, 4], k=max_objects)
        batch_item_detections_ = tf.gather(batch_item_detections, indices)
        return batch_item_detections_

    def pad():
        nonlocal batch_item_detections
        batch_item_num_detections = tf.shape(batch_item_detections)[0]
        batch_item_num_pad = tf.maximum(max_objects - batch_item_num_detections, 0)
        batch_item_detections_ = tf.pad(tensor=batch_item_detections,
                                        paddings=[
                                            [0, batch_item_num_pad],
                                            [0, 0]],
                                        mode='CONSTANT',
                                        constant_values=0.0)
        return batch_item_detections_

    batch_item_detections = tf.cond(tf.shape(batch_item_detections)[0] >= 100,
                                    filter,
                                    pad)
    return batch_item_detections


def decode(hm, wh, reg, max_objects=100, nms=True, flip_test=False, num_classes=20, score_threshold=0.1):
    if flip_test:
        hm = (hm[0:1] + hm[1:2, :, ::-1]) / 2
        wh = (wh[0:1] + wh[1:2, :, ::-1]) / 2
        reg = reg[0:1]
    scores, indices, class_ids, xs, ys = topk(hm, max_objects=max_objects)
    b = tf.shape(hm)[0]
    # (b, h * w, 2)
    reg = tf.reshape(reg, (b, -1, tf.shape(reg)[-1]))
    # (b, h * w, 2)
    wh = tf.reshape(wh, (b, -1, tf.shape(wh)[-1]))
    # (b, k, 2)
    topk_reg = tf.gather(reg, indices, batch_dims=1)
    # (b, k, 2)
    topk_wh = tf.cast(tf.gather(wh, indices, batch_dims=1), tf.float32)
    topk_cx = tf.cast(tf.expand_dims(xs, axis=-1), tf.float32) + topk_reg[..., 0:1]
    topk_cy = tf.cast(tf.expand_dims(ys, axis=-1), tf.float32) + topk_reg[..., 1:2]
    scores = tf.expand_dims(scores, axis=-1)
    class_ids = tf.cast(tf.expand_dims(class_ids, axis=-1), tf.float32)
    topk_x1 = topk_cx - topk_wh[..., 0:1] / 2
    topk_x2 = topk_cx + topk_wh[..., 0:1] / 2
    topk_y1 = topk_cy - topk_wh[..., 1:2] / 2
    topk_y2 = topk_cy + topk_wh[..., 1:2] / 2
    # (b, k, 6)
    detections = tf.concat([topk_x1, topk_y1, topk_x2, topk_y2, scores, class_ids], axis=-1)

    if nms:
        detections = tf.map_fn(lambda x: evaluate_batch_item(x[0],
                                                             num_classes=num_classes,
                                                             score_threshold=score_threshold),
                               elems=[detections],
                               fn_output_signature=tf.float32)

    return detections


def centernet(num_classes, backbone='resnet50', use_triplet_attention=False, input_size=512, max_objects=100, score_threshold=0.1,
              nms=True,
              flip_test=False):
    assert backbone in ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 
    'densenet201', 'densenet161', 'densenet264', 'mobilenet', 'hourglassnet104', 'hourglassnet52']
    output_size = input_size // 4

    image_input = Input(shape=(input_size, input_size, 3))
    hm_input = Input(shape=(output_size, output_size, num_classes))
    wh_input = Input(shape=(max_objects, 2))
    reg_input = Input(shape=(max_objects, 2))
    reg_mask_input = Input(shape=(max_objects,))
    index_input = Input(shape=(max_objects,))

    if use_triplet_attention:
      triplet_attention = TripletAttention
    else:
      triplet_attention = None


    if backbone == 'resnet18':
        backbone_model = resnet_models.ResNet18(image_input, include_top=False, triplet_attention=triplet_attention)
    elif backbone == 'resnet34':
        backbone_model = resnet_models.ResNet34(image_input, include_top=False, triplet_attention=triplet_attention)
    elif backbone == 'resnet50':
        backbone_model = resnet_models.ResNet50(image_input, include_top=False, triplet_attention=triplet_attention)
        # resnet = ResNet50(input_tensor=image_input, include_top=False)
    elif backbone == 'resnet101':
        backbone_model = resnet_models.ResNet101(image_input, include_top=False, triplet_attention=triplet_attention)
    elif backbone == 'resnet152':
        backbone_model = resnet_models.ResNet152(image_input, include_top=False, triplet_attention=triplet_attention)
    elif backbone == 'hourglassnet104':
        backbone_model = HourglassNet(image_input, num_stacks=2, include_top=False)
    elif backbone == 'hourglassnet52':
        backbone_model = HourglassNet(image_input, num_stacks=1, include_top=False)
    elif backbone == 'densenet201':
        backbone_model = DenseNet201(include_top=False, weights=None, input_tensor=image_input, input_shape=(512, 512, 3), pooling=max)
    elif backbone == 'densenet161':
        backbone_model = DenseNet161(image_input, include_top=False, pooling=max)
    elif backbone == 'densenet264':
        backbone_model = DenseNet264(image_input, include_top=False, pooling=max)
    else: 
        backbone_model = MobileNet(image_input, include_top=False)


    #print("Backbone Summary:", backbone_model.summary())
    

    if (backbone == 'hourglassnet104'):
      weights_path = 'checkpoints/Pretrained/ctdet_coco_hg.hdf5'
      backbone_model.load_weights(weights_path, by_name=True)
      backbone_model.trainable = False

    for i in range(len(backbone_model.outputs)):
      print("Stage-",i+1,"output:", backbone_model.outputs[i].shape)

    # (b, 16, 16, 2048) for Resnet50
    x = backbone_model.outputs[-1]
    x = Dropout(rate=0.5)(x)

    if (backbone in ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'densenet161', 'densenet201', 'mobilenet']):  
      
      # Passing through Dual Attention Network 
      pam_out1 = PAM_Module(filters=x.shape[-1])(x)
      cam_out1 = CAM_Module(filters=x.shape[-1])(x)
      x = Add()([pam_out1, cam_out1])


      '''# Passing through CBAM 
      x = cbam_block(x)'''
      

      # decoder
      num_filters = 256
      for i in range(3):
          num_filters = num_filters // pow(2, i)
          x = Conv2DTranspose(num_filters, (4, 4), strides=2, use_bias=False, name="dec_conv_trans{}".format(i),
                              padding='same', kernel_initializer='he_normal',
                              kernel_regularizer=l2(5e-4))(x)
          #x = BatchNormalization(name="dec_bn{}".format(i))(x)
          x = ReLU(name="dec_relu{}".format(i))(x)

          '''shortcut = backbone_model.outputs[2-i]
          shortcut = Conv2D(num_filters, (1, 1), use_bias=False, name="shortcut{}".format(3-i), 
                    padding="same", kernel_initializer='he_normal',
                    kernel_regularizer=l2(5e-4))(shortcut)
          x = concatenate([x, shortcut], axis=-1)'''
    
    
    '''# Passing through Dual Attention Network 
    pam_out2 = PAM_Module(filters=x.shape[-1])(x)
    cam_out2 = CAM_Module(filters=x.shape[-1])(x)
    x = Add()([pam_out2, cam_out2])'''
    

    '''# Passing through CBAM 
    #print("x shape before:", x.shape)
    x = cbam_block(x)
    #print("x shape after:", x.shape)'''
    
    # hm header
    y1 = Conv2D(64, 3, padding='same', use_bias=False, name="hm_conv", kernel_initializer='he_normal', kernel_regularizer=l2(5e-4))(x)
    y1 = ReLU(name="hm_relu")(y1)
    y1 = Conv2D(num_classes, 1, name="hm_conv1x1", kernel_initializer='he_normal', kernel_regularizer=l2(5e-4), activation='sigmoid')(y1)
    #y1 = Conv2D(num_classes, 1, name="hm_conv1x1", kernel_initializer='he_normal', kernel_regularizer=l2(5e-4))(y1)
    #y1 = Activation('sigmoid', name='hm_sigmoid')(y1)

    # wh header
    y2 = Conv2D(64, 3, padding='same', use_bias=False, name="wh_conv", kernel_initializer='he_normal', kernel_regularizer=l2(5e-4))(x)
    y2 = ReLU(name="wh_relu")(y2)
    y2 = Conv2D(2, 1, name="wh_conv1x1", kernel_initializer='he_normal', kernel_regularizer=l2(5e-4))(y2)

    # reg header
    y3 = Conv2D(64, 3, padding='same', use_bias=False, name="reg_conv", kernel_initializer='he_normal', kernel_regularizer=l2(5e-4))(x)
    y3 = ReLU(name="reg_relu")(y3)
    y3 = Conv2D(2, 1, name="reg_conv1x1", kernel_initializer='he_normal', kernel_regularizer=l2(5e-4))(y3)

    
    loss_ = Lambda(loss, name='centernet_loss')(
        [y1, y2, y3, hm_input, wh_input, reg_input, reg_mask_input, index_input])

    model = Model(inputs=[image_input, hm_input, wh_input, reg_input, reg_mask_input, index_input], outputs=[loss_])

    #detections = decode(y1, y2, y3)
    detections = Lambda(lambda x: decode(*x,
                                         max_objects=max_objects,
                                         score_threshold=score_threshold,
                                         nms=nms,
                                         flip_test=flip_test,
                                         num_classes=num_classes))([y1, y2, y3])
    prediction_model = Model(inputs=image_input, outputs=detections)
    debug_model = Model(inputs=image_input, outputs=[y1, y2, y3])
    return model, prediction_model, debug_model
