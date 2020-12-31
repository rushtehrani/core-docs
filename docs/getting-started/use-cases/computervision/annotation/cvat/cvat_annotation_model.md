---
title: Training with built-in models
sidebar_label: Training with built-in models
description: Onepanel - Training with built-in models
---

## Training with built-in models from CVAT

1. Click on **Open** for a task you want to train a model on.
  
  ![Open task](/img/cvat_open.png)

2. Click on **Job #X**, where X could be any job number. Annotate a few frames. For testing, you can annotate one frame. But ideally, you want to have thousands of objects to train a deep learning model on. Alternatively, you can just run pre-annotation if your labels are common ones.

3. Click on **Actions** for a task you want to train a model on. Then, click on **Execute training Workflow**.

  ![Select training workflow](/img/cvat_select_workflow_execution.png)

4. Select a training Workflow Template. By default, you can use **TF Object Detection Training** for object detection or **MaskRCNN Training** for semantic segmentation.

  ![Train a model from CVAT](/img/tf-object-detection.png)

  :::tip
  Note you can easily add your own models as well. See our [documentation](/docs/getting-started/use-cases/computervision/annotation/cvat/adding_custom_model) for more information on adding custom models. 
  :::

5. Update hyperparameters and settings depending on your model and data. See below for more information.

6. You can optionally select the checkpoint path from previously trained model or leave this field empty.

7. Click **Submit**. This will execute the Onepanel Workflow for selected model. You can see Workflow logs by going to Workflow execution page. You can find the URL for the same in the notification card.
  
  ![Workflow URL](/img/execution_url.png)

  Trained model and other outputs will be stored on cloud storage and will be synced with CVAT locally so that you can use this to pre-annotate other frames. 

  :::note
  You can also use this trained model to run pre-annotation in CVAT. See our [documentation](/docs/getting-started/use-cases/computervision/annotation/cvat/cvat_automatic_annotation) for more information on pre-annotation.
  :::

## TensorFlow Object Detection models

You can use any supported model for TensorFlow Object Detection API to train your custom pre-annotation models. Here, we provide a brief explanation of choosing one model over another based on your needs. Some models are faster than others, whereas some are more accurate than others. We hope this information will help you choose the right model for your task. 

![TensorFlow Object Detection Workflow](/img/tf-object-detection.png)

### Choosing the right model

- We currently support several faster-rcnn models. All of these models are similar except that of the backbone used for the feature extraction. The backbones used are, in increasing order of complexity (i.e more layers), ResNet50, ResNet101. As the model complexity increases, the computation requirement will also increase. If you have very complex data (i.e hundreds of annotations in one image), it is recommended that you choose a more complex model (i.e ResNet101).

- Faster-rcnn models are generally more accurate than ssd models. However, sometimes you are better off using ssd models if your data is easy to learn (i.e 1 or 2 bounding box per image).

#### frcnn-nas-coco:

- If you are using frcnn-as-coco, choose a machine with at least 2 GPUs as this model requires more memory. A device with 1 GPU will throw an error.

This is a type of faster-rcnn model with NAS backbone. If you are not sure about which model to use, we recommend using the SSD-based model (i.e SSD-mobilenet-v2).

Depending upon your data, you can set epochs to train your model. There is no standard value that can work for all datasets. You generally have to try a different number of epochs to get the best model. Ideally, you do so by monitoring loss of your model while training. But if you are looking for a recommendation, we recommend you set epochs as follows: (number of images / batch_size (default: 1)) * 50. For instance, if you have 100 images, then your epochs will be 5000(rounded). Please note that the model will be trained using a pre-trained model, so you don't need to train as long as you would have to when not using the pre-trained model.

Note that the current implementation of faster-rcnn in TensorFlow Object Detection API does not support batch training. That is, you shouldn't change `batch-size`.

**_Defaults_**: batch-size: 1, initial-learning-rate=0.003, num-steps=10000

#### frcnn-res101-coco: 

This is a type of faster-rcnn model with ResNet101 backbone. If you are not sure about which model to use then we recommend you use SSD based model (i.e ssd-mobilenet-v2). 

For how to set epochs, you can take a look at frcnn-nas-coco model since both models are faster-rcnn based.

Note that the current implementation of faster-rcnn in TensorFlow Object Detection API does not support batch training. That is, you shouldn't change `batch-size`.

**_Defaults_**: batch-size: 1, initial-learning-rate: 0.0003, num-steps=10000

#### frcnn-res101-lowp

This is a type of faster-rcnn model with ResNet101 backbone with low number of proposals. If you are not sure about which model to use then we recommend you use SSD based model (i.e ssd-mobilenet-v2). If you are looking for a more complex and accurate model, then check out frcnn-res101-coco.

For how to set epochs, you can take a look at frcnn-nas-coco model since both models are faster-rcnn based.

Note that the current implementation of faster-rcnn in TensorFlow Object Detection API does not support batch training. That is, you shouldn't change `batch-size`.

**_Defaults_**: batch-size: 1, initial-learning-rate: 0.0003, num-steps=10000

#### frcnn-res50-coco

This is a type of faster-rcnn model with ResNet50 backbone. If you are not sure about which model to use then we recommend you use SSD based model (i.e ssd-mobilenet-v2). If you are looking for a more complex and accurate model, then check out frcnn-res101-coco.

For how to set epochs, you can take a look at frcnn-nas-coco model since both models are faster-rcnn based.

Note that the current implementation of faster-rcnn in TensorFlow Object Detection API does not support batch training. That is, you shouldn't change `batch-size`.

**_Defaults_**: batch-size: 1, initial-learning-rate: 0.0003, num-steps=10000

#### ssd-mobilenet-v2-coco

SSD-based networks such as `ssd-mobilenet-v2` are faster than faster-rcnn based models. However, they are not as accurate as faster-rcnn based models. This model is generally recommended since its accurate and fast enough. If you don't know much about your data or your data's complexity, then we recommend you go with this model.

You will find the pre-trained model and config file for ssd-mobilenetv2 model trained on COCO dataset.

This model is a good place to start if you don't have any specific model in mind. If you are data is very complicated (i.e many annotations per image), you should prefer faster-rcnn models over SSD.

Depending upon your data, you can set epochs to train your model. There is no standard value that can work for all datasets. You generally have to try a different number of epochs to get the best model. Ideally, you do so by monitoring loss of your model while training. But if you are looking for a recommendation, we recommend you set epochs as follows: (number of images / batch-size (default: 24)) * 1000. For instance, if you have 100 images, then your epochs will be 4000 (rounded). Note that the model will be trained using a pre-trained model, so you don't need to train as long as you would have to when not using the pre-trained model.

**_Defaults_**: batch-size: 24, initial-learning-rate: 0.004, num-steps=15000

Note that same instructions apply for **ssd-mobilenet-v1** and **ssd-mobilenet-lite**. The only difference is the backbone model (i.e mobilenet v1) that they use.

### TFOD hyperparameters

You can specify some arguments in the `Hyperparameters` field seperated by a new line. 

Here is a sample for Tensorflow Object Detection API: 

```bash
num-steps=100
initial-learning-rate=0.0003
``` 
Our models are using default parameters used to outperform on the COCO dataset, and you should not change them in most cases.

#### Basic Hyperparameters

| Parameter                 | Description                                                                 |
| ------------------------- | --------------------------------------------------------------------------- |
| num-steps                 | number of steps to train your model for.                                    |
| initial-learning-rate     | initial learning rate for the model.                                        |
| batch-size                | batch size for the training (should not be changed for faster-rcnn models). |
| num-clones                | number of GPUs to train the model.                                          |

Note that you need to set `num-clones` to `4` (number of GPUs) if you select a node pool with say 4 GPUs (Tesla V100).

#### Optimizer Hyperparameters

**_faster-rcnn_** based models use SGD with momentum and learning rate schedule; you can change these parameters:

| Parameter                 | Description                                     |
| ------------------------- | ----------------------------------------------- |
| schedule-step-1           | step 1 for learning rate decay.                 |
| schedule-lr-1             | learning rate used after schedule-step-1 steps. |
| schedule-step-2           | step 2 for learning rate decay.                 |
| schedule-lr-2             | learning rate used after schedule-step-2 steps. |
| momentum_optimizer_value  | momentum factor.                                |

**_ssd_**  based models use RMSprop optimizer; you can change these parameters:

| Parameter                 | Description                                     |
| ------------------------- | ----------------------------------------------- |
| decay_steps               | steps for exponential learning rate decay.      |
| decay_factor              | factor for learning rate decay.                 |
| momentum_decay            | RMSprop decay.                                  |
| momentum_epsilon          | RMSprop epsilon.                                |
| momentum_optimizer_value  | momentum factor.                                |

#### Post-processing Hyperparameters

| Parameter                                         | Description                                               |
| ------------------------------------------------- | --------------------------------------------------------- |
| second_stage_nms_score_threshold                  | non-maximal supression score threshold.                   |
| second_stage_nms_iou_threshold                    | non-maximal supression intersection over union threshold. |
| second_stage_max_detections_per_class             | max detections per class.                                 |
| second_stage_max_detections_max_total_detections  | max total detections.                                     |  
| second_stage_use_dropout                          | use dropout (True; False)                                 |
| second_stage_dropout_keep_probability             | dropout keep probability.                                 | 
| second_stage_regularizer_weight                   | l2 regularizer weight.                                    | 

#### Network Hyperparameters

**_faster-rcnn_**

| Parameter                       | Description                                               |
| ------------------------------- | --------------------------------------------------------- |
| first_stage_features_stride     | feature extractor stride.                                 |
| height_stride                   | anchor generator height stride.                           |
| width_stride                    | anchor generator width stride.                            |
| first_stage_regularizer_weight  | l2 regularizer weight.                                    |
| first_stage_nms_score_threshold | non-maximal supression score threshold.                   |
| first_stage_nms_iou_threshold   | non-maximal supression intersection over union threshold. |
| first_stage_max_proposals       | max total proposals.                                      |
| maxpool_kernel_size             | max pool layers kernel size.                              |
| maxpool_stride                  | max pool layers stride.                                   |

**_ssd_**  

| Parameter                       | Description                                     |
| ------------------------------- | ----------------------------------------------- |
| first_stage_features_stride     | feature extractor stride.                       |
| first_stage_activation          | activation function used in first stage(NONE = 0; RELU = 1; RELU_6 = 2; SWISH = 3;) |
| first_stage_batchnorm_decay     | factor for batch normalization decay.           |
| first_stage_batchnorm_epsilon   | factor for batch normalization epsilon.         |
| boxcoder_y_scale                | boxcoder y scale.                               |
| boxcoder_x_scale                | boxcoder x scale.                               |
| boxcoder_height_scale           | boxcoder height scale.                          |
| boxcoder_width_scale            | boxcoder width scale.                           |
| matched_threshold               | matcher threshold to set feature as mached.     |
| unmatched_threshold             | matcher threshold to set feature as unmached.   |
| second_stage_activation         | activation function used in first stage(NONE = 0; RELU = 1; RELU_6 = 2; SWISH = 3;) |
| second_stage_batchnorm_decay    | factor for batch normalization decay.           |
| second_stage_batchnorm_epsilon  | factor for batch normalization epsilon.         |
| second_stage_kernel_size        | CNN kernel size.                                |
| second_stage_box_code_size      | box code kernel size.                           |
| anchor_generator_num_layers     | anchor generator num layers.                    |
| anchor_generator_min_scale      | anchor generator min scale.                     |
| anchor_generator_max_scale      | anchor generator max scale.                     |


## MaskRCNN model

MaskRCNN is a popular model for segmentation tasks. We use [this](https://github.com/matterport/Mask_RCNN) implementation of MaskRCNN for training and inference.

The process to train a Mask-RCNN model on CVAT is similar to the above procedure except that you need to select Mask-RCNN after clicking on Create Annotation Model.

![MaskRCNN Workflow](/img/maskrcnn-training.png)

:::note
Polygons must be used in annotations to train MaskRCNN properly. 
:::

### MaskRCNN hyperparameters 

Even though you don't need to enter any other parameters to start the training of Mask-RCNN, it is recommended that you pass the correct epochs according to your data. Mask-RCNN is a very deep model that takes too much time to train and get enough accuracy. 
We allow you to set epochs for three different parts of the model. These parts are called `stage1`, `stage2` and `stage3`. You can set corresponding epochs as follows:

```bash
stage-1-epochs=1
stage-2-epochs=2
stage-3-epochs=3
```

Epochs are cumulative on each stage. For instance, if you set stage-1-epochs=1, stage-2-epochs=2 and stage-3-epochs=3, you will train the model for three epochs in total, one per phase. 

If you have a few images (few hundreds), we recommend setting total epochs (stage1+stage2+stage3) less than 10. We advise you to set more epochs for stage1 than others. As your data size increases or your data's complexity increases, you might want to increase epochs. 


<!-- ## Notes

- There are certain parameters that are prefixed with `cvat` in TF Object Detection Training and MaskRCNN Training workflows. Those are special parameters and will be populated in whole or partly by the CVAT. For example, `cvat-output-path` is generated by the CVAT and it won't be shown to users. Another example is `cvat-finetune-checkpoint`. CVAT will automatically find all available checkpoints for a given workflow/model since they are available locally because of file syncer. 
- Note that these instructions are for default models that we provide. You can always edit these workflows or even add your own workflows/models and train them. -->