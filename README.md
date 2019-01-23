# SNPECam

A framework that incoporate android camera and Qualcomm SNPE deep learning model deployment in a single app. The default android samples provided from Qualcomm SNPE works with a bundle of images & model files that is very hard to use and not intuitive at all. This sample shows how to open android camera and feed frames to snpe models for inference continuously.


If you want to knoe about SNPE, please refer to: https://developer.qualcomm.com/software/qualcomm-neural-processing-sdk

About the mobilenet ssd model included with the repo:
Model name: mobilenet SSD V1
The model was converted to SNPE's dlc format from this link: http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2017_11_17.tar.gz

The Camera part comes from Google's tensorflow android example and I adapted it for use in this framework. Tensorflow repo is available from: https://github.com/tensorflow/tensorflow






