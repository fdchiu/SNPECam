# SNPECam

A framework that incoporate android camera and Qualcomm SNPE deep learning model deployment in a single app. The default android samples provided from Qualcomm SNPE works with a bundle of images & model files that is very hard to use and not intuitive at all. This sample shows how to open android camera and feed frames to snpe models for inference continuously.


If you want to knoe about SNPE, please refer to: https://developer.qualcomm.com/software/qualcomm-neural-processing-sdk

About the mobilenet ssd model included with the repo: 

Model name: mobilenet SSD V1

The model was converted to SNPE's dlc format from this link: 

http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2017_11_17.tar.gz

This model is fully working. You can build android apk use this model by selecting SSDMobilenet build flavor.

Second model: CaffeSSD

This model was converted from the Caffe SSD repo here: 
https://github.com/chuanqi305/MobileNet-SSD

I then converted it into SNPE dlc format. This model is not working normally. The app build based on CaffeSSD flavor will crash after running for a few frames. I am still in the process finding out the cause of the crash. Will update the model once I am able to convert and run it successfully.

Model Conversion:
I will be publishing a blog about how to convert models into dlc format. 

The Camera part comes from Google's tensorflow android example and I adapted it for use in this framework. Tensorflow repo is available from: https://github.com/tensorflow/tensorflow






