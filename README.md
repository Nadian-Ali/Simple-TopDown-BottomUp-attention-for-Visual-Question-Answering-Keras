# VisualQuestionAnswering-with-top-down-bottom-up-attention
VQA is complext task. Given an image and a question, vqa tries to find the answer to the question using the input image. 



![The Task](https://github.com/Nadian-Ali/Simple-TopDown-BottomUp-attention-for-Visual-Question-Answering-Keras/blob/main/miscellaneous/example%20image.jpg)


In [This]( https://github.com/Nadian-Ali/Visual-Question-Answering-implementation-in-keras-with-VQA2#visual-question-answering-implementation-in-keras-with-vqa2 ) post, a complete for CNN-LSTM vqa architecture was provided. In this reop, the attention mechanism explained in [top down bottom up attention for vaq [paper](https://arxiv.org/abs/1707.07998) is implemented. The code is provided using Keras library. 




MS COCO dataset and VQA2 questions have been utilized  
Text preprocessing was performed by codes from the VQA website 
In order to use this repo you need to download the following files from the VQA2 web site from https://visualqa.org/download.html download:

Image data
36 features per image features can be downloaded from [here](https://github.com/peteanderson80/bottom-up-attention)

Text data:
[Training annotations 2017 v2.0 (https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Train_mscoco.zip)
[Validation annotations 2017 v2.0 (https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Val_mscoco.zip)

[Training questions 2017 v2.0](https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Train_mscoco.zip)
[Validation questions 2017 v2.0](https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Val_mscoco.zip)
[Testing questions 2017 v2.0](https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Test_mscoco.zip)

Image featuers links:
[2014 Train/Val Image Features (120K / 25GB)](https://imagecaption.blob.core.windows.net/imagecaption/trainval_36.zip)
[2014 Testing Image Features (40K / 9GB)](https://imagecaption.blob.core.windows.net/imagecaption/test2014_36.zip)
[2015 Testing Image Features (80K / 17GB)](https://imagecaption.blob.core.windows.net/imagecaption/test2015_36.zip)


arrange your files in the folders as follows:
in annotations folder:
v2_mscoco_train2014_annotations.json
v2_mscoco_val2014_annotations.json



