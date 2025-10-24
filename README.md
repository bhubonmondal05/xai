# <center>Explainable AI for Chest X-Ray Analysis</center>
#### This is the history to track our progress.

1. Findout xray dataset for the required project. [Dataset Link](https://www.kaggle.com/datasets/jtiptj/chest-xray-pneumoniacovid19tuberculosis) (Unzip and keep in current folder.)
   <br>
   <center>
       <img src="sample/cov.jpg" style="width:120px; height:120px; object-fit:cover;">
       <img src="sample/nor.jpeg" style="width:120px; height:120px; object-fit:cover;">
       <img src="sample/pnu.jpeg" style="width:120px; height:120px; object-fit:cover;">
       <img src="sample/tub.png" style="width:120px; height:120px; object-fit:cover;">
   </center>
   <br>
$~~~~~~~~~~~~~~$   Covid19   $~~~~~~~~~~~~~~$   Normal  $~~~~~~~~~~~~~~$   Pneumonia  $~~~~~~~~~~~~~~$  Tuberculosis
   <br>
2. Convert all images into same extension (jpg).
   <br>
   jpeg, png  --> jpg
   <br>
3. feature extraction using DenseNet saved in *save_features.csv*.
   <br>
4. apply Dense Neural Network (DNN) classifier and generate tarined model named *dnn_classifier.pkl*
   <br>
5. Developed Flask API to host the model and process image uploads from the frontend.
   <br>
6. Developed Flutter application.


Some Required Files = [Google Drive Link](https://drive.google.com/drive/folders/1BbfmUoDlft2BvakvLJ5ndcyxdFIJBOtK?usp=sharing)
(Those files are not avail to upload in github due to 25mb file size limit)
