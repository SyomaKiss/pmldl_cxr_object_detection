[Pneumonia detection winners](https://www.kaggle.com/c/vinbigdata-chest-xray-abnormalities-detection/discussion/207694)

### Public EDA
Class correlation https://www.kaggle.com/bjoernholzhauer/eda-dicom-reading-vinbigdata-chest-x-ray
Class pictures https://www.kaggle.com/debarshichanda/vinbigdata-chest-x-ray-eda-with-plotly

### ObjDet datasets

1. https://competitions.codalab.org/competitions/25848
2. https://www.kaggle.com/c/rsna-pneumonia-detection-challenge/
3. https://www.kaggle.com/raddar/nodules-in-chest-xrays-jsrt
4. https://www.kaggle.com/nih-chest-xrays/data
5. https://www.kaggle.com/c/siim-acr-pneumothorax-segmentation

[rare x-ray datasets](https://www.kaggle.com/raddar/datasets)
### Data split 

[multi-label stratified k-fold](https://github.com/trent-b/iterative-stratification)

# Libs

Metrics https://github.com/rafaelpadilla/Object-Detection-Metrics
Ensemble bboxes https://github.com/ahrnbom/ensemble-objdet
https://github.com/yhenon/pytorch-retinanet

Pneumonia detection winners

# Issues 

1. "Instead we have ILD (Interstitial Lung Disease), which is a diverse and varied set of findings and possible pathologies. We have calcification / nodule/mass / lung opacity / other lesion, which will have a bit of overlap I suspect."
2. Ensure model computes loss on images with no bboxes.
3. Reduce impact of rotation to bounding box sizes, instead of rotating the corners, rotate two points at each edge, choose min/max of rotated points.
4. Adress the labeling process.

##### data
1. R9 highlightes more boxes than others
    1. 347180362348e522905047dde655b6d7
    2. 9a5094b2563a1ef3ff50dc5c7ff71345

# Stolen ideas

1. Modify architecture for small boxes? Like nodule/mass
2. Add classification in output using glogal labels from original dataset. Lung opacity? 
3. Ensembles
4. TTA
5. Create labeling simulation.

# Existing solutions

 - [[LB0.155]](https://www.kaggle.com/c/vinbigdata-chest-xray-abnormalities-detection/discussion/208837) detection (512x512, sparseRCNN (resnet34 on image net), aug) + classification (512x512, )
 - [[LB0.211]](https://www.kaggle.com/awsaf49/vinbigdata-2-class-filter#2-Class-Filter-+-1x1-bbox-trick-🔥) detection (yoloV5, often gives FP) + classification(EfficientNetB6, tackle FP (boost from 0.15)) + 1x1 Bbox-trick (add confidence to empty images gives BOOST)
 - [[LB0.23]](https://www.kaggle.com/corochann/vinbigdata-2-class-classifier-complete-pipeline/comments#Next-step) + "mixup" augmentation, + label smoothing, + ChexPert
 - [[LB ~0.23]](https://www.kaggle.com/c/vinbigdata-chest-xray-abnormalities-detection/discussion/219672) only detection using training with normal images
