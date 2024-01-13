# FALL-DETECTION-AND-ACTIVITY-RECOGNITION

Recognition of human activity has become a focal point of recent research, with machine learning techniques demonstrating effectiveness in monitoring movements such as walking, running, standing, sitting, jumping, and falling. As the aging population increases, detecting falls becomes crucial, given their association with severe injuries, especially among the elderly. Traditional fall detection methods utilizing sensors face drawbacks such as discomfort, inconsistency, and the need for constant monitoring. To address these challenges, this project proposes a vision-based solution using images captured by a video camera and leveraging the CatBoost Classifier for activity prediction based on human skeleton features.

## Problem Statement:

Falls, particularly among the elderly, pose a significant risk, leading to catastrophic injuries. Existing solutions employing sensors like accelerometers and gyroscopes are invasive, uncomfortable, and require continuous user compliance. The proposed vision-based system aims to overcome these limitations by relying on non-intrusive image analysis. The project specifically addresses the need for accurate fall detection and activity recognition without the drawbacks associated with traditional sensor-based methods.

## The Data

The project utilizes the UP-FALL dataset, a multi-modal dataset freely available for fall detection research. This dataset incorporates wearable sensors, ambient sensors, and vision equipment, allowing for image segmentation of the human body and non-intrusive monitoring. The UP-FALL dataset provides a rich source of raw data and features, making it suitable for validating the proposed vision-based fall detection and activity recognition system.

Dataset Link: [UP-FALL dataset](https://sites.google.com/up.edu.mx/har-up/)

The project utilizes Random Forest and CatBoost algorithms for fall detection and activity recognition. Pre-processing involves skeleton sequence cleanup, pose feature extraction, and employing the CatBoost Classifier. Results demonstrate that the CatBoost algorithm outperforms Random Forest, offering a non-intrusive vision-based solution with superior accuracy, precision, recall, and F1-score for identifying human activity, especially beneficial for the elderly.
