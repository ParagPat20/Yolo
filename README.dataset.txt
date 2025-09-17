# JeCH > 2025-09-17 12:40pm
https://universe.roboflow.com/iot-q3ahg/jech-fbc1d

Provided by a Roboflow user
License: CC BY 4.0

This project develops a robust computer vision model to estimate crowd density from live urban video feeds, with applications in public safety, event management, urban planning, and retail analytics.

Dataset:

522 images from New York and Tokyo live feeds.

Captures diverse conditions: lighting, weather, times of day, and varying crowd densities.

Challenges include heavy occlusion, scale variation, and variable lighting.

Annotation Strategy:

Point-based annotation: each person labeled with a single point on their head/torso.

Annotated using Roboflow for efficient, accurate labeling.

Model Architecture:

Base: VGG16 pre-trained on ImageNet, used as a frozen feature extractor.

Custom regression layers added on top to predict total crowd count per image.

Regression approach is chosen over detection to handle high occlusion.

Goal & Applications:

Accurate real-time crowd density estimation.

Supports public safety alerts, traffic/pedestrian management, event planning, and retail analytics.