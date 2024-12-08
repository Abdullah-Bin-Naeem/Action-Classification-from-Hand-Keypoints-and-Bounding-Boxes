# Hand Action Classification using Keypoints and Bounding Boxes

This repository focuses on classifying hand actions using keypoints and bounding boxes extracted from annotated datasets. The models are implemented and tested in the provided Jupyter notebook `models.ipynb`. The classification is achieved through various architectures that incorporate keypoints, bounding boxes, and temporal frame sequences. The notebook includes code for extracting and transforming features, as well as for training and evaluating the models.

## Table of Contents
1. [Overview](#overview)
2. [Architectures](#architectures)
3. [Requirements](#requirements)
4. [Usage](#usage)
5. [Dataset](#dataset)
6. [License](#license)
7. [Acknowledgments](#acknowledgments)

---

## Overview

This repository implements and tests four distinct architectures for hand action classification. Each architecture varies in terms of the use of keypoints, bounding boxes, temporal frame sequences, and model types. Below is a brief description of the four architectures, which are all implemented within the Jupyter notebook `models.ipynb`.

---

## Architectures

### Architecture 1: Keypoints and Bounding Boxes with ResNet50
- **Description**: In this architecture, keypoints and bounding boxes are extracted from the annotated data. These features are passed through linear learnable layers for transformation, then embedded into the alpha channel of the image. The modified image is fed into ResNet50 for classification.
  
- **Key Steps**:
  1. Extract keypoints and bounding boxes.
  2. Transform features using learnable linear layers.
  3. Embed the transformed features into the alpha channel of the image.
  4. Pass the modified image through ResNet50 for classification.

### Architecture 2: Temporal Frame Variations with ResNet50
- **Description**: This architecture tests the effect of using different frame sequences: 6 frames (2 previous, 2 next, and the current frame) and 3 frames (1 previous, 1 next, and the current frame). Keypoints and bounding boxes are extracted and transformed as in Architecture 1, and the frames are passed through ResNet50 for classification.

- **Key Steps**:
  1. Use 6 frames (2 previous, 2 next, and the current frame) or 3 frames (1 previous, 1 next, and the current frame).
  2. Extract and transform keypoints and bounding boxes.
  3. Pass the frames through ResNet50 for classification.

### Architecture 3: ResNet Bottleneck with LSTM
- **Description**: In this architecture, a ResNet bottleneck layer is followed by an LSTM network to capture temporal context autoregressively. The LSTM helps to model dependencies between the previous and next frames, improving classification accuracy.

- **Key Steps**:
  1. Extract keypoints and bounding boxes.
  2. Pass the features through a ResNet bottleneck.
  3. Use LSTM to capture temporal context.
  4. Classify the features after the LSTM layer.

### Architecture 4: ViT for Attention and Feature Fusion with ResNet50
- **Description**: This architecture uses Vision Transformer (ViT) to capture attention from a sequence of 6 frames (1 current, 2 previous, and 2 next). The image features are extracted using ResNet50 and fused with transformed keypoints and bounding boxes. The fused features are passed through ResNet50 for final classification.

- **Key Steps**:
  1. Use 6 frames: 1 current, 2 previous, and 2 next frames from the video.
  2. Apply Vision Transformer (ViT) for attention capture.
  3. Extract image features using ResNet50.
  4. Transform keypoints and bounding boxes using learnable layers.
  5. Fuse all features (ViT features, ResNet50 features, transformed keypoints/bounding boxes).
  6. Pass the fused features through ResNet50 for classification.

---

## Requirements

To run the notebook and experiment with the models, make sure you have the following dependencies installed:

- Python 3.x
- Jupyter Notebook or JupyterLab
- PyTorch
- torchvision
- numpy
- OpenCV (for video frame processing)
- PIL (Python Imaging Library)
- Any other relevant libraries used in the notebook (e.g., transformers)

You can install the required libraries by running:

```bash
pip install -r requirements.txt

### Explanation of the Sections:

1. **Overview**: Provides a brief introduction to what the repository is about and gives context to the work.
2. **Architectures**: A detailed description of each architecture implemented in the notebook, explaining how keypoints and bounding boxes are processed and classified.
3. **Requirements**: Lists all the dependencies needed to run the notebook and the installation steps.
4. **Usage**: Instructions on how to clone the repository, install dependencies, and use the Jupyter notebook to run the models.
5. **Dataset**: Mentions the need for a compatible dataset and provides a placeholder for any dataset-specific instructions or preprocessing details.
6. **License**: Includes licensing information. You can adjust it if you're using a different license.
7. **Acknowledgments**: Credits the major papers and libraries used in the project, such as ResNet50, ViT, and OpenCV.

### Customization:
- Update the **GitHub repository URL** in the clone command.
- If you have specific dataset instructions, you can expand the **Dataset** section with more details on how users can preprocess or download datasets.
