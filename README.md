
## Unity Handwritten Character Recognition 
Handwritten character recognition (letters A-Z and digits 0-9) implemented as a mobile app using the Unity game engine and C#. The predictions are made in real time by a neural network, which is implemented from scratch in a simple, straightforward way. 

The user's input is recorded by an interactive drawing interface, created mainly for touch screen mobile devices. 

Training and data exploration was done inside Unity Editor.

<p align="center">
  <img src="https://github.com/Durostolar/unity-written-character-recognition/blob/master/Screenshots/inference.png" alt="1" width="219"/> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <img src="https://github.com/Durostolar/unity-written-character-recognition/blob/master/Screenshots/collect_scene.png" alt="2" width="220"/> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <img src="https://github.com/Durostolar/unity-written-character-recognition/blob/master/Screenshots/train_scene.png" alt="3" width="220"/>
</p>

----------
### App Structure
-   **Inference Scene:** Final app interface with drawing canvas, pen thickness slider, and prediction results
-   **Draw Scene:** For collecting handwritten samples with pen thickness variability. User drawing interface with "Clear" and "Submit" buttons.
-   **Training Scene:** Loads datasets, procedures to train the neural network within Unity Editor.
----------

### Network and training
- Basic MLP (576 x 200 x 36) with SGD training on combined datasets for alphanumeric symbol classification.
- ReLU, sigmoid. He initialization.
- Model parameters are saved and loaded in JSON format for easy reuse.
----------

### Data
Following sets were used for training:
 1. A-Z Handwritten Alphabets [1]
  2. MNIST digits [2] 
  3. New dataset collected with help of friends (717 samples). It was collected directly via the Draw Scene in this application. It is available in the `Assets/Data` folder.


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<img src="https://github.com/Durostolar/unity-written-character-recognition/blob/master/Screenshots/collected_data.png" alt="drawing" width="400"/>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Example from the collected data


- **Preprocessing:** Everything is implemented here: image binarization, centering and re-scaling to 24x24. 
- **Augmentation** via rotation to enrich the most underrepresented classes. 
----------

### Evaluation and results
-   **Test accuracy: 93.89%** on standard test split.    
-   Collected dataset accuracy: *70.15%* due to variability in handwriting styles and pen thickness.    
-   Common misclassifications include visually similar symbols (e.g., **'0' vs 'O'** or **'5' vs 'S'** or **'2' vs 'Z'**).

----------

### References
[1] A-Z Handwritten Alphabets dataset (https://www.kaggle.com/datasets/sachinpatel21/az-handwritten-alphabets-in-csv-format/data)

[2] MNIST dataset by Yann LeCun (http://yann.lecun.com/exdb/mnist/)
