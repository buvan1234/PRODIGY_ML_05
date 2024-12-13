

# Food Item Recognition and Calorie Estimation Model

This project uses deep learning to create a model that can accurately recognize food items from images and estimate their calorie content, enabling users to track their dietary intake and make informed food choices.

## Overview

The goal of this project is to develop a deep learning model that:
1. Classifies food items from images.
2. Estimates the calorie content of each recognized food item.

We use TensorFlow and Keras for building the model, and the dataset utilized is **Food-101**, which contains images of 101 food categories.

## Features
- **Food Recognition**: The model is trained to recognize over 100 different food categories.
- **Calorie Estimation**: The model estimates the calorie content of the recognized food item.
- **Visualization**: The dataset is visualized to display random images from each class of food.
  
## Setup Instructions

### Requirements

- Python 3.7+
- TensorFlow 2.x
- Keras
- OpenCV
- Matplotlib
- NumPy
- Other dependencies specified in `requirements.txt`.

### Installing Dependencies

1. Clone this repository:
    
    git clone https://github.com/buvan1234/PRODIGY_ML_05.GIT
    

2. Create and activate a virtual environment (optional but recommended):
    
    python3 -m venv env
    source env/bin/activate  # For Mac/Linux
    env\Scripts\activate     # For Windows
  

3. Install the required dependencies:
    
    pip install -r requirements.txt
    

### Dataset

The model uses the **Food-101 dataset**, which can be downloaded directly from the official source. You can also use the `get_data_extract()` function to download and extract the data automatically.

```python
def get_data_extract():
    if "food-101" in os.listdir():
        print("Dataset already exists")
    else:
        print("Downloading the data...")
        !wget http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz
        print("Dataset downloaded!")
        print("Extracting data..")
        !tar xzvf food-101.tar.gz
        print("Extraction done!")
```

The dataset consists of images categorized into 101 food classes, and we use this data to train the model.

### Model Overview

1. **Image Preprocessing**: Images are resized, normalized, and augmented to improve model performance.
2. **Model Architecture**: We use a pre-trained model like **InceptionV3** and fine-tune it for food classification and calorie prediction.
3. **Training**: The model is trained to classify food items and estimate calorie content using a **multitask learning** approach, where both classification and regression tasks are learned together.
4. **Calorie Prediction**: For each recognized food item, the model predicts the calorie content based on its class.

### Visualizing Dataset

The dataset is visualized with one image from each of the 101 classes:

```python
rows = 17
cols = 6
fig, ax = plt.subplots(rows, cols, figsize=(25,25))
fig.suptitle("Showing one random image from each class", y=1.05, fontsize=24)
data_dir = "food-101/images/"
foods_sorted = sorted(os.listdir(data_dir))
food_id = 0
for i in range(rows):
    for j in range(cols):
        try:
            food_selected = foods_sorted[food_id]
            food_id += 1
        except:
            break
        if food_selected == '.DS_Store':
            continue
        food_selected_images = os.listdir(os.path.join(data_dir, food_selected))
        food_selected_random = np.random.choice(food_selected_images)
        img = plt.imread(os.path.join(data_dir, food_selected, food_selected_random))
        ax[i][j].imshow(img)
        ax[i][j].set_title(food_selected, pad = 10)
plt.setp(ax, xticks=[], yticks=[])
plt.tight_layout()
```

### Running the Model

Once you have the dataset, you can proceed to train the model. Here's an example code snippet for training:

```python
# Example of food classification model with InceptionV3
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras import models
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D

base_model = InceptionV3(weights='imagenet', include_top=False)
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
x = Dense(num_classes, activation='softmax')(x)  # Modify based on your food categories
model = models.Model(inputs=base_model.input, outputs=x)

# Compile and train the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_data, epochs=10, steps_per_epoch=steps_per_epoch)
```

### Calorie Prediction Model

A regression head can be added for calorie estimation:

```python
# Add a regression head for calorie prediction
x = Dense(1, activation='linear')(x)  # Predict continuous calorie value
model = models.Model(inputs=base_model.input, outputs=x)

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
```

### Evaluation

Once trained, the model can be evaluated using:
- **Classification accuracy** for food recognition.
- **Mean Squared Error (MSE)** or **Mean Absolute Error (MAE)** for calorie estimation.

