# FastAI Practical Deep Learning - Lesson 1 Notes

## Overview

Lesson 1 demonstrates building an image classifier from scratch using FastAI. The complete pipeline covers data collection, organization, training, and inference for a bird vs forest classifier.

## Data Collection with DuckDuckGo

### Image Search Function

```python
from ddgs import DDGS
from fastcore.all import *

def search_images(keywords, max_images=200): 
    return L(DDGS().images(keywords, max_results=max_images)).itemgot('image')
```

**Key Components:**

- **L()**: FastAI's enhanced list class with additional methods
- **itemgot()**: Extracts the 'image' field from each search result dictionary
- **DDGS()**: DuckDuckGo search API for finding images

### Single Image Download

```python
from fastdownload import download_url
from fastai.vision.all import *

urls = search_images('bird photos', max_images=1)
download_url(urls[0], 'bird.jpg', show_progress=False)
im = Image.open('bird.jpg')
im.to_thumb(256,256)
```

## Data Organization

### Creating Folder Structure

```python
searches = 'forest', 'bird'
path = Path('bird_or_not')

for o in searches:
    dest = (path/o)
    dest.mkdir(exist_ok=True, parents=True)
    download_images(dest, urls=search_images(f'{o} photo'))
    time.sleep(5)
    resize_images(path/o, max_size=400, dest=path/o)
```

**Functions:**

- **Path()**: FastAI's pathlib wrapper for file operations
- **download_images()**: Batch download with error handling
- **resize_images()**: Standardizes image sizes for training
- **time.sleep(5)**: Rate limiting for API calls

### Data Validation

```python
failed = verify_images(get_image_files(path))
failed.map(Path.unlink)
len(failed)  # 3 corrupted images removed
```

**Key Functions:**

- **verify_images()**: Checks image integrity and removes corrupted files
- **get_image_files()**: Recursively finds all image files in directory
- **map()**: Applies function to each item in FastAI's L() list

## DataBlock and DataLoaders

### Creating the Data Pipeline

```python
dls = DataBlock(
    blocks=(ImageBlock, CategoryBlock), 
    get_items=get_image_files, 
    splitter=RandomSplitter(valid_pct=0.2, seed=42),
    get_y=parent_label,
    item_tfms=[Resize(192, method='squish')]
).dataloaders(path, bs=32)

dls.show_batch(max_n=6)
```

**DataBlock Parameters:**

- **blocks=(ImageBlock, CategoryBlock)**: Input type (images) and output type (categories)
- **get_items=get_image_files**: Function to collect all training data
- **splitter=RandomSplitter(valid_pct=0.2, seed=42)**: 80/20 train/validation split
- **get_y=parent_label**: Extract labels from folder names ('bird' or 'forest')
- **item_tfms=[Resize(192, method='squish')]**: Resize images to 192x192 pixels

## Transfer Learning with ResNet18

### What is ResNet?

**ResNet** (Residual Network) is a type of Convolutional Neural Network (CNN) architecture that revolutionized computer vision in 2015. It won the ImageNet competition and became the foundation for modern deep learning.

**Key Innovation - Skip Connections:** Traditional deep networks suffered from the "vanishing gradient" problem - as networks got deeper, they became harder to train. ResNet solved this with "skip connections" or "residual connections" that allow information to flow directly between layers.

```
Normal network: Input → Layer1 → Layer2 → Layer3 → Output
ResNet: Input → Layer1 → Layer2+Input → Layer3+Previous → Output
```

**Why ResNet18?**

- **18** refers to the number of layers (relatively shallow and fast)
- Perfect balance of accuracy and speed for learning
- Small enough to train quickly, powerful enough for real results
- Industry standard for prototyping and production

### Model Creation and Training

```python
learn = vision_learner(dls, resnet18, metrics=error_rate)
learn.fine_tune(3)
```

**What happens when we call vision_learner()?**

1. Downloads ResNet18 pre-trained on ImageNet (1.4M images, 1000 categories)
2. Removes the final layer (designed for 1000 categories)
3. Adds a new final layer for our problem (2 categories: bird vs forest)
4. Freezes early layers (they already know how to detect edges, shapes, textures)
5. Only trains the final layer initially

**Why Transfer Learning Works:**

- Early layers learn universal features (edges, textures, shapes)
- Later layers learn specific features (eyes, wings, leaves)
- We only need to retrain the "decision making" part for our specific task

### Training Results Explained

```
Initial training (frozen backbone):
epoch | train_loss | valid_loss | error_rate | time
0     | 1.085264   | 0.732065   | 0.324324   | 00:04

Fine-tuning (entire network):
epoch | train_loss | valid_loss | error_rate | time
0     | 0.289634   | 0.206621   | 0.054054   | 00:09
1     | 0.221042   | 0.230008   | 0.027027   | 00:09
2     | 0.159444   | 0.239604   | 0.027027   | 00:09
```

**What's happening in these two phases?**

**Phase 1 (1 epoch):** Only training the final layer

- Error rate drops from random (50%) to 32% in 4 seconds
- The pre-trained features are already useful!

**Phase 2 (3 epochs):** Fine-tuning the entire network

- Error rate drops to 2.7% (97.3% accuracy)
- All layers adjust slightly to our specific bird vs forest task

**Why such good results so quickly?**

- ResNet18 was trained on ImageNet, which includes many birds and nature images
- The model already "knows" what birds and trees look like
- We're just teaching it to distinguish between them

Final accuracy: **97.3%** - Professional quality results in under 1 minute of training!

## Model Inference

### Making Predictions

```python
is_bird, _, probs = learn.predict(PILImage.create('bird.jpg'))
print(f"This is a: {is_bird}.")
print(f"Probability it's a bird: {probs[0]:.4f}")

# Output:
# This is a: bird.
# Probability it's a bird: 0.9987
```

**Prediction Returns:**

- **is_bird**: Predicted class label
- **_**: Prediction index (unused)
- **probs**: Confidence scores for each class

## Core FastAI Concepts

### Transfer Learning Process

**Why not train from scratch?** Training a CNN from scratch would require:

- Millions of images
- Weeks of training time
- Expensive GPU resources
- Deep expertise in architecture design

**Transfer Learning shortcut:**

1. **Start with knowledge**: ResNet18 already learned from 1.4M ImageNet images
2. **Adapt**: Replace final layer for our specific task (bird vs forest)
3. **Fine-tune**: Adjust the entire network with small learning rates
4. **Result**: Professional accuracy in minutes instead of weeks

**What did ImageNet teach ResNet18?**

- **Low-level features**: Edges, textures, corners, color gradients
- **Mid-level features**: Shapes, patterns, object parts
- **High-level features**: Complex objects, spatial relationships

Since birds and forests were in ImageNet, our model already has relevant knowledge!

### DataBlock Architecture

The DataBlock provides a declarative way to define data pipelines:

- **What** your data looks like (ImageBlock, CategoryBlock)
- **Where** to find it (get_image_files)
- **How** to split it (RandomSplitter)
- **How** to label it (parent_label)
- **How** to transform it (Resize transformations)

### FastCore Utilities

- **L()**: Enhanced list with chainable operations
- **Path()**: Pathlib wrapper with additional methods
- **itemgot()**: Extract specific fields from dictionaries
- **map()**: Apply functions to list items

## Image Preprocessing

### Resize Methods

```python
# Squish: Fast but may distort aspect ratio
item_tfms=[Resize(192, method='squish')]

# Crop: Preserves aspect ratio, may lose image parts
item_tfms=[Resize(192, method='crop')]

# Pad: Preserves aspect ratio, adds padding
item_tfms=[Resize(192, method='pad')]
```

The example uses 'squish' for simplicity, trading some image distortion for consistent input sizes.

## FastAI Documentation - Key Objects and Methods

### DataLoaders Object

The DataLoaders class is the central object that combines training and validation data for model training. It's built on top of PyTorch's DataLoader with additional FastAI functionality.

#### Key Methods:

```python
# Display sample data with labels
dls.show_batch(max_n=6)

# Get a single batch for inspection
batch = dls.one_batch()

# Access individual dataloaders
train_dl = dls.train
valid_dl = dls.valid

# Check number of input features
n_features = dls.n_inp
```

**show_batch() Details:** show_batch is a type-dispatched function that shows decoded samples. The function automatically determines how to display data based on the data types (images show as a grid, text shows as text, etc.)

**Common Parameters:**

- `max_n`: Maximum number of samples to display
- `figsize`: Figure size for matplotlib display
- `nrows`, `ncols`: Grid layout for image display

### vision_learner Function

vision_learner is the main function for creating computer vision models with transfer learning. It automatically handles the complex setup of pre-trained models.

#### Key Parameters:

```python
learn = vision_learner(
    dls,                    # DataLoaders object
    resnet18,               # Architecture (resnet18, resnet50, etc.)
    metrics=error_rate,     # Metrics to track during training
    pretrained=True,        # Use pre-trained weights (default)
    cut=None,              # Where to cut the pre-trained model
    y_range=None,          # Output range for regression
    loss_func=None         # Custom loss function
)
```

**What vision_learner() Does:**

1. Downloads pre-trained weights if pretrained=True
2. Cuts the model at the appropriate layer (usually before final classification)
3. Adds a new head suitable for your number of classes
4. Sets up proper parameter groups for transfer learning
5. Configures the loss function based on your data type

### Learner Object Methods

#### Training Methods:

```python
# Fine-tune with best practices
learn.fine_tune(epochs)

# Train from scratch or continue training
learn.fit(epochs)

# One cycle training with learning rate scheduling
learn.fit_one_cycle(epochs, lr)

# Unfreeze all layers for fine-tuning
learn.unfreeze()
```

#### Prediction and Inference:

```python
# Predict single item
prediction, index, probabilities = learn.predict(PILImage.create('image.jpg'))

# Get predictions for validation set
preds, targets = learn.get_preds()

# Test time augmentation for better accuracy
tta_preds = learn.tta()
```

#### Model Analysis:

```python
# Learning rate finder
learn.lr_find()

# Show training history
learn.recorder.plot_losses()

# Model interpretation
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix()
interp.plot_top_losses(9)
```

#### Saving and Loading:

```python
# Export model for production
learn.export('model.pkl')

# Load exported model
learn_inference = load_learner('model.pkl')

# Save full training state
learn.save('checkpoint')

# Load training state
learn.load('checkpoint')
```

### Path and File Utilities

#### Common FastAI Path Functions:

```python
from fastai.vision.all import *

# Get all image files recursively
files = get_image_files(path)

# Verify images and remove corrupted ones
failed = verify_images(files)
failed.map(Path.unlink)  # Delete corrupted files

# Download and resize images
download_images(dest_folder, urls=image_urls)
resize_images(source_folder, max_size=400, dest=dest_folder)
```

### L() - FastAI's Enhanced List

L() is FastAI's enhanced list class that provides additional functionality beyond standard Python lists.

```python
# Create enhanced list
items = L(['item1', 'item2', 'item3'])

# Extract fields from list of dictionaries
urls = L(search_results).itemgot('image')

# Apply function to each item
results = items.map(some_function)

# Chain operations
processed = L(data).filter(condition).map(transform)
```

### DataBlock Advanced Features

#### Alternative Factory Methods:

FastAI provides multiple factory methods for different data organization patterns:

```python
# From folder structure
dls = ImageDataLoaders.from_folder(path, train='train', valid='valid')

# From filename patterns
dls = ImageDataLoaders.from_name_re(path, fnames, pattern)

# From function
dls = ImageDataLoaders.from_path_func(path, files, label_func)

# From CSV/DataFrame
dls = ImageDataLoaders.from_csv(path, csv_file, folder='images')
```

#### Transform Options:

```python
# Item transforms (applied to individual items)
item_tfms = [Resize(224)]

# Batch transforms (applied to entire batches, usually on GPU)
batch_tfms = [*aug_transforms(), Normalize.from_stats(*imagenet_stats)]
```

### Prediction Output Explained

When calling `learn.predict()`, you get a 3-tuple:

```python
prediction, index, probabilities = learn.predict(image)
```

- **prediction**: The decoded class name (e.g., 'bird', 'forest')
- **index**: The class index in the model's vocabulary (usually not needed)
- **probabilities**: Tensor of confidence scores for each class

This is the standard format across all FastAI applications - the first element is always the human-readable prediction, and the last is always the confidence scores.
