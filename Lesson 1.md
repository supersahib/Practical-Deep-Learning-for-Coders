

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

## Results Summary

- **Data**: ~400 images per class (after removing 3 corrupted files)
- **Architecture**: ResNet18 with transfer learning
- **Training**: 3 epochs of fine-tuning
- **Performance**: 97.3% accuracy
- **Speed**: Under 10 seconds on GPU, few minutes on CPU