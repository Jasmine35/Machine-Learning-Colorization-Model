## COLORIZATION MODEL 

Training an AI model to convert Grayscale image to a plausible colorized version 

## Project Content:

## Datasets (Used for training, validation, and testing. Subsets are available for quicker experimentation):
- test_samples 
- train_samples
- valid_samples 

## Models:
- best_checkpoint.pth: model with epoch value of lowest L1 loss
- colorization_model.pth: FINAL model training with 50 epochs

## Scripts:
**train_colorization.py**: trains the model 
**infer_colorization.py**: given a colored image, willl remove the a and b channels (grayscale) and use colorization_model.pth to predict colorized version, for comparison
**eval_colorization.py**: evaluates the accuracy of the model with all the images in the dataset
**test.py**: test script to see if the normalization of the training and inference scripts match

## Results: 
**final_model_results**: shows results from training final model for 50 epochs

1. Install dependencies
pip install -r requirements.txt

2. Train for a new model:
# Train on 100 images only
python train_colorization.py --subset 100 --epochs 10

# With custom parameters
python train_colorization.py \
  --train train_samples \
  --val valid_samples \
  --subset 200 \
  --epochs 5 \
  --batch_size 4 \
  --lr 1e-4

**Note**: datasets not provided because of large file size 
Example images provided in test_samples

3. Testing the model:
python infer_colorization.py --input your_image.jpg --output colorized.jpg


4. Evaluating the model:
python eval_colorization.py --model colorization_model.pth --data_folder test_samples



## Key Features
- Hybrid ResNet34-U-Net architecture
- Lab color space separation (L channel input → ab channel prediction)
- L1 + optional perceptual loss for realistic color generation
- Gradient clipping & learning rate scheduling for stable training
- Post-processing options (sharpening, color boosting)

## Datasets
To download the DataSets use this link: https://www.kaggle.com/datasets/pankajkumar2002/places365 
