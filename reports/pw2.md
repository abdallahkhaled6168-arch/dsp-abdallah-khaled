# PW2 - Model Industrialization

## Overview
In this project, I worked on building a machine learning pipeline to predict house prices. The main goal was not only to train a model, but also to organize the project in a clean and structured way that looks closer to a real-world system.

## Project Structure
I divided the project into different folders to make it more organized:
- data/: contains the training and testing datasets
- house_prices/: includes the main code (preprocessing, training, and inference)
- models/: used to store the trained model
- notebooks/: contains the final notebook that runs the pipeline
- reports/: includes this report

## Training
For training, I used a Linear Regression model.  
I split the data into training and testing sets using an 80/20 ratio.  
The model was evaluated using Mean Absolute Error (MAE), which helps measure how far predictions are from actual values.

## Inference
After training the model, I loaded it and used it to generate predictions on the test dataset.  
The predictions are saved so they can be reused later.

## Industrialization
To make the project more realistic and organized, I:
- separated training and inference into different files
- used functions to make the code reusable
- saved the trained model using joblib

## Conclusion
Overall, this project helped me understand how to structure a machine learning pipeline properly.  
Instead of writing everything in one notebook, I organized the code into modules, which makes it easier to maintain and reuse.
