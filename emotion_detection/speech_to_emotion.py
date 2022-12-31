"""
1) Build data loaders for each dataset
    i) extract emotion: have a global dict that holds the emotions you can detect; many-to-one relationship for different
    keys from different datasets that correspond to the same emotion
    ii) ensure wav file
    iii) essentially; iterate through each file in dataset, extract emotion key from relevant dict, and pass file for training
    along with the extracted emotion - core functionality of data loader
    iv) useful to have all files in one big dataset folder for easy training and shuffling later
2) Build a training pipeline utilising the different datasets
    i) pipeline should allow for training with subsets of each dataset with shuffling enabled
    ii) pipeline should allow for training with particular datasets with input definitions; perhaps model datasets
    as keys in a dict and you can pass a list of strings corresponding to the datasets you want to use for training
3) Build lstm model as in video
4) Train model
    i) train with a subset of one dataset to start with and benchmark performance
    ii) train with each dataset and benchmark performance
    iii) train with entire dataset and benchmark performance
5) Save models as pickle files
    i) have models corresponding to the different training processes
5) Build function to obtained pickled model from directory
6) Use model to predict sales call data
7) Look into building multiple classifier to aggregate predictions to determine if an ensemble performs better than a
single model
"""