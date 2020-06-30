"""Global constants shared throughout the code."""

# Truncation (in voxels).
TRUNCATION = 3

# Number of classes.
NUM_CLASSES = 13

# Class weights for training SUNCG data.
WEIGHT_CLASSES = [
    0.1, 2.0, 0.4, 2.0, 0.4, 0.6, 0.6, 2.0, 2.0, 2.0, 0.4, 0.5, 0.1
]

# Number of classes.
NUM_CLASSES = 14

# Class weights for training SUNCG data.
WEIGHT_CLASSES = [
    0.1, 2.0, 0.4, 2.0, 0.4, 0.6, 0.6, 2.0, 2.0, 2.0, 0.4, 0.5, 0.1, 0.1
]



# Number of classes.
NUM_CLASSES = 12
# Class weights for ScanNetScan2CAD
WEIGHT_CLASSES = [
    0.5825622 , 0.82839054, 0.74294346, 0.73363114, 0.85788301,
       0.76457722, 0.78067847, 0.80523654, 0.75675786, 1.        ,
       0.74430139, 0.80288971
    ]