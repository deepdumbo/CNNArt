# constants labeling modes
MASK_LABELING = 0
PATCH_LABELING = 1

# constants patching modes
PATCHING_2D = 0
PATCHING_3D = 1

# constants splitting modes
NONE_SPLITTING = 0
SIMPLE_RANDOM_SAMPLE_SPLITTING = 1
CROSS_VALIDATION_SPLITTING = 2
PATIENT_CROSS_VALIDATION_SPLITTING = 3
DIY_SPLITTING = 4

# constants storage mode
STORE_DISABLED = 0
STORE_HDF5 = 1
STORE_PATCH_BASED = 2

# optimizer constants
SGD_OPTIMIZER = 0
RMS_PROP_OPTIMIZER = 1
ADAGRAD_OPTIMIZER = 2
ADADELTA_OPTIMIZER = 3
ADAM_OPTIMIZER = 4

# Data Augmentation Parameters
WIDTH_SHIFT_RANGE = 0.2
HEIGHT_SHIFT_RANGE = 0.2
ROTATION_RANGE = 30
ZOOM_RANGE = 0.2