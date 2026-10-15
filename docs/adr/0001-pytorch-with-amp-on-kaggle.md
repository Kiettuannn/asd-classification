# Use PyTorch with AMP for Kaggle 3D-CNN Training

We decided to implement the 3D-CNN fMRI classification pipeline in PyTorch with Automatic Mixed Precision (AMP) rather than Keras, despite the original paper mentioning Keras. 3D volumetric convolutional networks operating on 150-slice multi-view tensors have high GPU memory footprints. PyTorch provides granular memory allocation control, native `torch.amp` fp16 acceleration on Kaggle GPUs (NVIDIA T4 / P100), and efficient `torch.utils.data.DataLoader` multiprocessing.
