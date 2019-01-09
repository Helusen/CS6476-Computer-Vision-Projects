# CS 6476 project 3: [Camera Calibration and Fundamental Matrix Estimation with RANSAC](https://www.cc.gatech.edu/~hays/compvision/proj3/)

# Setup
- Install [Miniconda](https://conda.io/miniconda). It doesn't matter whether you use 2.7 or 3.6 because we will create our own environment anyways.
- Create a conda environment using the given file by modifying the following command based on your OS (`linux`, `mac`, or `win`): `conda env create -f environment_<OS>.yml`
- This should create an environment named `cs6476`. Activate it using the following Windows command: `activate cs6476` or the following MacOS / Linux command: `source activate cs6476`.
- Run the notebook using: `jupyter notebook ./code/proj3.ipynb`
- Generate the submission once you're finished using `python zip_submission.py`

In the code i provided, I set the iteration time in RANSAC as 500, and the threshold as 0.06. But actually I tested different iteration time and threshold in different datasets. And I also tested the non-normalized data in calculating projection matrix and camera center. And I only randomly output 50 epipolars in my index.html. Basically if you just need to verify the correctness of my program, you can just run it without any input rectification.
