# CS 6476 project 5: [Face Detection with a Sliding Window](https://www.cc.gatech.edu/~hays/compvision/proj5/)

# Setup
- Install [Miniconda](https://conda.io/miniconda). It doesn't matter whether you use 2.7 or 3.6 because we will create our own environment anyways.
- Create a conda environment, using the appropriate command. On Windows, open the installed "Conda prompt" to run this command. On MacOS and Linux, you can just use a terminal window to run the command. Modify the command based on your OS ('linux', 'mac', or 'win'): `conda env create -f environment_<OS>.yml`
- This should create an environment named `cs6476p5`. Activate it using the following Windows command: `activate cs6476p5` or the following MacOS / Linux command: `source activate cs6476p5`.
- Run the notebook using: `jupyter notebook ./code/proj5.ipynb`
- Generate the submission once you're finished using `python zip_submission.py`

For the extra credits, I implemented multiple scales in function mine_hard_negs() and get_random_negative_features() to improve the accuracy. You can just run the program using starter code. It takes some time to run run_detector() funtcion since I used multiple scales in it, and set the step size as 10 to abtain an accuracy above 0.8.
