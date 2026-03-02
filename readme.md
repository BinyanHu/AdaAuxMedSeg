# Code for Review

This repository contains the reference implementation of our proposed method in the paper.
The code is provided for review purposes only and is not a complete, runnable package.
The primary goal of this repository is to allow reviewers to inspect the core algorithmic logic and implementation details of our method.


## Repository Structure

Folder `data/` contains utensils for constructing data input pipeliens for segmentation and auxiliary tasks

Folder `losses/` contains loss functions for segmentation and auxiliary tasks

Folder `tasks/` contains the core implementations of our proposed method:
- `medical_image_segmentation.py`
Base class for single-task (segmentation only) training.
- `multi_auxiliary_task_learning.py`
Base class to support joint learning with multiple auxiliary tasks.
- `joint_loss.py`
Implementation the Joint-Loss baseline, which trains a parameter-shared model with the sum of all task losses at every iteration step.
- `ours.py`
Implementation of our method.

`main.py` is intended as the entry point for experiments. Currently configured to run Ours by default.
