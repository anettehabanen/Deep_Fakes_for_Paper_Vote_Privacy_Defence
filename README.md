# Deepfakes_for_Paper_Vote_Privacy_Defence

This is the repository for my master thesis called "Deepfakes for Paper Vote Privacy Defence".

## Contents of the repository

* creating_datasets - Script for downloading and creating datasets used in this project.
* dataset - Example datasets for training models and running the pipelines
* pipelines - Resulting pipeline to change to 3-digit code in a video. One script is for only covering up the old digit, the other one adds also the new 3-digit code.
* training_WavePaint - Code to train the WavePaint model. (includes model weights)
* training_YOLO - Code to train the YOLO model. (includes model weights)

Because Github has a file size limit, then the collected dataset of voting ballot images and videos can be found [here](https://drive.google.com/file/d/1ETJoOAPoZNsUJW-IezeUcsXR-c4XBuTV/view?usp=sharing).


## Running the repository

First clone the repository:
```
git clone https://github.com/anettehabanen/Deepfakes_for_Paper_Vote_Privacy_Defence.git
cd Deepfakes_for_Paper_Vote_Privacy_Defence
```

Then create an environment and install requirements:
```
python -m venv deepfakes_venv
source deepfakes_venv/bin/activate
pip install -r requirements.txt
```

Most of my files are Jupyter Notebook files. To run these:
```
pip install jupyterlab
jupyter lab
```

## Reflex app

This project has also a Reflex app implementation. This can be found in a separate repository [here](https://github.com/anettehabanen/Change_My_Vote.git).
