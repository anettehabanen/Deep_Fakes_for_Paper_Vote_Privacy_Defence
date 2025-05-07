# Deep_Fakes_for_Paper_Vote_Privacy_Defence

This is the repository for my master thesis called "Deep Fakes for Paper Vote Privacy Defence".

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
git clone https://github.com/anettehabanen/Deep_Fakes_for_Paper_Vote_Privacy_Defence
cd Deep_Fakes_for_Paper_Vote_Privacy_Defence
```

Then create an environment and install requirements:
```
python -m venv deep_fakes_venv
source deep_fakes_venv/bin/activate
pip install -r requirements.txt
```
