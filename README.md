<h1 style="text-align: center;">AnimalVision-DirectML</h1>

<h3 style='text-align:center'>Animal Image Classification Project built using PyTorch and DirectML backend</h3>
<hr>

[<img src="ignore\torch_dml.jpg">](https://github.com/microsoft/DirectML)

![PyPI - Python Version](https://img.shields.io/pypi/pyversions/torch?style=flat-square) ![GitHub License](https://img.shields.io/github/license/Gurneet1928/AnimalVision-DirectML?style=flat-square) ![GitHub last commit](https://img.shields.io/github/last-commit/Gurneet1928/AnimalVision-DirectML?style=flat-square) ![GitHub commit activity](https://img.shields.io/github/commit-activity/w/Gurneet1928/AnimalVision-DirectML?style=flat-square) ![GitHub Downloads (all assets, all releases)](https://img.shields.io/github/downloads/Gurneet1928/AnimalVision-DirectML/total)


The Animal Image Classification Project leverages PyTorch for its deep learning framework, utilizing the DirectML backend to optimize performance across various hardware AMD Devices. This project aims to accurately classify images of different animal species using a ResNet18 Model. The model is trained on Animal10 dataset and can distinguish between multiple animal classes with high accuracy. This project serves as an excellent example using AMD devices for deep learning purpose, dedicated to Windows users.


#### The Project implements a 3 Stage Pipeline

## Table of Contents

1. Features
2. Tech Stack Used
3. Installation and Usage
4. Result Comparision
5. Future Improvements
6. Development and Contributions
7. License
8. Aftermath

## Features
 - 3 Stage Pipeline, allows to execute complete code in single command (Data Ingestion -> Model Training -> Model Inference)
 - Added logs for debugging and future purpose
 - Uses [Animal10 Dataset from Kaggle](https://www.kaggle.com/datasets/alessiocorrado99/animals10)
 - Leaverages DirectML for utilizing the capabilites of AMD GPUS
 - Seperate file for model hyperparameters, for easy tuning

## Tech Stack Used:
- PyTorch, w/ TorchVision
- DirectML
- Logger
- Opendatasets
and few more from requirements.txt file

## Installation

For the code to work, make sure to install [PyTorch](https://pytorch.org/get-started/locally/) and [Torch-DirectML](https://github.com/microsoft/DirectML)

For the dataset to be downloaded, make sure to get Kaggle API keys in **kaggle.json** file (in same folder). The project will automatically load the file from folder. Don't worry, the file will never be shared and will stay on your local machine.

Incase you face issues while installing Torch-DirectML, follow this >> [Installation Article](https://www.linkedin.com/feed/update/urn:li:activity:7202695984961785856/)

Run the below codes after making sure you install PyTorch and Torch-DirectML properly.

1. Clone the repository
```bash
git clone https://github.com/Gurneet1928/AnimalVision-DirectML.git
```

2. Install Requirements using:
```bash
pip install -r requirement.txt --use-deprecated=legacy-resolver
```

3. Run the pipeline using:
```bash
python main.py
```

The Pipeline will begin by downloading dataset. Incase dataset is already present, this step will be skipped.

Model will be trained once dataset is available.
After Training, the best model (w/ best epoch) will be saved and used for inferencing.

## Result Comparison

...............

## Future Improvements

Here are some thoughts regarding what can be added in future:
- Command Line arguements for which stage to skip or execute, inference file path, etc.
- Frontend using Streamlit or Gradio 
- Custom Built Image Classification model (soon)
- A benchmark file to add more benchmark results

## Development and Contributions

Have some ideas to make this project better? Or found a bug? Maybe tested on your device and want to share results? 
Feel free to add an issue on this repo and I will personally look onto this.

Or mail me to gurneet222@gmail.com

## License

MIT License

Distributed under the License of MIT, which provides permission to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software. Check LICENSE file for more info.

OR
Free to use
But please make sure attribute the developer....

**Free Software, Hell Yeah!**

### Reached the End ? I appreciate you reading this README in its entirety (maybe). Please remember to give this software a star if you found it useful in any way.   (❁´◡`❁)   (●ˇ∀ˇ●)

## Aftermath

AMD Users after learning about DirectML and now they can use AMD GPUs for Deep Learning....
![alt text](image.png)




