# Interferometric Lensless Imaging 
## Rank-One Projections of image frequencies with Speckle Illuminations

---

[![image](https://img.shields.io/badge/License-MIT-blue.svg?style=flat)](https://github.com/olivierleblanc/interferometric_lensless_imaging/blob/master/LICENSE) [![image](https://img.shields.io/badge/arXiv-2306.12698-red.svg?style=flat)](https://arxiv.org/abs/2306.12698)  


## Repository Structure

The repository is organized as follows:
```
|- images/                      images to be reconstructed.
|- paper_experiments/           code that generated figures 2,3, and 6 of the paper.
|- reconstructions_demo/        some notebooks that analyze the reconstructions dependency on various parameters.
|- tests_and_visualizations/    tests, visualizations, and additional code examples.
|- utils/                       utility functions and modules.
|- demo.ipynb                   Jupyter Notebook demonstrating the project functionalities.
```

## Installation and Usage

To use the code in this repository, follow these steps:

1. Clone this repository using [``git clone``](https://docs.github.com/fr/repositories/creating-and-managing-repositories/cloning-a-repository) or download the zip file.

2. Create a minimal virtual environment and activate it. We suggest [conda](https://docs.conda.io/en/latest/), if you want to use [pipenv](https://pipenv.pypa.io/en/latest/) instead, replace step 2. below by the instructions in [pipenv.md](pipenv.md)
. With ``conda``, use the following commands:
```
conda create --name your_env_name python==3.8.8
conda activate your_env_name
pip install -r requirements.txt
```

This project uses old versions of Python, scipy, and other packages. ``requirements.txt`` ensures you will run the code in a controlled environment.

**Note:** to see the newly create venv, you may need to relaunch your python editor.

To deactivate the environment when you don't want to work on this project:
```
conda deactivate
```

3. install my forks of ``pyunlocbox``, ``pyproximal`` and ``pylops`` from Github: 
   * clone each repo from Github or download the zip file
    ```
    git clone https://github.com/olivierleblanc/pyunlocbox.git 
    ```
    * then in command line, ensure the created virtual env is activated, then pip install the repo from the fork you just downloaded.
    ```
    pip install -e /path/to/repo
    ```

    repeat with <br>
    https://github.com/olivierleblanc/pyproximal.git and https://github.com/olivierleblanc/pylops.git 

4. Run the demo Jupyter Notebook (`demo.ipynb`) to see an example of how to use the project functionality.
5. Explore the project files and directories to find the desired code and resources.

## Data 

The experimental data can be download
<a href="https://drive.google.com/drive/folders/1fYSA78RPlp3rA9Baj2oCMKVLMHoInO-Y?usp=share_link"> here
<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/1/12/Google_Drive_icon_%282020%29.svg/768px-Google_Drive_icon_%282020%29.svg.png" width="30px" align="center" >
</a>

## Contributions

Contributions to this project are welcome. If you find any issues or want to add new features, please open an issue or submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE).

## Contact

Don't hesitate to discuss with me eiter on Github directly or email me at o.leblanc@uclouvain.be.


> [!NOTE]
In the code, you sometimes used the word *diagless* or *dl* that is a synonym of *hollow* matrix. It is related to the debiasing trick explained in the paper.


