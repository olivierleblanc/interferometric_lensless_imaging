If you want to use [pipenv](https://pipenv.pypa.io/en/latest/) instead of [conda](https://docs.conda.io/en/latest/), replace step 2. in [README.md](README.md) by these steps.

You may need to install ``pipenv`` if it is the first time:
```
pip install pipenv
```

1. Create a minimal virtual environment and activate it using the following commands:
```
cd /path/to/interferometric_lensless_imaging/  # naviguate to the repo location
pipenv --python python==3.8.8                  # create the virtual environment
pipenv shell                                   # activate the env
pip install -r requirements.txt                # install the dependencies (except pyunlocbox, pylops and pyproximal)
```

**Note:** to see the newly create venv, you may need to relaunch your python editor.

To deactivate the environment when you don't want to work on this project:
```
pipenv shell --deactivate
```