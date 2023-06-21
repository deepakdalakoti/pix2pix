## Computer vision: Pixel-2-Pixel

Datasets:
1. [Facades](https://cmp.felk.cvut.cz/~tylecr1/facade/): 606 images
2. [Cityscapes](https://www.cityscapes-dataset.com/): 20000 images
3. [Shoes](https://vision.cs.utexas.edu/projects/finegrained/utzap50k/): 50K images

Datasets will need to be downloaded and placed in appropriate directory. Use DATA_PATH to place larger files.

Type `make up` to stand up the containers. Make sure to fill and copy `default_env` file to `.env` file. This will stand up four containers:

1. app-pix2pix: A container to train models and run inferences.
2. notebook-pix2pix: A container which runs a jupyter notebook.
3. logs-pix2pix: A container which runs tensorboard for tracking training.
4. mlflow-pix2pix: A container which runs mlflow ui for tracking/versioning model. By default all logs are saved to file in a bind volume mounted in containers which persists data even if containers are removed.

For uniqueness, users' LAN ID is appended to the end of the containers names.

Very minimal libraries are installed by default. It's left to users to install additional libraries as required.

Note: If running on cluster, forward relevant ports (as defined in .env file) to access notebook and tensorboard.

An example script is provided on how to use mlflow (`src/mlflow_example.py`). By default mlflow will write in /data directory which is mounted in the container. 

## Resources

1. [MLflow](https://mlflow.org/docs/latest/index.html): MLflow documentation.
2. [Pix2Pix Paper](https://arxiv.org/pdf/1611.07004.pdf): Pix2Pix paper discussed in meeting. 
3. [Original Pix2Pix Implementation](https://github.com/phillipi/pix2pix): Pix2Pix implementation by paper authors.
4. [PyTorch](https://pytorch.org/): PyTorch