# gnn-test

Pre install cuda version 10.1

Follow the below steps to create a virtual environment with python 3.8


```bash
python3 -m venv /path/to/virtual/environment

# activate the virtual environment
source /path/to/virtual/environment/bin/activate

# inside the virtual environment execute the below command to install the required packages
python -m pip install -r requirements.txt
```

Install the relevant pytorch-geometric libraries
```bash
python -c "import torch; print(torch.__version__)"
python -c "import torch; print(torch.version.cuda)"

pip install torch==1.7.0
pip install scipy==1.6.0
pip install --no-index torch-scatter -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install --no-index torch-sparse -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install --no-index torch-cluster -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install --no-index torch-spline-conv -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install torch-geometric

pip install --no-index torch-scatter -f https://pytorch-geometric.com/whl/torch-1.7.0+cu102.html
pip install --no-index torch-sparse -f https://pytorch-geometric.com/whl/torch-1.7.0+cu102.html
pip install --no-index torch-cluster -f https://pytorch-geometric.com/whl/torch-1.7.0+cu102.html
pip install --no-index torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.7.0+cu102.html
pip install torch-geometric
```
