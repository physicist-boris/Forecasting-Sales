conda create -n "forecasting-sales" python=3.11.0
conda activate forecasting-sales
cd src
pip install -r ./env/requirements-dev.txt
pre-commit install
$env:PYTHONPATH = $pwd