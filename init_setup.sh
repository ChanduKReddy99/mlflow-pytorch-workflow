conda create -p .venv python=3.10 -y
conda activate .venv/
pip install -r requirements.txt
conda install pytorch torchvision torchaudio cpuonly -c pytorch -y
conda env export > conda.yaml