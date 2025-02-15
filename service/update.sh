git fetch
git reset --hard origin/master

python3.10 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

pip install -U openmim
mim install mmengine
mim install "mmcv==2.1.0"
mim install mmpose


