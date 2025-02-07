git fetch
git reset --hard origin/master

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"
mim install mmpose


