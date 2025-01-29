python3 -m venv venv  
source venv/bin/activate

sudo systemctl daemon-reload
sudo systemctl start flask_app.service
sudo systemctl status flask_app.service


