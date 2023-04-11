mkdir ~/app
mv * ~/app
cd ~/app
python -m pip install --upgrade pip
pip install -r requirements.txt
dvc pull