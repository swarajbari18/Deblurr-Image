deactivate:
	deactivate

deployment:
	streamlit run app.py

setup:
	sudo update -y
	sudo apt update -y
	python -m pip install --upgrade pip
	pip install -r requirements.txt


