install: #not usefull when using containers (docker)
	python3 -m venv venv
	venv/bin/pip install -U pip
	venv/bin/pip install -r requirements.txt
	
	venv/bin/python -m ipykernel install --user --name=benchmark --display-name="BENCHMARK"

uninstall:
	venv/bin/python -m jupyter kernelspec uninstall hmai
	rm --force --recursive venv/

