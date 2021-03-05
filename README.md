# Adversarial attacks & defence strategies benchmark

This project hosts several notebooks and snippets used to build [our benchmark](https://medium.com/meetech/adversarial-learning-benchmark-evaluating-attack-and-defence-strategies-48085ab3ac3).

## Getting started
You will need to use a UNIX-based server with:
 - [python 3](https://www.python.org/downloads/)
 - GNU Make
 - [Jupyter](https://jupyter.org/install)
 - [Jupytext](https://jupytext.readthedocs.io/en/latest/install.html)

You can then run the following commands:

```bash
# sudo apt-get install build-essential  # to install GNU Make if necessary
git clone https://github.com/cedricgoubard/benchmark-adversarial-attacks.git
cd benchmark-adversarial-attacks
make install
```

Once everything is installed, you need to create you configuration file. You can copy `config.yaml.example` and simply replace the defaut values with your own. 
```bash
cp config.yaml.example config.yaml
```

You are now ready to open any notebooks from the `notebooks` folder.

