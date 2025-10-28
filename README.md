# Low-Power Person Detection for UAV Search and Rescue Operations

This repository contains the code and documentation for the `Low-Power Person Detection for UAV Search and Rescue Operations` project in TinyML for the Stockholm University by [Gabriele Pattarozzi](https://github.com/gpatta), [Timon Coucke](https://github.com/LGDTimtou) and [Jakub Schwenkbeck](https://github.com/JakubSchwenkbeck)




[Project Demo](https://github.com/user-attachments/assets/02fc9ed3-450b-45ba-8a88-bcdbbf37bd96)




## Instructions


### Model Optimization

Create a Python (3.11) virtual environment:

```bash
python3 -m venv opt_venv
```

Activate it:

```bash
source opt_venv/bin/activate
```

Install requirements:
```
pip install -r src/optimizing/requirements.txt
```

### Model Runtime

Create a Python (3.11) virtual environment that includes the system site packages for picamera2 support:

```bash
python3 -m venv --system-site-packages venv
```

Activate it:

```bash
source venv/bin/activate
```

Install requirements and force reinstall for incompatible system site packages:

```bash
pip install -r requirements.txt
```

