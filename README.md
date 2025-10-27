# low-power-person-detection-uav-sar

Low-Power Person Detection for UAV Search and Rescue Operations


https://github.com/user-attachments/assets/8252d80d-e621-4c57-8b4e-a60b99bf5c8f


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

