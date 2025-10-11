# low-power-person-detection-uav-sar

Low-Power Person Detection for UAV Search and Rescue Operations

## Instructions


### Model Conversion

Create a Python (3.11) virtual environment:

```bash
python3 -m venv conversion_venv
```

Activate it:

```bash
source conversion_venv/bin/activate
```

Install requirements:

```bash
pip install -r src/conversion/requirements.txt
```


### Model Runtime

Create a Python (3.11) virtual environment that includes the system site packages for picamera2 support:

```bash
python3 -m venv --system-site-packages runtime_venv
```

Activate it:

```bash
source runtime_venv/bin/activate
```

Install requirements and force reinstall for incompatible system site packages:

```bash
pip install --force-reinstall -r src/runtime/requirements.txt
```

