# Argus

## Setup (laptop)

```bash
make venv       # creates .venv (Python >= 3.10)
make install    # pip install -e '.[api,dev]'
make test
cp .env.example .env   # set ARGUS_DEVICE=mps on Apple Silicon, cpu otherwise
```

## Setup (Colab)

```python
from google.colab import drive
drive.mount('/content/drive')
!git clone https://github.com/AnomaSight/Argus.git
%cd /content/Argus
!mkdir -p /content/drive/MyDrive/argus/{data,artifacts}
!rm -rf data artifacts && ln -s /content/drive/MyDrive/argus/data data && ln -s /content/drive/MyDrive/argus/artifacts artifacts
!make install && make test
```

Colab's preinstalled CUDA torch is kept (only lower bounds are pinned). Keep `ARGUS_DEVICE=cuda`.
