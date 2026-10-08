"""Train a Green kernel for the Dirichlet operator -Delta - 30."""

import os

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

from neuralainv.training import main

if __name__ == "__main__":
    main()
