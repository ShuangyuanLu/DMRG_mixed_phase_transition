import os


def set_single_thread_env():
    os.environ.update({
        "OPENBLAS_NUM_THREADS":"1",
        "MKL_NUM_THREADS":"1",
        "OMP_NUM_THREADS":"1",
        "MKL_DYNAMIC":"FALSE",
        "OPENBLAS_WAIT_POLICY":"PASSIVE",
        "OMP_WAIT_POLICY":"PASSIVE",
        "KMP_BLOCKTIME":"0",
    })