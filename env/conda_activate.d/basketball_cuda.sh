# 使 onnxruntime-gpu 能找到 conda 安装的 libcudnn
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
