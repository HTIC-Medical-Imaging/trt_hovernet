docker run -p 8888:8888 --gpus all --ipc=host --ulimit memlock=-1 -it --rm -v $(pwd)/:/workspace/trt_hovernet tensorrt:test tail -f /dev/null
