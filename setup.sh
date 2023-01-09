docker run -p 9000:8888 --gpus all --ipc=host --ulimit memlock=-1 -it --rm -v $(pwd)/:/workspace/trt_hovernet tensorrt:test
