docker run -p 10000:8888 --gpus all --ipc=host --ulimit memlock=-1 -it --rm -v $(pwd)/:/workspace/trt_hovernet triton:test tail -f /dev/null
