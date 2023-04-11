docker run -p 8888:8888 --gpus all --ipc=host --ulimit memlock=-1 -it --rm -v $(pwd)/:/workspace -v /storage/users/dockeruser/data:/data torch_starter:base tail -f /dev/null
