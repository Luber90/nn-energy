docker run --rm --mount type=bind,source="$(pwd)"/model_output,target=/workspace/output \
 --mount type=bind,source="$(pwd)"/unlabeled2017,target=/workspace/unlabeled2017 \
 --gpus all \
 -p 8000:8000\
 --ipc=host \
 -it network