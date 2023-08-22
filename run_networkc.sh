docker run --rm --mount type=bind,source="$(pwd)"/output,target=/workspace/output \
 --mount type=bind,source="$(pwd)"/unlabeled2017,target=/workspace/unlabeled2017 \
 --mount type=bind,source="$(pwd)"/results.txt,target=/workspace/results.txt \
 --gpus all \
 -p 8000:8000\
 --ipc=host \
 --pid=host \
 -v /etc/localtime:/etc/localtime \
 -it networkc