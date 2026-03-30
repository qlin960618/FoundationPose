IMAGE_NAME=${IMAGE_NAME:-foundationpose:cu128}
CONTAINER_NAME=${CONTAINER_NAME:-foundationpose}
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
DIR=$(cd -- "${SCRIPT_DIR}/.." && pwd)
DEVICE_ARGS=()
DISPLAY_ARGS=()
FOUND_VIDEO_DEVICES=()
FOUND_MEDIA_DEVICES=()

docker rm -f "${CONTAINER_NAME}" 2>/dev/null || true

if [ -n "${DISPLAY}" ]; then
  xhost +local:docker >/dev/null 2>&1 || true
  DISPLAY_ARGS+=(-v /tmp/.X11-unix:/tmp/.X11-unix -e "DISPLAY=${DISPLAY}")
fi

shopt -s nullglob
for device in /dev/video* /dev/media*; do
  if [ -e "${device}" ]; then
    DEVICE_ARGS+=(--device="${device}:${device}")
    case "${device}" in
      /dev/video*)
        FOUND_VIDEO_DEVICES+=("${device}")
        ;;
      /dev/media*)
        FOUND_MEDIA_DEVICES+=("${device}")
        ;;
    esac
  fi
done
shopt -u nullglob

if [ ${#FOUND_VIDEO_DEVICES[@]} -gt 0 ]; then
  echo "Found webcam video devices: ${FOUND_VIDEO_DEVICES[*]}"
else
  echo "Found webcam video devices: none"
fi

if [ ${#FOUND_MEDIA_DEVICES[@]} -gt 0 ]; then
  echo "Found webcam media devices: ${FOUND_MEDIA_DEVICES[*]}"
fi

docker run --gpus all --env NVIDIA_DISABLE_REQUIRE=1 -it --network=host \
  --name "${CONTAINER_NAME}" \
  --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  --ipc=host \
  "${DEVICE_ARGS[@]}" \
  -e GIT_INDEX_FILE \
  -e TORCH_CUDA_ARCH_LIST=12.0 \
  -v "$DIR:$DIR" \
  -v /home:/home \
  -v /mnt:/mnt \
  -v /tmp:/tmp \
  "${DISPLAY_ARGS[@]}" \
  "${IMAGE_NAME}" \
  bash -c "cd $DIR && bash"
