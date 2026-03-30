DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

cd "${DIR}/mycpp/" && mkdir -p build && cd build && cmake .. -DPYTHON_EXECUTABLE="$(which python)" && make -j"$(nproc)"

if [ -d /kaolin ]; then
  cd /kaolin && rm -rf build *egg* && IGNORE_TORCH_VER=1 python setup.py develop
fi

cd "${DIR}/bundlesdf/mycuda" && rm -rf build *egg* *.so && python setup.py develop

cd "${DIR}"
