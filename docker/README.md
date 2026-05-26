# Docker

Build the local image:

```bash
docker build --network host -f docker/dockerfile -t foundationpose:cu128 .
```

Run the container:

```bash
bash docker/run_container.sh
```

First time inside the container, build the extensions:

```bash
bash build_all.sh
```

Later, if the container is still running:

```bash
docker exec -it foundationpose bash
```
