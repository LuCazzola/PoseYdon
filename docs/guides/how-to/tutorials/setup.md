# Setting up

By the end of this page you will have the containers built, the tests passing
and the `poseydon` command answering.

## What you need

**Docker, and nothing else.** You don't install Python, pip or uv on your
machine. Every command in these guides runs inside a container defined in
`docker-compose.yml`, which is why they all start with `docker compose run`.

There are five containers. You need the first one now; the others only when you
get to what they do.

| container | what it's for | when you need it |
|---|---|---|
| `test` | the tests, the `poseydon` command, the dataset scripts (CPU only) | always |
| `train` | training on the GPU | [training](train-and-generate.md) |
| `fbxsdk` | reading and writing FBX files | exporting FBX |
| `fbx` | Blender, used only to render an FBX to a video | looking at an FBX |
| `compat` | the reference implementation, for comparing against | rarely |

## Build and run the tests

```bash
docker compose build test
docker compose run --rm -T test python -m pytest -q
```

The suite checks the [data round trip](../data/roundtrip.md) on every clip of
the dataset, so it needs the dataset built first (see
[the next tutorial](build-a-corpus.md)). On a machine without it, the
round-trip tests have nothing to run. The whole BVH half takes about two
minutes.

## Ask the command what it knows

```bash
docker compose run --rm test python -m poseydon.cli list
```

This prints every component you can pick by name in a config: feature blocks,
conditioners, losses, samplers and so on. When a guide says
`features: [ric_pos, rot6d_joint, ...]`, those names come from this list.

## Optional: FBX support

FBX files are read and written through the Autodesk FBX SDK. It is free to
use but may not be redistributed, so you fetch your own copy and accept
Autodesk's license yourself:

```bash
scripts/fetch_fbx_sdk.sh --accept-license
docker compose build fbxsdk
```

The SDK exists for x86-64 only. On an ARM machine the `fbxsdk` container runs
under QEMU emulation, which has to be installed once (and again after a reboot):

```bash
docker run --privileged --rm tonistiigi/binfmt --install amd64,386
```

BVH works everywhere without any of this.

**Next:** [Building the dataset](build-a-corpus.md).
