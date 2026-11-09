# Testing

The library, tests and scripts never run on host Python: they run through docker compose
(`docker-compose.yml`). The docs site and the ticket scripts are the two exceptions, and
run on the host.

## Test suite

```bash
docker compose run --rm -T test python -m pytest -q
```

The `test` image has no FBX SDK, so the FBX half of `tests/test_roundtrip.py` and all of
`tests/test_verify.py` skip there. A note at the end of the run says when the FBX half
was skipped. To check both formats, run the suite in `fbxsdk`:

```bash
docker compose run --rm -T fbxsdk python -m pytest -q -k Camel
```

- `--dataset <name>` (default `truebones`, `tests/conftest.py`) picks the corpus
  `data/<name>/` and the config `configs/data/<name>.yaml`; raw files come from that
  config's `paths.raw`.
- There is one round-trip test per clip, so `-k Rig` or `-k Rig/action` narrows a run.
- `fbxsdk` is an amd64 image. On an ARM host it runs under QEMU, which has to be
  registered after every reboot:
  `docker run --privileged --rm tonistiigi/binfmt --install amd64,386`. The full corpus
  takes hours there, so narrow with `-k`.
- The round-trip tests need a prepared corpus and the raw vendor files, which are
  licensed and not in git.

## Motion verifier

```bash
scripts/verify_motion.sh --data truebones --rig Camel   # or --smoke
```

It runs on the host and calls docker compose for each step. The run directory goes to
`tmp/verify` by default. Its knobs and smoke set are in `configs/verify/verify.yaml`.
See [Verification](../user-guide/verification.md).

## Docs checks

```bash
python3 docs/parallax/tools/check_docs.py
```

This checks that the paths docs name exist, that each User Guide page has a `Code:` line,
and that API stubs match the packages. A pass means they match, not that what they say
is right.

The site must also build with `--strict`:

```bash
python3 docs/parallax/tools/serve.py --build
```

## Preview the site

```bash
python3 docs/parallax/tools/serve.py
```

Then open <http://127.0.0.1:8000>. It reloads as pages change. It needs only Python: the first run
without `uv` sets up the tools in a private folder, ignored by git. Versions are pinned in
`docs/parallax/requirements.txt`, the one place to change them.
