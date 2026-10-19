# conditioners

Declared conditioning: each extra signal a batch may carry, with its whole
path, extracted per item from a `ClipView`, collated across the batch with
joint padding, and optionally a default encoder. The shipped ones are all
skeleton-derived: `topology`, `tpose`, `norm_stats`, `joint_names`. Only
declared conditioners are loaded, so an unconditional corpus declares none.

`joint_names` reads T5 embeddings from a cache keyed by raw joint name
(`data/truebones/cache/joint_names_t5.npz`). `text` builds it: the reference's
name clean-up and pooling, ported and measured bit-identical to the
reference's own cache on all 1185 names it held. Building needs the `text`
extra (`compat` image); training never does.

How a model injects a signal stays in [models](../models/index.md), and so does
conditioning dropout for guidance. The batch type, `Cond`, is in
[core](../core/index.md). Declared as `conditioners:` in `configs/train.yaml`
and `configs/sample.yaml`; the list is recorded in the checkpoint recipe and a
mismatch raises at sampling ([training](../training/index.md)).

## Limitations

- **AnyTop ignores `joint_names` unless `name_embedding_dim: 768` is set**
  (`configs/model/anytop.yaml`): its projection stays `None` and the embeddings
  are dropped without a warning. `MoDiffAE` wires names differently, and
  passing it the same key raises (`configs/model/modiffae.yaml`).
- **The name clean-up is the reference's, quirks kept.** It strips every
  `_REMOVE_PREFIXES` match in turn and drops one-letter parts; a `Sabrecat`
  name also loses its last six characters, which on SabreToothTiger drops
  the side (`Sabrecat_Finger4_LF04_` and `..._RF04_` both read `Finger`).
  Crow's names (`_00`..`_23`) read as nothing and embed as zeros. 1187 names
  read as 381 texts. Changing a rule breaks parity with the reference's
  checkpoints and needs the cache rebuilt (`tools/build_joint_name_cache.py`).
- **`topology` reads the clip's own offsets**, which differ between clips of a
  rig ([preparation](../build/preparation.md#limitations)), so the clip you
  template on matters.

## Class diagram

```mermaid
classDiagram
  direction LR
  class Conditioner {
    <<base class>>
    name: str
    extract(item)* payload
    collate(payloads, max_joints)* batched
    encoder(d_model) module or None
  }
  class Topology {
    max_path: int
  }
  class TPose
  class NormalizationStats
  class JointNames
  Conditioner <|-- Topology
  Conditioner <|-- TPose
  Conditioner <|-- NormalizationStats
  Conditioner <|-- JointNames
  Conditioner --> ClipView : extract(view), per clip
  cssClass "Conditioner" iface
  cssClass "Topology,TPose,NormalizationStats,JointNames" impl
  cssClass "ClipView" ext
```

`encoder` defaults to `None`; how the payload is injected stays with the
model. `MotionDataset` builds one of each configured name from `CONDITIONERS`
and calls `extract` on the `ClipView` it assembles ([data](../data/index.md)).

## Reference

::: poseydon.conditioners
    options:
      members: false

::: poseydon.conditioners.base

::: poseydon.conditioners.skeleton

::: poseydon.conditioners.text
