# data

The read path: a prepared corpus to normalized feature windows, collated into
a `MotionBatch` ([core](../core/index.md)). Per item, `MotionDataset` loads
`clips/<Rig>/<action>.npz`, applies the rig's reduction from `skeleton.npz`,
then any augmentations ([augment](../augment/index.md)), extracts the
configured features ([features](../features/index.md)), normalizes with
`stats.npz` (transported through the augmentation's `JointEdit`), and crops a
window. `collate` pads to the batch's own maxima; `balanced_weights` gives
every rig an equal share of samples.

`data` also owns the statistics' format and policy (`Normalizer`,
`BlockPolicy`): stage 2 fits and saves them ([build](../build/index.md)), the
dataset only loads them, and retargeting uses them to denormalize. The policy
per block is declared in `configs/dataset/truebones.yaml`; the window in
`configs/data/`. It does not own writing the corpus, what a feature or a
conditioner is ([conditioners](../conditioners/index.md)), or the DataLoader
and worker setup (`src/poseydon/training/lightning.py`,
[training](../training/index.md)).

## Class diagram

```mermaid
classDiagram
  direction TB
  class MotionDataset {
    features: tuple
    records: list
    spec FeatureSpec
    #95;#95;getitem#95;#95;(i) Item
    set_worker_seed(worker_id)
  }
  class Window {
    <<interface>>
    count(n_frames)* int
    bounds(n_frames, index, rng)* tuple
  }
  class FullClip
  class RandomCrop { length }
  class SlidingWindow {
    length
    stride
  }
  class Normalizer {
    <<dataclass>>
    mean
    std
    fit(...)$
    normalize(x)
    denormalize(x)
  }
  class BlockPolicy {
    <<dataclass>>
    name
    center
    scale
  }
  class ClipView { <<dataclass>> }
  class Item { <<dataclass>> }
  Window <|-- FullClip
  Window <|-- RandomCrop
  Window <|-- SlidingWindow
  MotionDataset o-- Window : window, RandomCrop() if None
  MotionDataset *-- Conditioner : one per configured name
  MotionDataset *-- AugmentPipeline : from augmentations
  MotionDataset o-- ClipRecord : records of the split
  MotionDataset *-- Normalizer : loaded once per rig
  MotionDataset *-- JointReduction : built once per rig
  MotionDataset ..> ClipView : per item, for conditioners
  MotionDataset ..> Item : #95;#95;getitem#95;#95;
  Normalizer o-- FeatureSpec : spec
  ClipView o-- Normalizer
  cssClass "Window" iface
  cssClass "MotionDataset,FullClip,RandomCrop,SlidingWindow" impl
  cssClass "Normalizer,BlockPolicy,ClipView,Item" data
  cssClass "Conditioner,AugmentPipeline,ClipRecord,JointReduction,FeatureSpec" ext
```

`BlockPolicy` says how one feature block is centred and scaled when the
statistics are fitted at build time ([build](../build/index.md)'s
`BuildConfig.normalize`). `collate` turns a list of `Item` into a `MotionBatch`
([core](../core/index.md)).

## Reference

::: poseydon.data
    options:
      members: false

::: poseydon.data.collate

::: poseydon.data.dataset

::: poseydon.data.normalize

::: poseydon.data.sampler

::: poseydon.data.window
