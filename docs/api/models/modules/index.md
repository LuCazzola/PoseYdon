# models.modules

The building blocks of [models](../index.md): the graph backbone MoDiffAE's
encoder and decoder share (with `ConditionAttention`, multi-head semantic
modulation, averaged or summed), AnyTop's relation-biased `GraphAttention`,
positional and timestep embeddings, and the two maskers.

The maskers give the latent a bottleneck and guidance its unconditional branch.
Without masking, the semantic encoder sees the whole clip it is asked to
denoise and can carry a near-copy of it, which is fatal for retargeting.
`MotionMasker` hides frames, joints and whole kinematic chains (`chain_ids`)
from the encoder; `ConditionMasker` drops the whole latent for its learned
`null_latent`. Each null embedding exists only when its masking is on, so a
checkpoint trained with masking off has none to load, and guidance on it
raises.

## Class diagram

```mermaid
classDiagram
  direction TB
  class GraphMotionDecoder {
    layers: ModuleList
    #95;#95;init#95;#95;(make_layer, num_layers)
  }
  class GraphMotionDecoderLayer
  class SpatialAttention
  class TemporalAttention
  class ConditionAttention {
    n_heads
    aggregate
  }
  class GraphMultiHeadAttention
  class GraphRelEmbedder
  class GraphAttention {
    n_heads
    value_bias
  }
  class MotionMasker {
    frames, joints, chains
    null_token
    mask(...)
    apply(...)
  }
  class ConditionMasker {
    cond_drop
    null_latent
    drop(...)
  }
  GraphMotionDecoder *-- GraphMotionDecoderLayer : one per layer, via make_layer
  GraphMotionDecoderLayer *-- SpatialAttention
  GraphMotionDecoderLayer *-- TemporalAttention
  GraphMotionDecoderLayer *-- ConditionAttention : optional, cross
  SpatialAttention *-- GraphMultiHeadAttention
  GraphMultiHeadAttention *-- GraphRelEmbedder : graph_embed
  cssClass "GraphMotionDecoder,GraphMotionDecoderLayer,SpatialAttention,TemporalAttention,ConditionAttention,GraphMultiHeadAttention,GraphRelEmbedder,GraphAttention,MotionMasker,ConditionMasker" impl
```

Three separate pieces: the graph backbone `MoDiffAE`'s encoder and decoder
are built on, `GraphAttention` (AnyTop's spatial attention, used by
`AnyTopLayer`), and the two maskers `MoDiffAE` owns
([models](../index.md)). All are `nn.Module`s.

## Reference

::: poseydon.models.modules
    options:
      members: false

::: poseydon.models.modules.embeddings

::: poseydon.models.modules.graph_attention

::: poseydon.models.modules.graph_backbone

::: poseydon.models.modules.masking
