# sampling.controls

The step controls: `Inbetween` imputes known frames at every step, `LatentPin`
holds one semantic latent fixed for the whole trajectory (how retargeting
decodes one clip's content on another rig), and `LatentMix` blends two
latents on a schedule (`SCHEDULES` in
`src/poseydon/sampling/controls/latent_mix.py`). Each is built by the
operation of the same task ([ops](../../ops/index.md)) and run by a sampler
([sampling](../index.md)).

## Class diagram

```mermaid
classDiagram
  direction LR
  class Control {
    <<base class>>
    before_step(z_t, t, cond) tuple
    after_step(z_t, t, cond) Tensor
  }
  class Inbetween {
    reference
    known
    after_step(...)
  }
  class LatentMix {
    reference
    target
    alpha
    key
    before_step(...)
  }
  class LatentPin {
    latent
    key
    before_step(...)
  }
  Control <|-- Inbetween
  Control <|-- LatentMix
  Control <|-- LatentPin
  Inbetween o-- GaussianDiffusion : process, to re-noise the known frames
  cssClass "Inbetween,LatentMix,LatentPin" impl
  cssClass "Control,GaussianDiffusion" ext
```

`Control` is in [sampling](../index.md), `GaussianDiffusion` in
[process](../../process/index.md). Each control is built by the operation of
the same task ([ops](../../ops/index.md)).

## Reference

::: poseydon.sampling.controls
    options:
      members: false

::: poseydon.sampling.controls.inbetween

::: poseydon.sampling.controls.latent_mix

::: poseydon.sampling.controls.latent_pin
