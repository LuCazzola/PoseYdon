import numpy as np
from poseydon.build.prepare import PrepareChain
from poseydon.core.animation import Animation
from poseydon.core.rotations import QUAT_IDENTITY
from poseydon.features.reduce import build_reduction
from poseydon.export import CharacterExporter
from poseydon.io.bvh import BVH

def _exporter():
    anim=Animation(rotations=np.tile(QUAT_IDENTITY,(2,2,1)),translations=np.zeros((2,2,3)),offsets=np.array([[0,0,0],[0,1,0.]],float),parents=np.array([-1,0]),names=("Root","Leaf"),fps=30)
    rigid=anim.as_rigid_body(joint_translation="drop")
    return CharacterExporter(__import__("pathlib").Path("."),__import__("pathlib").Path("."),"Rig",PrepareChain(()),{},build_reduction(rigid)),anim

def test_prepared_bvh_preserves_model_topology():
    exporter,anim=_exporter()
    out=exporter.prepared_bvh(anim).to_animation()
    assert out.names==anim.names
    np.testing.assert_array_equal(out.parents,anim.parents)

def test_missing_fbx_is_typed_absence():
    exporter,anim=_exporter()
    import pytest
    with pytest.raises(Exception, match="source_surface_missing"):
        exporter.prepared_fbx(anim)
