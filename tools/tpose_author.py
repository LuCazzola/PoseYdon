"""Authoring T-poses for Truebones rigs whose rest pose comes from a motion clip: the whole toolkit in one file.

WHY
    15 rigs have no usable vendor T-pose, so their
    rest pose is a clip's first frame (seated, mid-trot, coiled ...). This tool authors a clean one-frame rest pose
    per rig, `data/_raw/truebones/source/<Rig>/__TPOSE_authored.bvh`, with the clips' joint set and channels.
    It changes nothing under src/ and never overwrites a vendor `__TPOSE.bvh`.

WHAT A GOOD POSE IS (the owner judges on the skinned mesh and the skeleton, so check the same renders yourself
and iterate BEFORE showing anything; do not make the owner be the feedback loop)
    - canonical frame: +Y up, facing +Z, the rig's right side at -X; tail, torso (spine) and neck straight and
      parallel to the ground; the head facing forward
    - feet planted flat: every toe / finger pad and the sole on the floor, no tiptoe, no upturned toes or claws,
      no claw tip hanging below the pads, paws not crumpled or twisted
    - joints kinematically plausible, stance not overly flexed (the walk / idle clips show the animal's own stance)
    - skeleton symmetric at pixel level in the front and top views (`skelsym`); the right side is the exact mirror of
      the left (pose one side, mirror it). Only where the vendor mesh itself is asymmetric (Sabre's front paws)
      may a part be posed per side, and that is written down in the todo row.

WORKFLOW (all through docker compose; never host Python)
    1. build the problem (once per rig; FBX SDK, the `fbxsdk` service, emulated on ARM hosts: minutes):
         docker compose run --rm -T --entrypoint python -e HOME=/tmp fbxsdk tools/tpose_author.py build Rig
       -> tmp/tposes4/<Rig>_problem.npz (bone vectors, canonical frame T, rule-based start pose W, bind Wbind,
          twin pairs, midline joints, legs, and the vendor skin as inf_j / inf_u / inf_w) and <Rig>_faces.npz
    2. look at the animal in its own clips (what a natural stance, paw and finger placement look like):
         ... train tools/tpose_author.py ref OUT Rig __Walk.bvh 0         (sheet + paws of a clip frame)
    3. write the rig's config (YAML, ops below) to .superpowers/tpose/manual/<Rig>.yaml (or start from RIGS here)
       and pose it; iterate on the renders:
         ... train tools/tpose_author.py pose OUT Rig --paws LeftFrontPaw LeftHindPaw
       -> OUT/<Rig>_tpose.bvh, OUT/sheet_<Rig>.png (front / side / top / 3/4 of the shaded skinned mesh with the
          skeleton), OUT/feet.png (each paw close up, left beside right, front + side), OUT/skelsym_<Rig>.png
          (skeleton vs its mirror, pixel diff), and a JSON line (joint mirror error, midline error, facing).
         ... train tools/tpose_author.py paw OUT Rig LeftHand    (a paw from 4 angles; front views are telling)
         ... train tools/tpose_author.py flips Rig               (the left -> right axis flip per twin pair)
         ... train tools/tpose_author.py belly OUT Rig [clip:frame]   (twist check: back red / belly blue; snakes, spines)
       where "..." is `docker compose run --rm -T --entrypoint python -e HOME=/tmp`.
    4. when the owner approves: `... train tools/tpose_author.py save OUT Rig` (copies the BVH next to the clips,
       checks joint names and channels against a clip) and update the todo row (which rigs have one; any vendor
       asymmetry that could not be posed away).

CONFIG OPS (applied in this order; joints named on the LEFT side are mirrored onto their twin unless noted)
    base: rule | bind | clip:<file>:<frame>   start pose: the rule base, the vendor bind, or a clip frame
    root_from_rule: true          with a clip base: the root's own turn from the rule base (else the body can hang
                                  off the midline, as Trex's did by 45 units)
    local_from_rule: [joint]      a joint's turn on its parent from the rule base, what hangs below keeping its world
                                  turn (check the joint's skin still hangs the right way: the rule base can flip it)
    clip_local: {clip, frame, joints}   subtrees given their clip-frame turn relative to the parent (neck + head)
    sym_skin: [joint]             a midline subtree yawed / rolled until its twin children (clavicles) or its midline
                                  skin mirror (fixes a body or neck twisted off the midline)
    sym_skin_late: [joint]        the same, after the level / rotate / mirror edits (a head left tilted)
    sym_hips: [[joint, wmid]]     (late) the joint's own turn solved so its twin children (hips) mirror, against wmid x its
                                  midline children leaving the midline (Lion: the vendor hips sit 3.5 off-mirror on a straight spine)
    sym_roll: [joint]             (late) a midline joint rolled about its own bone onto its skin's mirror, nothing else
                                  moving (a torso joint whose belly leans to one side)
    sym_chain: [joint, ...]       (late) a midline chain root to tip (a tail): each joint yawed and rolled until every segment's own
                                  skin mirrors itself; joints stay on the midline (only the tip yaws freely)
    clip_twist: {clip, frame, joints}   midline joints given the clip's twist, bone direction kept (lopsided belly)
    roll_to_bind: true | clip:frame   each midline joint rolled about its own bone until its bind (or a flat-lying clip
                                  frame's) up axis points up: a straight chain's rolls are arbitrary in the rule base
                                  (Anaconda's twisted up to 180; its bind rears the neck, so a flat frame is the reference)
    rotate: [[joint, x|y|z, deg]] world-axis turn of a subtree
    legs_down: [[root, end, (gap)]]   (after level) the leg chain root -> end straight down, splayed outward only until
                                  the feet (end and below) clear each other by gap (default 2% of the extent)
    palms_down: [hand | [hand, shoulder, (skin)]]   (after aim) the hand (or the whole arm from shoulder) rolled about
                                  the arm's axis until its fingers lie flat (skin: until the hand's skin, feathers, is flat)
    aim: [[joint, child, [x,y,z], (one)]]   point a bone along a world direction (subtree follows)
    level: [chain, ^chain, A>B]   straighten a chain parallel to the ground ("^" also levels the bone into it by
                                  pitching the parent; "A>B" stops at B, which keeps its world turn). Use it for
                                  tail, neck (Neck>Head) and back (Spine1>Neck1)
    mirror_skin: [limb]           right end joints turned until their skin mirrors the left's
    stance_leg: {clip, body: [tail end, head end], legs: [root | [root, joint the paw goes under]]}
                                  each leg as at its own mid-stance frame of a clip (paw planted, under the root),
                                  relative to that frame's body heading
    torso_pca: true               body pitched until its long axis is level
    level_pca: [limb]             a limb tilted until its first PCA axis is level (pincers, claws)
    aim_pca: [[part, [x,y,z]]]    a part turned whole until its skin's long axis points along a direction (a wing
                                  finger skinned to the wrist: [jt_Wrist_L, [1, 0, 0]])
    stance: {clip, feet}          a foot oriented as in its own stance frame of a clip
    plant: [[foot, x|xz, (lim), (one)]]   a foot rocked until most sole vertices touch the floor
    aim_late: like aim, after the stance edits (set a humerus / forearm / cannon bone by hand)
    clip_paw_body: [tail end, head end];  clip_paw: [[clip, frame, paw]];  clip_paw_forward: true
                                  a paw (this joint and below) as in a clip frame where it is planted, heading-
                                  corrected; forward: yawed so its digits point straight ahead
    mirror_legs: [limb or A>B]    right bones aimed at the mirrored left joints, twist fitted to the skin
    mirror_feet: [foot]           the right foot turned to its skin mirror
    mirror_rig: true              every right joint = the exact mirror of its left twin, W_r = M W_l S (S: the
                                  vendor's per-pair axis flip, see FLIPS below); limb roots whose vendor offsets
                                  are not mirrored (clavicles) are re-aimed so their child joint mirrors too
    fit_paw: [{paw, palm, all_phalanges, mirror, reg, digit_deg, pad_q}]   (pad_q 0.25: a toe lies along the floor, not on its tip)   the paw's turn (3) and each digit's pitch solved on the
                                  mesh (differential evolution) so every digit pad (and the palm / heel with palm,
                                  every phalanx with all_phalanges) rests on the paw's lowest point; mirror copies
                                  the result to the twin (symmetric meshes); without mirror, list both paws
    flat_toes: [paw | [paw, deg]] (before fit_paw) each digit's base bone pitched level (or deg below level), heading
                                  kept, the claw beyond keeping its curl: toes on their pads, claw tips down
    fit: [[upper, lower, axis, (sign), (foot), (back)]]   bend so this leg's paw meets the floor the other legs
                                  set (parallel bend, or with foot a z-bend that keeps the paw's turn)
    plant_final: like plant, after the mirror (each side on its own, small limit)
    fit_one: like fit, this side only
    ground: [limb]                legs tilted about their root until they touch the floor (twin mirrored)

LESSONS (each cost an iteration with the owner)
    - FLIPS: the Max biped flips one local axis on every right joint (a left thigh offset [0,0,17.85] is [0,0,
      -17.85] on the right), so copying a left rotation is NOT a mirror. mirror_rig picks the flip from the child
      offsets; when they tie (within 1% of their length: end bones, offsets on an axis), the bind pose decides.
      A wrong flip leaves the skeleton mirrored but twists the mesh 180 deg (crumpled claws, splayed toes):
      run `flips` when a mirrored paw looks different from its twin.
    - Look at the paws close up (feet, paw): the sheet hides floating toes and claw spikes. The paw must rest
      on its true lowest point; ignoring a "lone spike" let Sabre's paws stand on one claw tip 1.45 above the pads.
    - Paw and finger placement come from the animal's own clips: the vendor bind pose holds fingers 30-40 deg
      off every clip. The medoid of the planted frames (or a clip where all four paws are flat, e.g. Monkey's
      __Idle1 frame 0) is a good start; fit_paw then grounds every digit.
    - A natural standing clip frame (Trex __STILL frame 0) can be the base itself; then centre it
      (root_from_rule, sym_skin), level tail / back / neck, mirror_rig, fit_paw.
    - Splay limbs a little outward if hands or feet meet at the midline; straighten arms / unbend legs if the
      stance is too low; keep a slight knee bend if a straight leg looks like a post.
    - A straight chain's rolls are arbitrary in the rule base: a snake looked straight but its mesh was twisted up to
      180 deg. Check with `belly` (needs a flat reference: the bind, or a clip frame lying flat) and fix with roll_to_bind.
    - Some vendor bones cross the midline (Goat's clavicles, Trex's hip pivots: jt_Hip_L sits on the right with a bone
      across to jt_Thigh_L); leave those bones alone and start leg edits below them. `tree` shows where joints sit.
    - The dewclaw (a digit up the leg) is not a support: fit_paw skips digits higher than 30% of the paw.
    - Vendor asymmetries that cannot be posed away: Goat's upper-arm helper bones, Lion's hips (3.5 off),
      Sabre's clavicles (2.5) and front paw meshes. Record them in the todo row.

STATUS (2026-10-19): saved for all 15: Anaconda, Ant, Bird, Camel, Crab, Cricket, Goat, Jaguar, Lion, Monkey, Pteranodon, Rat,
    SabreToothTiger, Scorpion, Trex. The configs in RIGS are the
    last ones used; rigs saved early were posed with earlier versions of these ops, so a rerun may differ slightly
    from the saved BVH, which is the reference.
"""
import argparse, json, os, re, sys, shutil
from pathlib import Path
import numpy as np
import yaml
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from tools.verify.sheet import _plt, draw

PROB = Path("tmp/tposes4")                     # problem + faces files (build output)
CONFIGS = Path(".superpowers/tpose/manual")    # a <Rig>.yaml here overrides RIGS below
RAW = Path("data/_raw/truebones/source")
M = np.diag([-1.0, 1, 1])

RIGS = {   # the configs used for the authored poses (see STATUS)
    "Ant": """
base: rule
level: [Bip01_Spine]
ground: [Bip01_L_Thigh, Bip01_L_Clavicle]
torso_pca: true
level_pca: [BN_Clip_L_01]
rotate: [[BN_Clip_L_01, y, 25]]   # mandibles swung outward
""",
    "Crab": """
base: rule
ground: [BN_leg_L_01, BN_leg_L_06, BN_Leg_L_11]
torso_pca: true
level_pca: [BN_Arm_L_01]
rotate: [[BN_Arm_L_01, y, 25]]   # claws swung outward
""",
    "Cricket": """
base: rule
level: [BN_Tail_01, BN_Feeler_L_05, Bip01_Spine]
ground: [BN_leg_L_01, Bip01_L_Thigh, Bip01_L_Clavicle]
""",
    "Scorpion": """
base: rule
level: [Bip01_Tail]
mirror_skin: [Bip01_L_Thigh_1, Bip01_L_Thigh_2, Bip01_L_Thigh_3, Bip01_L_Thigh_4]
ground: [Bip01_L_Thigh_1, Bip01_L_Thigh_2, Bip01_L_Thigh_3, Bip01_L_Thigh_4]
torso_pca: true
level_pca: [Bip01_L_Clavicle]
rotate: [[Bip01_L_Clavicle, y, 25]]
""",
    "Jaguar": """
base: rule
level: [^BN_Tail_01]
mirror_skin: [Bip01_L_Clavicle, Bip01_L_Thigh]
stance: {clip: __Walk.bvh, feet: [Bip01_L_Foot]}
aim:
  - [Bip01_L_Thigh, Bip01_L_Calf, [0, -1, 0]]
  - [Bip01_L_Calf, Bip01_L_HorseLink, [0, -1, 0]]
  - [Bip01_L_HorseLink, Bip01_L_Foot, [0, -1, 0]]
fit:
  - [Bip01_L_Thigh, Bip01_L_Calf, x]
  - [Bip01_L_UpperArm, Bip01_L_Forearm, x, 1]
""",
    "Rat": """
base: rule
level: [Tail01]
mirror_skin: [LeftArm, LeftUpLeg]
aim:
  - [LeftArm, LeftForeArm, [0, -1, 0]]
  - [LeftForeArm, LeftHand, [0, -1, 0]]
  - [LeftUpLeg, LeftLeg, [0, -1, 0]]
  - [LeftLeg, LeftFoot, [0, -1, 0]]
stance: {clip: __Trottle.bvh, feet: [LeftHand, LeftFoot]}
plant: [[LeftHand, xz], [LeftFoot, xz]]
fit:
  - [LeftArm, LeftForeArm, x, 1]
  - [LeftUpLeg, LeftLeg, x, -1, LeftFoot]
""",
    "Camel": """
base: rule
level: [^BN_Tail_01, Bip01_Neck1>Bip01_Head]   # tail straight back and neck straight forward (head keeps its turn), parallel to the ground
mirror_skin: [Bip01_L_Clavicle, Bip01_L_Thigh]   # right hooves turned to mirror the left skin
aim:               # legs straight down, then one bend each
  - [Bip01_L_Thigh, Bip01_L_Calf, [0, -1, 0]]
  - [Bip01_L_Calf, Bip01_L_HorseLink, [0, -1, 0]]
  - [Bip01_L_HorseLink, Bip01_L_Foot, [0, -1, 0]]
stance: {clip: __SlowWalk.bvh, feet: [Bip01_L_Foot]}
plant: [[Bip01_L_Foot, xz]]
fit:
  - [Bip01_L_Thigh, Bip01_L_Calf, x, -1, Bip01_L_Foot, 1.5]   # z-bend: stifle forward, hock back (1.5x), hoof under the hip
mirror_rig: true   # right side the exact mirror of the left (the right hind leg was its own fit)
""",
    "Goat": """
base: rule
level: [^BN_Tail_01, ^Bip01_Spine>Bip01_Neck, Bip01_Neck>Bip01_Head]   # tail straight back, neck straight forward (head keeps its walk turn)
mirror_skin: [Bip01_L_Clavicle, Bip01_L_Thigh]   # right hooves turned to mirror the left skin
clip_local: {clip: __Walk.bvh, frame: 0, joints: [Bip01_Neck]}   # neck, head and shoulders (crossing the midline) as in the walk (the rule base rolls Bip01_Neck 179 deg)
sym_skin: [Bip01_Neck, Bip01_Neck1]   # the walk frame turns them a little: Neck rolled until the clavicle roots mirror, Neck1 until the head's skin does
aim:               # legs straight down (the base zig-zags the front ones), then one bend each
  - [Bip01_L_Clavicle, Bip01_L_UpperArm, [1, 0, 0]]   # shoulder bones straight across (they cross the midline), mirrored
  - [Bip01_L_Thigh, Bip01_L_Calf, [0, -1, 0]]
  - [Bip01_L_Calf, Bip01_L_HorseLink, [0, -1, 0]]
  - [Bip01_L_HorseLink, Bip01_L_Foot, [0, -1, 0]]
  - [Bip01_L_UpperArm, Bip01_L_Forearm, [0, -1, 0]]
  - [Bip01_L_Forearm, Bip01_L_Hand, [0, -1, 0]]
  - [Bip01_L_Hand, Bip01_L_Finger0, [0, -1, 0]]
# (torso_pca dropped: the back is levelled by its spine bones)
rotate: [[Bip01_Head, x, -20]]   # head up, muzzle forward (it pointed down; the raised neck adds the rest)
stance: {clip: __Walk.bvh, feet: [Bip01_L_Finger0, Bip01_L_Foot]}
plant: [[Bip01_L_Foot, xz], [Bip01_L_Finger0, xz]]
mirror_feet: [Bip01_L_Foot]   # right hind hoof refitted to mirror the planted left (symmetric skeleton)
fit:
  - [Bip01_L_UpperArm, Bip01_L_Forearm, x, 1, Bip01_L_Hand, 1.5]   # z-bend: elbow back, forearm forward, hoof under the withers
  - [Bip01_L_Thigh, Bip01_L_Calf, x, -1, Bip01_L_Foot, 1.5]   # z-bend: stifle forward, hock back, hoof under the hip
aim_late:   # upper neck raised so the neck runs straight into the head (level, its mesh dipped); from Neck1: the shoulders hang off Bip01_Neck
  - [Bip01_Neck1, Bip01_Head, [0, 0.35, 1]]
sym_skin_late: [Bip01_Head]   # head rolled level (it tilted to the right)
sym_chain: [BN_Tail_01, BN_Tail_02]   # tail rolled (tip also yawed) until each segment's skin mirrors itself (the tip was lopsided)
""",
    "Lion": """
base: rule
clip_local: {clip: __Walk.bvh, frame: 0, joints: [Bip01_Neck]}   # neck, head and shoulders as in the walk (the rule base rolls Bip01_Neck 172 deg)
stance_leg: {clip: __Walk.bvh, body: [Bip01_Pelvis, Bip01_Neck], legs: [Bip01_L_UpperArm, Bip01_L_Thigh]}   # each leg as at its own mid-stance
sym_skin: [Bip01_Neck, Bip01_Neck1]   # Neck rolled until the clavicle roots mirror, Neck1 until the head's skin does
rotate: [[Bip01_Head, x, -50]]   # muzzle up from the walk's lowered head
level: [^BN_Tail_01, Bip01_Neck>Bip01_Head]   # tail straight back, neck straight forward (head keeps its walk turn)
aim:
  - [Bip01_L_Clavicle, Bip01_L_UpperArm, [1, -0.3, -0.19]]   # shoulder bones cross the midline: mirrored, at the walk's mean slope
torso_pca: true
aim_late:   # a little less crouched than the walk's mid-stance: front forearm upright, hind cannon upright
  - [Bip01_L_Forearm, Bip01_L_Hand, [0, -1, 0.15]]
  - [Bip01_L_HorseLink, Bip01_L_Foot, [0, -1, 0.1]]
plant: [[Bip01_L_Hand, xz], [Bip01_L_Foot, xz]]   # paws flat (each side on its own)
mirror_legs: [Bip01_L_UpperArm, Bip01_L_Thigh]   # right legs rebuilt as mirrors of the left (bones aimed at the mirrored joints, twist fitted to the skin)
fit:
  - [Bip01_L_Thigh, Bip01_L_Calf, x]   # hind legs unbent at the hip, then the knee, until their paws reach the floor the front paws set
  - [Bip01_L_Calf, Bip01_L_HorseLink, x]
  - [Bip01_L_UpperArm, Bip01_L_Forearm, x, 1]   # (only if still needed) front legs bent, elbow back
plant_final: [[Bip01_L_Hand, xz, 20], [Bip01_L_Foot, xz, 20]]   # soles flat again after the bend (heel down)
fit_one:   # then each leg on its own back onto the floor
  - [Bip01_L_UpperArm, Bip01_L_Forearm, x, 1]
  - [Bip01_R_UpperArm, Bip01_R_Forearm, x, 1]
  - [Bip01_L_Calf, Bip01_L_HorseLink, x]
  - [Bip01_R_Calf, Bip01_R_HorseLink, x]
sym_chain: [BN_Tail_01, BN_Tail_02, BN_Tail_03, BN_Tail_04]   # tail rolled until each segment's skin mirrors itself (it wandered side to side)
""",
    "SabreToothTiger": """
base: rule
stance_leg: {clip: __Run.bvh, body: [Sabrecat__pelv_, Sabrecat_Neck_Nek0_], legs: [Sabrecat_LeftClavicle_LClv_, Sabrecat_LeftThigh_LThi_]}
clip_twist: {clip: __Run.bvh, frame: 0, joints: [Sabrecat_Spine_Spn0_, Sabrecat_Spine_Spn1_, Sabrecat_Spine_Spn2_, Sabrecat_Spine_Spn3_]}
level: [Sabrecat_Tail0_Tal0_]
aim_late:
  - [Sabrecat_LeftUpperArm_LUar_, Sabrecat_LeftForearm_LFar_, [-0.12, -0.55, -1]]
  - [Sabrecat_LeftForearm_LFar_, Sabrecat_LeftHand_LHnd_, [-0.4, -1, 0.3]]
  - [Sabrecat_LeftFoot_LFot_, Sabrecat_LeftToe0_LT00_, [0, -1, 0.35]]
clip_paw_body: [Sabrecat__pelv_, Sabrecat_Neck_Nek0_]
clip_paw:
  - [__Attack.bvh, 27, Sabrecat_LeftHand_LHnd_]   # medoid of the planted frames' finger turns
  - [__Attack.bvh, 27, Sabrecat_LeftToe0_LT00_]
mirror_rig: true
fit_paw:   # per side: the two front paw meshes differ
  - {paw: Sabrecat_LeftHand_LHnd_}
  - {paw: Sabrecat_RightHand_RHnd_}
fit:
  - [Sabrecat_LeftCalf_LClf_, Sabrecat_LeftFoot_LFot_, x]
  - [Sabrecat_LeftUpperArm_LUar_, Sabrecat_LeftForearm_LFar_, x, 1]
plant_final: [[Sabrecat_LeftToe0_LT00_, xz, 8]]
fit_one:
  - [Sabrecat_LeftUpperArm_LUar_, Sabrecat_LeftForearm_LFar_, x, 1]
  - [Sabrecat_RightUpperArm_RUar_, Sabrecat_RightForearm_RFar_, x, 1]
  - [Sabrecat_LeftCalf_LClf_, Sabrecat_LeftFoot_LFot_, x]
  - [Sabrecat_RightCalf_RClf_, Sabrecat_RightFoot_RFot_, x]
""",
    "Monkey": """
base: rule
stance_leg: {clip: __Walk.bvh, body: [Bip01_Pelvis, Bip01_Neck], legs: [[Bip01_L_Clavicle, Bip01_L_UpperArm], Bip01_L_Thigh]}
level: [BN_Tail_01]
torso_pca: true
aim_late:
  - [Bip01_L_UpperArm, Bip01_L_Forearm, [0.12, -1, -0.15]]
  - [Bip01_L_Forearm, Bip01_L_Hand, [0.12, -1, 0.1]]
  - [Bip01_L_Thigh, Bip01_L_Calf, [0, -1, 0.4]]
  - [Bip01_L_Calf, Bip01_L_Foot, [0.1, -1, -0.3]]
clip_paw_body: [Bip01_Pelvis, Bip01_Neck]
clip_paw_forward: true
clip_paw:   # plantigrade: hands and feet flat, as in the idle's first frame
  - [__Idle1.bvh, 0, Bip01_L_Hand]
  - [__Idle1.bvh, 0, Bip01_L_Foot]
mirror_rig: true
fit_paw:
  - {paw: Bip01_L_Hand, palm: true, mirror: true}
  - {paw: Bip01_L_Foot, palm: true, all_phalanges: true, mirror: true}
fit: [[Bip01_L_UpperArm, Bip01_L_Forearm, x, 1]]
""",
    "Anaconda": """
base: rule   # the whole body straight along +-Z, head forward, belly on the floor (the slight front-view lopsidedness is the vendor mesh)
roll_to_bind: __Idle2.bvh:83   # untwist: every body joint rolled to how it lies in a flat idle frame (the rule base rolled them up to 180 deg; the bind rears the neck)
""",
    "Bird": """
base: rule   # wings out, body level
aim:   # wings straight out in a T, hand in line
  - [BN_UpperArm_L_01, BN_Forearm_L_01, [1, 0, 0]]
  - [BN_Forearm_L_01, BN_hand_L_01, [1, 0, 0]]
  - [BN_hand_L_01, BN_Finger_L_02, [1, 0, 0]]
palms_down: [[BN_hand_L_01, BN_UpperArm_L_01, skin]]   # the wing rolled until its feathers lie flat
legs_down: [[Bip01_L_Thigh, Bip01_L_Foot]]   # legs straight down, perpendicular to the spine
clip_paw_body: [Bip01_Pelvis, Bip01_Neck]
clip_paw_forward: true
clip_paw: [[__IdleLoop.bvh, 0, Bip01_L_Foot]]   # toes spread and flat as when standing (idle)
mirror_rig: true
fit_paw: [{paw: Bip01_L_Foot, palm: true, all_phalanges: true, mirror: true}]   # every toe on the floor
""",
    "Pteranodon": """
base: rule   # wings straight out
level: [^jt_Spine1_C, jt_Neck1_C>jt_Head_C]   # back level (it sloped down to the hips); neck straight forward
clip_paw_body: [jt_Hips_C, jt_Neck1_C]
clip_paw_forward: true
clip_paw: [[__Lander.bvh, 40, jt_Foot_L]]   # toes fanned as when landing (the rule base crosses them)
aim_pca: [[jt_Head_C, [0, 0, 1]], [jt_Jaw_C, [0, 0, 1]], [jt_Wrist_L, [1, 0, 0]]]   # head (beak to crest) level, jaw closed onto it;   # wing finger (skinned to the wrist) straight out (it swept forward)
legs_down: [[jt_Thigh_L, jt_Foot_L]]
mirror_rig: true
flat_toes: [[jt_Foot_L, 6]]   # toes forward on their pads like the Bird's, each 6 deg down (fully flat crushed them)
fit_paw: [{paw: jt_Foot_L, all_phalanges: true, mirror: true}]   # every toe and claw tip on the floor, the sole slightly arched
""",
    "Trex": """
base: clip:__STILL.bvh:0
root_from_rule: true
sym_skin: [jt_Cog_C]
level: [^jt_Tail1_C, ^jt_Spine1_C>jt_Neck1_C, jt_Neck1_C>jt_Head_C]   # the whole spine level, root to neck
aim:   # arms in a T, hands in line
  - [jt_Shoulder_L, jt_Elbow_L, [1, 0, 0]]
  - [jt_Elbow_L, jt_Wrist_L, [1, 0, 0]]
  - [jt_Wrist_L, jt_FingerOutter_L, [1, 0, 0]]
legs_down: [[jt_Thigh_L, jt_Foot_L]]   # perpendicular to the spine; the hip pivots sit across the midline (their bones stay)
palms_down: [[jt_Wrist_L, jt_Shoulder_L]]   # the whole arm rolled, no wrist twist
mirror_rig: true
fit_paw: [{paw: jt_Foot_L, palm: true, all_phalanges: true, mirror: true}]
""",
}


# ======================================================================================================== build
# (fbxsdk service) the vendor FBX skin on the BVH rig, a rule-based start pose, and the mesh as linear-blend
# skinning data in the canonical frame. Everything after this runs on the saved arrays alone.

_SIDE = re.compile(r"Right|Left|right|left|(?<![A-Z])[RL](?![a-z])|(?<![A-Za-z])[rl](?![a-z])")
_SWAP = {"Right": "Left", "Left": "Right", "right": "left", "left": "right", "R": "L", "L": "R", "r": "l", "l": "r"}
def names_side(n): return _SIDE.search(n) is not None
def children(parents):
    k = [[] for _ in parents]
    for j, p in enumerate(parents):
        if p >= 0: k[p].append(j)
    return k
def subtree(kids, j): return [j] + [x for c in kids[j] for x in subtree(kids, c)]
def side_pairs(names, parents, pos):
    """(right, left) twin joints: by name first, then their unnamed children matched by mirrored offset."""
    idx = {n: i for i, n in enumerate(names)}; pairs = set()
    for n, i in idx.items():
        o = _SIDE.sub(lambda m: _SWAP[m.group(0)], n)
        if o != n and o in idx and _SIDE.search(n).group(0) in ("Right", "right", "R", "r"): pairs.add((i, idx[o]))
    kids = children(parents); grown = True
    while grown:
        grown = False; paired = {j for p in pairs for j in p}
        for r, l in sorted(pairs):
            fl = [c for c in kids[l] if c not in paired]
            for c in [c for c in kids[r] if c not in paired]:
                if not fl: break
                mir = M @ (pos[c] - pos[r]); best = min(fl, key=lambda d: np.linalg.norm(mir - (pos[d] - pos[l])))
                pairs.add((c, best)); fl.remove(best); paired |= {c, best}; grown = True
    return sorted(pairs)


class Skin:
    """The vendor FBX skin: per mesh part (bind-space vertices, link names, link bind matrices, weights)."""
    def __init__(self, rig, asset=None):
        import fbx
        import csv
        from poseydon.core.skeleton import SkeletonManifest
        from poseydon.io.fbx import FBX, _matrix
        # The vendor skin in its own bind: the FBX of the rest pose's clip, else the rig's first.
        rows = [r for r in csv.DictReader(open("data/truebones/fbx_clips.csv")) if r["rig"] == rig]
        rest = SkeletonManifest.load(f"data/truebones/rigs/{rig}/manifest.yaml").rest_pose
        row = next((r for r in rows if r["bvh"] == rest), rows[0])
        self.scene = FBX.read(asset or Path("data/_raw/truebones/source") / row["file"])
        sc = self.scene.scene; self.parts = []; self.meshes = []
        for i in range(sc.GetSrcObjectCount(fbx.FbxCriteria.ObjectType(fbx.FbxSkin.ClassId))):
            skin = sc.GetSrcObject(fbx.FbxCriteria.ObjectType(fbx.FbxSkin.ClassId), i)
            mesh = skin.GetGeometry(); pts = np.array([[p[0], p[1], p[2]] for p in mesh.GetControlPoints()])
            links, binds, W = [], [], np.zeros((len(pts), skin.GetClusterCount())); mesh_t = None
            for c in range(skin.GetClusterCount()):
                cl = skin.GetCluster(c); m = fbx.FbxAMatrix(); cl.GetTransformLinkMatrix(m); t = fbx.FbxAMatrix(); cl.GetTransformMatrix(t)
                mesh_t = _matrix(t); links.append(cl.GetLink().GetName()); binds.append(_matrix(m))
                for a, w in zip(cl.GetControlPointIndices(), cl.GetControlPointWeights()): W[a, c] += w
            self.parts.append((np.c_[pts, np.ones(len(pts))] @ mesh_t.T, links, np.array(binds), W)); self.meshes.append(mesh)
        self.bind_world = {n: b for _, ls, bs, _ in self.parts for n, b in zip(ls, bs)}


def build(rig, out=PROB):
    """The problem file: rule-based start pose (spine / neck / tail straight, legs down, arms out, right side
    mirrored, feet on one floor), bind, twin pairs, midline, and the skin; plus the mesh triangles."""
    from poseydon.io.bvh import BVH
    from poseydon.core.skeleton import SkeletonManifest, resolve
    from poseydon.build.corpus import rest_source
    from poseydon.core.rotations import quat_to_matrix, matrix_to_quat, quat_between
    from poseydon.ingest.align import axis_vector, facing_quats
    ARM = re.compile(r"arm|hand|wing|finger|wrist|elbow|humer|claw", re.I); KEEP = re.compile(r"clav|scap|collar", re.I)
    TAIL = re.compile(r"tail|tal\d", re.I); HEAD = re.compile(r"head|skull|scull", re.I)
    FACE = re.compile(r"eye|mouth|jaw|nose|lip|beak|mandible|tongue|tone", re.I)
    UP, DOWN, FWD, BACK, OUT = np.array([0., 1, 0]), np.array([0., -1, 0]), np.array([0., 0, 1]), np.array([0., 0, -1]), np.array([1., 0, 0])
    CLEAN = (np.eye(3), np.diag([1., -1, -1]), np.diag([-1., 1, -1]), np.diag([-1., -1, 1]))
    def rot_about(axis, angle):
        a = axis / np.linalg.norm(axis); K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
        return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K
    def swing(world, vec, target): return quat_to_matrix(quat_between(world @ vec, target)) @ world
    def swing_pitch(world, vec, target):   # a near-opposite turn of a midline bone goes through a pitch, never a yaw
        d_ = world @ vec; d_ = d_ / np.linalg.norm(d_)
        if d_ @ target < -0.2: world = rot_about(OUT, np.arctan2(d_[1], d_[2]) - np.arctan2(target[1], target[2])) @ world
        return swing(world, vec, target)
    def midpoint(a, b):
        qa, qb = matrix_to_quat(a), matrix_to_quat(b)
        if qa @ qb < 0: qb = -qb
        return quat_to_matrix((qa + qb) / np.linalg.norm(qa + qb))
    def angle_of(R): return np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))

    m = SkeletonManifest.load(f"data/truebones/rigs/{rig}/manifest.yaml")
    rest_bvh = BVH.read(rest_source(m, sorted((RAW / rig).glob("*.bvh")))); base = rest_bvh.to_animation()
    names, parents = list(base.names), base.parents; n = len(names); kids = children(parents); order = list(range(n))
    skin = Skin(rig); cal = skin.cal; up = axis_vector(m.up_axis); tags = set(m.tags)
    has_pos = [len(rest_bvh.channels[j]) == 6 for j in range(n)]
    # 1. the bind on the BVH rig
    bind = skin.bind_world
    Wb = quat_to_matrix(base.global_transforms()[1][0]); Pb = base.global_positions()[0].copy(); linked = np.zeros(n, bool)
    for j, nm in enumerate(names):
        if nm in cal.joints and nm in bind:
            k = cal.joints.index(nm)
            b = np.linalg.solve(cal.scale * cal.rotation, bind[nm][:3, :3] @ np.linalg.inv(cal.offsets[k]))
            u, _, vt = np.linalg.svd(b); Wb[j] = u @ vt; Pb[j] = cal.to_source(bind[nm][:3, 3]); linked[j] = True
    bone = np.zeros_like(base.offsets); pos0, rot0 = base.global_transforms()
    for j in range(1, n):
        p = parents[j]
        if linked[j] and linked[p]: bone[j] = Wb[p].T @ (Pb[j] - Pb[p])
        else:
            bone[j] = quat_to_matrix(rot0[0, p]).T @ (pos0[0, j] - pos0[0, p])
            if not linked[j]: Wb[j] = Wb[p] @ quat_to_matrix(base.rotations[0, j]); Pb[j] = Pb[p] + Wb[p] @ bone[j]
    bone[np.linalg.norm(bone, axis=1) < 1e-4 * np.ptp(Pb, axis=0).max()] = 0.0   # bind-solve noise is not a bone
    def place(Wx):
        p = np.zeros((n, 3))
        for j in range(1, n): p[j] = p[parents[j]] + Wx[parents[j]] @ bone[j]
        return p
    def facing_frame(P_):
        U = quat_to_matrix(quat_between(up, UP))
        return quat_to_matrix(facing_quats((P_ @ U.T)[None], resolve(m, names).facing_indices, m.extra_yaw_deg)[0]) @ U
    pairs = side_pairs(names, parents, Pb)
    if pairs:
        a = sum(Pb[r] - Pb[l] for r, l in pairs); a -= (a @ up) * up; a /= np.linalg.norm(a); T = np.stack([-a, up, np.cross(up, a)])
    else: T = facing_frame(Pb)
    W = T @ Wb; P = place(W); ext = np.ptp(P, axis=0).max()
    right_of = {l: r for r, l in pairs}; left_of = {r: l for r, l in pairs}
    centre = [j for j in range(n) if j not in right_of and j not in left_of and not names_side(names[j])]
    mode_b = (max((np.linalg.norm(M @ P[r] - P[l]) for r, l in pairs), default=0.0) / ext > 0.01
              or max((abs(P[j, 0]) for j in centre), default=0.0) / ext > 0.01)
    if mode_b:   # an asymmetric bind: the manifest's facing sets the frame
        T = facing_frame(Pb); W = T @ Wb; P = place(W)
        pairs = side_pairs(names, parents, P); right_of = {l: r for r, l in pairs}; left_of = {r: l for r, l in pairs}
        centre = [j for j in range(n) if j not in right_of and j not in left_of and not names_side(names[j])]
    L0 = W.copy(); L0[1:] = np.swapaxes(W[parents[1:]], -1, -2) @ W[1:]
    side = {j: ("c" if j in centre else "l" if j in right_of else "r" if j in left_of else "u") for j in range(n)}
    S, Sc = {}, {}
    if not mode_b:   # a symmetric bind: its mirror conventions are exact
        def snap(R):
            best = max(CLEAN, key=lambda C: np.trace(C.T @ R)); return best if angle_of(best.T @ R) < 15 else R
        S = {r: snap((M @ W[l] @ M).T @ W[r]) for r, l in pairs}; Sc = {c: snap((M @ W[c] @ M).T @ W[c]) for c in centre}
        for r, l in pairs:
            pr = parents[r]
            Sp = S.get(pr) if (pr in left_of and parents[l] == left_of[pr]) else Sc.get(pr) if (pr in Sc and parents[l] == pr) else None
            if Sp is not None and has_pos[r] and has_pos[l]:
                mean = 0.5 * (bone[l] + M @ Sp @ bone[r])
                if np.linalg.norm(mean - bone[l]) <= 0.05 * np.linalg.norm(bone[l]): bone[l], bone[r] = mean, Sp.T @ M @ mean
        for c in centre:
            if parents[c] in Sc and has_pos[c]: bone[c] = 0.5 * (bone[c] + M @ Sc[parents[c]] @ bone[c])
    # 2. classification and rule targets
    def pick_main(j):
        cand = [k for k in kids[j] if np.linalg.norm(bone[k]) > 1e-8]; same = [k for k in cand if side[k] == side[j]] or cand
        return max(same, key=lambda k: (len(subtree(kids, k)), np.linalg.norm(bone[k])), default=None)
    main = [pick_main(j) for j in range(n)]
    def path_from(j):
        o = [j]
        while main[o[-1]] is not None: o.append(main[o[-1]])
        return o
    P = place(W); height = P[:, 1]; floor, reach = height.min(), np.ptp(height)
    limbs = [l for l in right_of if parents[l] not in right_of]; legs, arms = [], []
    for l in limbs:
        sub_ = subtree(kids, l); grounded = height[sub_].min() - floor < 0.15 * reach; armish = any(ARM.search(names[j]) for j in sub_)
        if (grounded and tags & {"quadruped", "biped", "flying"}) or ("quadruped" in tags and armish): legs.append(l)
        elif tags & {"biped", "flying"} and armish: arms.append(l)
    heads = [j for j in centre if HEAD.search(names[j])]; head_path = []
    if heads:
        k = heads[0]
        while k >= 0: head_path.append(k); k = parents[k]
        head_path = head_path[::-1]
    upright = bool(heads) and "biped" in tags and abs((P[head_path[-1]] - P[0])[1]) / (np.linalg.norm(P[head_path[-1]] - P[0]) + 1e-9) > np.cos(np.radians(45))
    target, cat, foot_root = {}, {}, {}
    if "snake" in tags:
        hs = set(heads)
        for j in centre:
            if main[j] is not None: target[j] = FWD if any(k in hs for k in subtree(kids, main[j])) else BACK; cat[(j, main[j])] = "snake"
    else:
        if tags & {"quadruped", "biped", "flying"}:
            for j in head_path:
                if main[j] is not None and j in centre:
                    nxt = head_path[head_path.index(j) + 1] if j != head_path[-1] else main[j]
                    if np.linalg.norm(bone[nxt]) > 1e-8: target[(j, nxt)] = UP if upright else FWD; cat[(j, nxt)] = "spine"
            for j in centre:
                if j and main[j] is not None and TAIL.search(names[main[j]]): target[(j, main[j])] = BACK; cat[(j, main[j])] = "tail"
        for l in legs:
            path = path_from(l); below = np.cumsum([np.linalg.norm(bone[k]) for k in path[:0:-1]])[::-1]
            short = next((i for i in range(len(path) - 1) if below[i] < 0.2 * below[0]), len(path) - 1); fi = max(short, len(path) - 3)
            for i, j in enumerate(path[:-1]):
                if KEEP.search(names[j]):
                    d_ = W[j] @ bone[path[i + 1]]; d_ = d_ / np.linalg.norm(d_); d_ = np.array([0.25, -1.0, d_[2]])
                    target[(j, path[i + 1])] = d_ / np.linalg.norm(d_); cat[(j, path[i + 1])] = "shoulder"
                else: target[(j, path[i + 1])] = FWD if i >= fi else DOWN; cat[(j, path[i + 1])] = "foot" if i >= fi else "leg"
            foot_root[l] = path[fi]
            for k in subtree(kids, path[fi]):
                if k in path or main[k] is None: continue
                d_ = (W[k] @ bone[main[k]]) * [1, 0, 1]
                target[(k, main[k])] = d_ / np.linalg.norm(d_) if np.linalg.norm(d_) > 1e-9 else FWD; cat[(k, main[k])] = "toe"
        for l in arms:
            path = path_from(l)
            for i, j in enumerate(path[:-1]):
                if KEEP.search(names[j]):
                    d_ = W[j] @ bone[path[i + 1]]; d_ = d_ / np.linalg.norm(d_); d_[1] = -abs(d_[1]); target[(j, path[i + 1])] = d_; cat[(j, path[i + 1])] = "shoulder"
                else: target[(j, path[i + 1])] = OUT; cat[(j, path[i + 1])] = "arm"
    aimed = {j: (c, t) for (j, c), t in ((k, v) for k, v in target.items() if isinstance(k, tuple))}
    face = {}
    if heads and "snake" not in tags:
        h = head_path[-1]; fj = [k for k in subtree(kids, h) if FACE.search(names[k]) and k != h]
        fj = [k for k in fj if not any(FACE.search(names[a_]) for a_ in [parents[k]] if a_ != h)] or fj
        if fj:
            v = P[fj].mean(axis=0) - P[h]; v[0] = 0.0
            if np.linalg.norm(v) > 1e-6 * ext: face[h] = W[h].T @ (v / np.linalg.norm(v)); aimed.pop(h, None)
    aimed.update({j: (main[j], t) for j, t in target.items() if not isinstance(j, tuple)})
    regions = {}
    for v_, links, binds, Wt in skin.parts:
        dom = Wt.argmax(1); strong = Wt.max(1) > 0.6
        for c_, name in enumerate(links):
            sel = v_[(dom == c_) & strong]
            if len(sel) >= 4: regions.setdefault(name, []).append((sel, binds[c_]))
    def deformed(j, Wj, pj):
        name = names[j]
        if name not in regions or name not in cal.joints: return None
        k = cal.joints.index(name); G = np.eye(4)
        G[:3, :3] = cal.scale * cal.rotation @ (T.T @ Wj) @ cal.offsets[k]; G[:3, 3] = cal.scale * (T.T @ pj) @ cal.rotation.T + cal.translation
        return cal.to_source(np.concatenate([(G @ np.linalg.inv(B) @ v_.T).T[:, :3] for v_, B in regions[name]])) @ T.T
    def chamfer(a_, b_): return 0.5 * (cKDTree(b_).query(a_)[0].mean() + cKDTree(a_).query(b_)[0].mean())
    grid = np.radians(np.arange(-90, 90.5, 1.0))
    def refine(cost):
        a0 = min(grid, key=cost); return min(a0 + np.radians(np.arange(-1, 1.01, 0.05)), key=cost)
    # 3a. the centre: aimed or on the midline, then its own mirror
    rolls = np.radians(np.arange(-90, 90.25, 0.25))
    for j in order:
        if j not in centre: continue
        start = W[j] if j == 0 else W[parents[j]] @ L0[j]
        if j in face: start = swing_pitch(start, face[j], FWD)
        elif j in aimed: c, t = aimed[j]; start = swing_pitch(start, bone[c], t)
        elif main[j] is not None and j:
            d_ = (start @ bone[main[j]]) * [0, 1, 1]
            if np.linalg.norm(d_) > 1e-9: start = swing_pitch(start, bone[main[j]], d_ / np.linalg.norm(d_))
        W[j] = start; cc = aimed[j][0] if j in aimed else main[j]
        if cc is None and j not in face: continue
        axis = FWD if j in face else W[j] @ bone[cc]; axis = axis / np.linalg.norm(axis)
        if j in Sc:
            W[j] = min((rot_about(axis, a_) @ W[j] for a_ in rolls), key=lambda R: np.sum((M @ R @ M @ Sc[j] - R) ** 2))
            W[j] = midpoint(W[j], M @ W[j] @ M @ Sc[j]); W[j] = swing(W[j], face[j], axis) if j in face else swing(W[j], bone[cc], axis)
        else:
            ch = [k for k in kids[j] if np.linalg.norm(bone[k]) > 1e-8]
            twin_ = [(a_, left_of[a_]) for a_ in ch if a_ in left_of and left_of[a_] in ch]; mids = [k for k in ch if k in centre and k != cc]
            pj = place(W)[j]
            if twin_:
                e1 = sum(bone[b_] - bone[a_] for a_, b_ in twin_); e1 = e1 / np.linalg.norm(e1)
                if j in aimed: a_v, t_v = bone[aimed[j][0]], aimed[j][1]
                elif cc is not None and cc in centre: a_v = bone[cc]; t_v = (W[j] @ a_v) * [0, 1, 1]
                else: a_v = W[j].T @ UP; t_v = UP
                def frame(x, v):
                    v = v - (v @ x) * x; v = v / np.linalg.norm(v); return np.stack([x, v, np.cross(x, v)], 1)
                W[j] = frame(OUT, np.asarray(t_v, float)) @ frame(e1, np.asarray(a_v, float)).T; continue
            if mids:
                cost = lambda th, mids=mids, axis=axis, j=j: sum((rot_about(axis, th) @ W[j] @ bone[k])[0] ** 2 for k in mids)
            elif deformed(j, W[j], pj) is not None:
                cost = lambda th, j=j, axis=axis, pj=pj: (lambda d_: chamfer(d_, d_ * [-1, 1, 1]))(deformed(j, rot_about(axis, th) @ W[j], pj))
            else: continue
            W[j] = rot_about(axis, refine(cost)) @ W[j]
    # 3b. the left side, hung from the centre; hands palms down
    for j in order:
        if side[j] in ("l", "u"):
            start = W[parents[j]] @ L0[j]
            if j in aimed: c, t = aimed[j]; start = swing(start, bone[c], t)
            W[j] = start
    for l in arms:
        for j in path_from(l):
            ch = [k for k in kids[j] if np.linalg.norm(bone[k]) > 1e-8]
            if j not in aimed or len(ch) < 2 or np.linalg.matrix_rank(bone[ch], tol=1e-6) < 2: continue
            axis = aimed[j][1]; costs = [(np.var([(rot_about(axis, a_) @ W[j] @ bone[k])[1] for k in ch]), a_) for a_ in np.radians(np.arange(-90, 90.5, 0.5))]
            best = min(costs)[0]; R = rot_about(axis, min((a_ for c_, a_ in costs if c_ <= best * 1.0001 + 1e-12), key=abs))
            for k in subtree(kids, j): W[k] = R @ W[k]
    # 3c. the right side
    if not mode_b:
        for r, l in pairs: W[r] = M @ W[l] @ M @ S[r]
    else:
        for r in order:
            if side[r] != "r": continue
            l = left_of[r]; start = W[parents[r]] @ L0[r]; cl = main[l]
            if cl is None: W[r] = W[parents[r]] @ (M @ (W[parents[l]].T @ W[l]) @ M); continue
            cr = left_of.get(cl, main[r])
            if cr is None or np.linalg.norm(bone[cr]) < 1e-8: W[r] = start; continue
            d_ = M @ (W[l] @ bone[cl]); d_ /= np.linalg.norm(d_); W[r] = swing(start, bone[cr], d_)
            twin_ = [(a_, left_of[a_]) for a_ in kids[r] if a_ in left_of and parents[left_of[a_]] == l and a_ != cr and np.linalg.norm(bone[a_]) > 1e-8]
            pos = place(W)
            if twin_: cost = lambda th, r=r, l=l, twin_=twin_, d_=d_: sum(np.sum((rot_about(d_, th) @ W[r] @ bone[a_] - M @ W[l] @ bone[b_]) ** 2) for a_, b_ in twin_)
            elif deformed(r, W[r], pos[r]) is not None and deformed(l, W[l], pos[l]) is not None:
                ref = deformed(l, W[l], pos[l]) * [-1, 1, 1]
                cost = lambda th, r=r, ref=ref, d_=d_, pos=pos: chamfer(deformed(r, rot_about(d_, th) @ W[r], pos[r]), ref)
            else: cost = None
            if cost is not None: W[r] = rot_about(d_, refine(cost)) @ W[r]
        for r, l in pairs:
            pr, pl = parents[r], parents[l]
            if has_pos[r] and has_pos[l] and ((pr in left_of and pl == left_of[pr]) or (pr == pl and pr in centre)):
                b_ = W[pr].T @ (M @ (W[pl] @ bone[l]))
                if np.linalg.norm(b_ - bone[r]) <= 0.05 * np.linalg.norm(bone[r]): bone[r] = b_
        for c in centre:
            if c and has_pos[c] and parents[c] in centre:
                v = W[parents[c]] @ bone[c]; Lg = np.linalg.norm(v); v = v * [0, 1, 1]
                if np.linalg.norm(v) > 1e-9: bone[c] = W[parents[c]].T @ (v / np.linalg.norm(v) * Lg)
        S = {r: (M @ W[l] @ M).T @ W[r] for r, l in pairs}
    # 4. feet on one floor (foot mesh): the longer legs bend
    def leg_bottom(l, Wx):
        p_ = place(Wx); lows = [d_[:, 1].min() for j in subtree(kids, l) for d_ in [deformed(j, Wx[j], p_[j])] if d_ is not None]
        return min(lows) if lows else p_[subtree(kids, l), 1].min()
    if len(legs) >= 2:
        P0 = place(W); front_z = max(P0[l, 2] for l in legs); bottoms = {l: leg_bottom(l, W) for l in legs}; floor_y = max(bottoms.values())
        for l in legs:
            drop = floor_y - bottoms[l]
            if drop < 1e-5 * ext: continue
            down = [j for j in path_from(l)[:-1] if j in aimed and np.allclose(aimed[j][1], DOWN)]
            if len(down) < 2: continue
            a_len, b_len = np.linalg.norm(bone[aimed[down[0]][0]]), np.linalg.norm(bone[aimed[down[1]][0]]); h = a_len + b_len - drop
            if h <= abs(a_len - b_len): continue
            alpha = np.arccos(np.clip((a_len ** 2 + h ** 2 - b_len ** 2) / (2 * a_len * h), -1, 1))
            sense = -1.0 if (P0[l, 2] >= front_z - 1e-9 and "quadruped" in tags) else 1.0
            top = P0[down[0]]; knee = top + a_len * (rot_about(OUT, -sense * alpha) @ DOWN); end = top + np.array([0.0, -h, 0.0])
            for j, tgt in ((down[0], knee - top), (down[1], end - knee)): W[j] = swing(W[j], bone[aimed[j][0]], tgt / np.linalg.norm(tgt))
        for r, l in pairs: W[r] = M @ W[l] @ M @ S[r]
    P = place(W)
    # the skin: each vertex's (up to 4) influences, its offset in the joint's frame, its weight
    inf_j, inf_u, inf_w, keep_all = [], [], [], []
    scene = skin.scene; fbx_parent = {nm: (scene.names[scene.parents[i]] if scene.parents[i] >= 0 else None) for i, nm in enumerate(scene.names)}
    def driver(nm):   # an FBX node not on the BVH rig follows its nearest ancestor that is
        while nm is not None and not (nm in cal.joints and nm in names): nm = fbx_parent.get(nm)
        return nm
    for v_, links, binds, Wt in skin.parts:
        top = np.argsort(-Wt, axis=1)[:, :4]; keep = np.zeros(len(v_), bool)
        for vi in np.where(Wt.sum(1) > 0)[0]:
            js, us, ws = [], [], []
            for c_ in top[vi]:
                w_ = Wt[vi, c_]; d_ = driver(links[c_]) if w_ > 0 else None
                if d_ is None: continue
                k = cal.joints.index(d_); B = skin.bind_world[d_] if d_ in skin.bind_world else binds[c_]
                if d_ != links[c_]: B = binds[c_] @ np.linalg.inv(skin.bind_world[links[c_]]) @ B if links[c_] in skin.bind_world else B
                js.append(names.index(d_)); us.append(cal.offsets[k] @ (np.linalg.inv(B) @ v_[vi])[:3]); ws.append(w_)
            if not js: continue
            keep[vi] = True; ws = np.array(ws) / sum(ws); pad = 4 - len(js)
            inf_j.append(js + [0] * pad); inf_u.append(us + [np.zeros(3)] * pad); inf_w.append(list(ws) + [0.0] * pad)
        keep_all.append(keep)
    tris, basev = [], 0   # triangles, indexed like the kept vertices
    for mesh, keep in zip(skin.meshes, keep_all):
        new = np.cumsum(keep) - 1 + basev
        for p in range(mesh.GetPolygonCount()):
            poly = [mesh.GetPolygonVertex(p, k) for k in range(mesh.GetPolygonSize(p))]
            if all(keep[q] for q in poly): tris += [(new[poly[0]], new[poly[k]], new[poly[k + 1]]) for k in range(1, len(poly) - 1)]
        basev += int(keep.sum())
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    np.savez(out / f"{rig}_problem.npz", names=np.array(names), parents=parents, bone=bone, W=W, Wbind=T @ Wb, T=T,
             offsets=base.offsets, fps=base.fps, pairs=np.array(pairs, dtype=int).reshape(-1, 2), centre=np.array(centre, dtype=int),
             legs=np.array(legs, dtype=int), inf_j=np.array(inf_j, dtype=int), inf_u=np.array(inf_u), inf_w=np.array(inf_w))
    np.savez(out / f"{rig}_faces.npz", tris=np.array(tris, dtype=int))
    print(json.dumps({"rig": rig, "mode": "B" if mode_b else "A", "joints": n, "vertices": len(inf_j), "triangles": len(tris),
                      "pairs": len(pairs), "legs": [names[l] for l in legs]}), flush=True)


# ========================================================================================================= pose
def load(rig):
    return dict(np.load(PROB / f"{rig}_problem.npz", allow_pickle=True))


def config(rig, base=None):
    p = CONFIGS / f"{rig}.yaml"
    cfg = yaml.safe_load(p.read_text()) if p.exists() else yaml.safe_load(RIGS.get(rig, "")) or {}
    return {"base": base} if base else cfg


def clip_frames(rig, f, d):
    """A vendor clip's global turns G (F, n, 3, 3) and positions Q (F, n, 3), canonical, on the problem's joints
    (joints the clip lacks, end sites some BVHs drop, take their parent's turn)."""
    from poseydon.io.bvh import BVH
    from poseydon.core.rotations import quat_to_matrix
    a = BVH.read(RAW / rig / f).to_animation()
    names = [str(x) for x in d["names"]]; par = d["parents"]; idx = {nm: i for i, nm in enumerate(a.names)}
    Ga = d["T"] @ quat_to_matrix(a.global_transforms()[1]); Qa = a.global_positions() @ d["T"].T
    G = np.zeros((len(Ga), len(names), 3, 3)); Q = np.zeros((len(Ga), len(names), 3))
    for j, nm in enumerate(names):
        if nm in idx: G[:, j] = Ga[:, idx[nm]]; Q[:, j] = Qa[:, idx[nm]]
        else: G[:, j] = G[:, par[j]]; Q[:, j] = Q[:, par[j]] + G[:, par[j]] @ d["bone"][j]
    return G, Q


def pose(rig, cfg):
    """Apply the config's ops to the start pose. Returns (problem, W world turns, P positions, V skin vertices)."""
    from scipy.optimize import minimize, differential_evolution
    d = load(rig); names = [str(x) for x in d["names"]]; parents = d["parents"]; n = len(names)
    base = str(cfg.get("base", "rule"))
    if base.startswith("clip:"): _, f, k = base.split(":"); W = clip_frames(rig, f, d)[0][int(k)]
    elif base == "bind": W = d["Wbind"].copy()
    else: W = d["W"].copy()
    if cfg.get("root_from_rule"): W[0] = d["W"][0]
    kids_ = children(parents)
    for name in cfg.get("local_from_rule") or []:   # a joint given its rule-base turn on its parent; below keeps its world turn
        j = names.index(name); W[j] = W[parents[j]] @ d["W"][parents[j]].T @ d["W"][j]
    twin = {}
    for r, l in d["pairs"]: twin[int(r)] = int(l); twin[int(l)] = int(r)
    kids = children(parents)
    def sub(j): return subtree(kids, j)
    def place():
        P_ = np.zeros((n, 3))
        for j in range(1, n): P_[j] = P_[parents[j]] + W[parents[j]] @ d["bone"][j]
        return P_
    def skin_():
        P_ = place(); return P_, (d["inf_w"][..., None] * (np.einsum("vkab,vkb->vka", W[d["inf_j"]], d["inf_u"]) + P_[d["inf_j"]])).sum(1)
    dom_ = d["inf_j"][np.arange(len(d["inf_j"])), d["inf_w"].argmax(1)]
    centre = set(int(c) for c in d["centre"])
    def say(s): print(f"  {rig}: {s}", flush=True)
    if cfg.get("clip_local"):
        cl = cfg["clip_local"]; Wc = clip_frames(rig, cl["clip"], d)[0][cl.get("frame", 0)]
        for name in cl["joints"]:
            j = names.index(name); p = parents[j]
            for q in sub(j): W[q] = W[p] @ Wc[p].T @ Wc[q]
    def sym_skin(entries):
        for name in entries or []:
            j = names.index(name); m = np.isin(dom_, [q for q in sub(j) if q in centre]); W0s = {q: W[q].copy() for q in sub(j)}
            def turn(yz):
                R = Rotation.from_rotvec([0, yz[0], yz[1]]).as_matrix()
                for q in sub(j): W[q] = R @ W0s[q]
            tw_kids = [(int(l), int(r)) for r, l in d["pairs"] if parents[int(l)] == j and parents[int(r)] == j]
            def cost(yz):
                turn(yz)
                if tw_kids: P_ = place(); return sum(np.linalg.norm(P_[l] - P_[r] * [-1, 1, 1]) for l, r in tw_kids)
                X = skin_()[1][m]; return cKDTree(X * [-1, 1, 1]).query(X)[0].mean()
            best = minimize(cost, np.zeros(2), method="Nelder-Mead", options={"xatol": 1e-4, "fatol": 1e-6}); turn(best.x)
            say(f"{name} yaw/roll {np.round(np.degrees(best.x), 1)} deg for a symmetric {'skeleton' if tw_kids else 'skin'}")
    sym_skin(cfg.get("sym_skin"))
    if cfg.get("clip_twist"):
        ct = cfg["clip_twist"]; Wc = clip_frames(rig, ct["clip"], d)[0][ct.get("frame", 0)]
        for name in ct["joints"]:
            j = names.index(name); p = parents[j]
            c = max((k for k in kids[j] if k in centre and np.linalg.norm(d["bone"][k]) > 1e-9), key=lambda k: len(sub(k)))
            Wt = W[p] @ Wc[p].T @ Wc[j]; a_, b_ = W[j] @ d["bone"][c], Wt @ d["bone"][c]
            W[j] = Rotation.align_vectors([a_ / np.linalg.norm(a_)], [b_ / np.linalg.norm(b_)])[0].as_matrix() @ Wt
    if cfg.get("roll_to_bind"):   # each midline joint rolled about its own bone until its reference up (and left) axes point up / left:
        X_, Y_ = np.array([1.0, 0, 0]), np.array([0, 1.0, 0])   # the rule base leaves a straight chain's rolls arbitrary (a twisted snake);
        rb = cfg["roll_to_bind"]; Wref = d["Wbind"]; wl = 1.0   # reference: the bind (true) or "clip:frame" (a frame lying flat; up only,
        if isinstance(rb, str): f_, k_ = rb.split(":"); Wref = clip_frames(rig, f_, d)[0][int(k_)]; wl = 0.0   # its heading is arbitrary)
        for j in sorted(centre):   # only the joint's own turn changes, so every joint stays where it is
            c = max(kids[j], key=lambda c: np.linalg.norm(d["bone"][c]), default=None)
            if c is None or np.linalg.norm(d["bone"][c]) < 1e-9:   # a chain tip: straight on from its parent (its parent's turn)
                if j and parents[j] in centre: W[j] = W[parents[j]].copy()
                continue
            ax = W[j] @ d["bone"][c]; ax /= np.linalg.norm(ax); au, al = Wref[j].T @ Y_, Wref[j].T @ X_
            th = min(np.radians(np.arange(-180, 180, 0.25)), key=lambda t_: -((Rotation.from_rotvec(t_ * ax).as_matrix() @ W[j] @ au) @ Y_ + wl * (Rotation.from_rotvec(t_ * ax).as_matrix() @ W[j] @ al) @ X_))
            W[j] = Rotation.from_rotvec(th * ax).as_matrix() @ W[j]
    for name, axis, deg in cfg.get("rotate") or []:
        j = names.index(name); R = Rotation.from_rotvec(np.radians(deg) * np.eye(3)["xyz".index(axis)]).as_matrix()
        for k in sub(j): W[k] = R @ W[k]
        if j in twin:
            for k in sub(twin[j]): W[k] = M @ R @ M @ W[k]
    def aim(entries):
        for name, child, dirn, *one in entries or []:
            j, c = names.index(name), names.index(child)
            for jj, cc, dd in [(j, c, np.array(dirn, float))] + ([(twin[j], twin[c], np.array(dirn, float) * [-1, 1, 1])] if j in twin and not one else []):
                v = W[jj] @ d["bone"][cc]; R = Rotation.align_vectors([dd / np.linalg.norm(dd)], [v / np.linalg.norm(v)])[0].as_matrix()
                for q in sub(jj): W[q] = R @ W[q]
    aim(cfg.get("aim"))
    for entry in cfg.get("palms_down") or []:   # a hand rolled about its arm's axis until its fingers lie flat (smallest such
        name, root, *mode = (entry, entry) if isinstance(entry, str) else entry   # roll); [hand, shoulder]: the whole arm rolls (no wrist
        flat_skin = bool(mode) and mode[0] == "skin"   # twist); a 3rd entry "skin": the hand's skin made flat (feathers), not its finger joints
        for j, j1 in [(names.index(name), names.index(root))] + ([(twin[names.index(name)], twin[names.index(root)])] if names.index(name) in twin else []):
            ch = [k for k in kids[j] if np.linalg.norm(d["bone"][k]) > 1e-9]; P_ = place()
            axis = (P_[j] - P_[parents[j]]) / np.linalg.norm(P_[j] - P_[parents[j]]); W0h = {q: W[q].copy() for q in sub(j1)}
            tips = [max(sub(k), key=lambda q: np.linalg.norm(P_[q] - P_[j])) for k in ch]
            def spread(a_):
                R = Rotation.from_rotvec(np.radians(a_) * axis).as_matrix()
                for q in sub(j1): W[q] = R @ W0h[q]
                if flat_skin: return np.var(skin_()[1][np.isin(dom_, sub(j)), 1])
                Pq = place(); return np.var(Pq[tips + ch, 1])
            costs = [(spread(a_), a_) for a_ in np.arange(-180, 180, 0.5)]; lo_ = min(costs)[0]
            a_ = min((a_ for c_, a_ in costs if c_ <= lo_ * 1.01 + 1e-9), key=abs); spread(a_); say(f"{names[j]} rolled {a_:.1f} deg, fingers flat")
    for name in cfg.get("level") or []:
        if name.startswith("^"):   # the bone into the chain levelled too, by pitching the parent (its other children keep their turn)
            name = name[1:]; c0 = names.index(name.partition(">")[0]); p0 = parents[c0]
            sib = max((k for k in kids[p0] if k != c0 and k in centre and np.linalg.norm(d["bone"][k]) > 1e-9), key=lambda k: len(sub(k)), default=None)
            v = W[p0] @ ((d["bone"][sib] - d["bone"][c0]) if sib is not None else d["bone"][c0])
            v = v / np.linalg.norm(v); R = Rotation.align_vectors([v * [1, 0, 1] / np.linalg.norm(v * [1, 0, 1])], [v])[0].as_matrix()
            keep = {q: W[q].copy() for k in kids[p0] if k != c0 for q in sub(k)}
            for q in sub(p0): W[q] = R @ W[q]
            for q, Wq in keep.items(): W[q] = Wq
        name, _, stop = name.partition(">")
        for j0 in [names.index(name)] + ([twin[names.index(name)]] if names.index(name) in twin else []):
            keep = {q: W[q].copy() for q in sub(j0) if q not in centre} if j0 in centre else {}   # limbs keep their turn
            if stop: keep.update({q: W[q].copy() for q in sub(names.index(stop))})
            chain = [j0]
            while stop not in [names[c] for c in chain]:
                nxt = [k for k in kids[chain[-1]] if np.linalg.norm(d["bone"][k]) > 1e-9 and (k in centre) == (j0 in centre)]
                if not nxt: break
                chain.append(max(nxt, key=lambda k: len(sub(k))))
            P_ = place()
            if j0 in centre: head = np.array([0, 0, 1.0 if P_[chain[-1], 2] > P_[chain[0], 2] else -1.0])   # the way the chain runs
            else: head = (P_[chain[-1]] - P_[chain[0]]) * [1, 0, 1]; head /= np.linalg.norm(head)
            for j, k in zip(chain, chain[1:]):
                v = W[j] @ d["bone"][k]; v /= np.linalg.norm(v); R = Rotation.align_vectors([head], [v])[0].as_matrix()
                for q in sub(j): W[q] = R @ W[q]
            e = chain[-1]; me = dom_ == e   # a tip with no bone: its skin's centroid aimed
            if not stop and not any(np.linalg.norm(d["bone"][k]) > 1e-9 for k in kids[e]) and me.sum() >= 3:
                P_, V_ = skin_(); v = V_[me].mean(0) - P_[e]; v /= np.linalg.norm(v); R = Rotation.align_vectors([head], [v])[0].as_matrix()
                for q in sub(e): W[q] = R @ W[q]
            if keep: W[list(keep)] = np.array(list(keep.values()))
    for root, end, *gap in cfg.get("legs_down") or []:   # a leg straight down, perpendicular to a level spine, splayed out
        j0, je = names.index(root), names.index(end); chain = [j0]   # only as far as needed for its foot to clear its twin's
        while chain[-1] != je: chain.append(next(k for k in kids[chain[-1]] if je in sub(k)))
        W0l = W.copy(); ml, mr = np.isin(dom_, sub(je)), np.isin(dom_, sub(twin[je])); margin = (gap[0] if gap else 0.02)
        def splay(s_):
            W[:] = W0l
            for jj, side_ in ((j0, 1.0), (twin[j0], -1.0)):
                ch = [jj]
                while len(ch) < len(chain): ch.append(next(k for k in kids[ch[-1]] if (twin[k] if side_ < 0 else k) in sub(chain[len(ch)])))
                for a_, b_ in zip(ch, ch[1:]):
                    tgt = np.array([side_ * s_, -1.0, 0.0]); tgt /= np.linalg.norm(tgt); v = W[a_] @ d["bone"][b_]
                    R = Rotation.align_vectors([tgt], [v / np.linalg.norm(v)])[0].as_matrix()
                    for q in sub(a_): W[q] = R @ W[q]
            V_ = skin_()[1]; return V_[ml, 0].min() - V_[mr, 0].max()   # left foot's inner edge minus the right's
        ext_ = np.ptp(skin_()[1], axis=0).max(); s_ = next((x for x in np.arange(0, 1.0, 0.005) if splay(x) >= margin * ext_), 1.0); splay(s_)
        say(f"{root} straight down, splayed {np.degrees(np.arctan(s_)):.1f} deg so the feet clear by {splay(s_):.1f}")
    def pca(m):
        X = skin_()[1][m]; X = X - X.mean(0); w, v = np.linalg.eigh(X.T @ X); return v[:, ::-1].T, w[::-1]
    def msk(name):   # the right end joints turned until their skin is the left's mirror image
        for l in sub(names.index(name)):
            if l not in twin or any(np.isin(dom_, sub(k)).any() for k in kids[l]): continue
            r = twin[l]; ml, mr = np.isin(dom_, sub(l)), np.isin(dom_, sub(r))
            if ml.sum() < 3 or mr.sum() < 3: ml = ((d["inf_j"] == l) & (d["inf_w"] > 0.1)).any(1); mr = ((d["inf_j"] == r) & (d["inf_w"] > 0.1)).any(1)
            if ml.sum() < 3 or mr.sum() < 3: continue
            target = cKDTree(skin_()[1][ml] * [-1, 1, 1]); W0r = W[r].copy()
            def cost(rv):
                W[r] = Rotation.from_rotvec(rv).as_matrix() @ W0r; return target.query(skin_()[1][mr])[0].mean()
            best = min((minimize(cost, x0, method="Nelder-Mead", options={"xatol": 1e-4, "fatol": 1e-6}) for x0 in (np.zeros(3), np.array([0, 0, np.pi / 2]), np.array([0, 0, -np.pi / 2]))), key=lambda o: o.fun)
            W[r] = Rotation.from_rotvec(best.x).as_matrix() @ W0r
    for name in cfg.get("mirror_skin") or []: msk(name)
    if cfg.get("stance_leg"):
        G, Q = clip_frames(rig, cfg["stance_leg"]["clip"], d); sp = [names.index(x) for x in cfg["stance_leg"]["body"]]
        for entry in cfg["stance_leg"]["legs"]:
            name, under = (entry, entry) if isinstance(entry, str) else entry
            for j, u in [(names.index(name), names.index(under))] + ([(twin[names.index(name)], twin[names.index(under)])] if names.index(name) in twin else []):
                def depth(q):
                    k = 0
                    while q != j: q = parents[q]; k += 1
                    return k
                foot = max((q for q in sub(j) if np.linalg.norm(d["bone"][q]) > 1e-9), key=depth)
                fwd = (Q[:, sp[1]] - Q[:, sp[0]]) * [1, 0, 1]; fwd /= np.linalg.norm(fwd, axis=1, keepdims=True)
                h = Q[:, foot, 1]; low = h <= h.min() + 0.1 * np.ptp(h)   # planted frames, then the one with the paw under its root
                off = np.abs(((Q[:, foot] - Q[:, u]) * fwd).sum(1)); off[~low] = np.inf; f = int(np.argmin(off))
                yaw = Rotation.from_rotvec(-np.arctan2(fwd[f, 0], fwd[f, 2]) * np.array([0, 1.0, 0])).as_matrix()
                for q in sub(j): W[q] = yaw @ G[f, q]
                v = (place()[foot] - place()[u]) * [1, 1, 0]   # swung about the forward axis until the paw is under its root
                R = Rotation.from_rotvec(np.arctan2(-v[0], -v[1]) * np.array([0, 0, 1.0])).as_matrix()
                for q in sub(j): W[q] = R @ W[q]
                say(f"{names[j]} leg as in {cfg['stance_leg']['clip']} frame {f}")
    if cfg.get("torso_pca"):
        axes, _ = pca(np.isin(dom_, sorted(centre))); a = next(v for v in axes if abs(v[0]) < 0.5); a = a if a[2] > 0 else -a
        R = Rotation.from_rotvec(np.arctan2(a[1], a[2]) * np.array([1.0, 0, 0])).as_matrix()
        keep = {q: W[q].copy() for q in range(n) if q not in centre}; W[:] = R @ W
        for q, Wq in keep.items(): W[q] = Wq
    for name, dirn in cfg.get("aim_pca") or []:   # a limb part turned whole until its skin's long (PCA) axis points along dirn
        j0 = names.index(name); m = np.isin(dom_, sub(j0)); axes, _ = pca(m); a = axes[0]
        if a @ (skin_()[1][m].mean(0) - place()[j0]) < 0: a = -a
        t_ = np.array(dirn, float); R = Rotation.align_vectors([t_ / np.linalg.norm(t_)], [a])[0].as_matrix()
        for q in sub(j0): W[q] = R @ W[q]
        if j0 in twin:
            for q in sub(twin[j0]): W[q] = M @ R @ M @ W[q]
    for name in cfg.get("level_pca") or []:
        j0 = names.index(name); m = np.isin(dom_, sub(j0)); axes, _ = pca(m); a = axes[0]
        if a @ (skin_()[1][m].mean(0) - place()[j0]) < 0: a = -a
        R = Rotation.align_vectors([a * [1, 0, 1] / np.linalg.norm(a * [1, 0, 1])], [a])[0].as_matrix()
        for q in sub(j0): W[q] = R @ W[q]
        if j0 in twin:
            for q in sub(twin[j0]): W[q] = M @ R @ M @ W[q]
    if cfg.get("stance"):
        G, Q = clip_frames(rig, cfg["stance"]["clip"], d); prs = [(int(r), int(l)) for r, l in d["pairs"]]
        for name in cfg["stance"]["feet"]:
            for j in [names.index(name)] + ([twin[names.index(name)]] if names.index(name) in twin else []):
                speed = np.r_[np.inf, np.linalg.norm(np.diff(Q[:, j], axis=0), axis=1)]; f = int(np.argmin(Q[:, j, 1] - Q[:, j, 1].min() + 0.2 * speed))
                across = sum(Q[f, r] - Q[f, l] for r, l in prs) * [1, 0, 1]
                R = Rotation.align_vectors([[-1, 0, 0]], [across / np.linalg.norm(across)])[0].as_matrix() @ G[f, j] @ W[j].T
                for q in sub(j): W[q] = R @ W[q]
                toe = max(kids[j], key=lambda k: np.linalg.norm(d["bone"][k]), default=None)
                if toe is not None and np.linalg.norm(d["bone"][toe]) > 1e-9: fw = (W[j] @ d["bone"][toe]) * [1, 0, 1]
                else: fw = (skin_()[1][np.isin(dom_, sub(j))].mean(0) - place()[j]) * [1, 0, 1]
                fw /= np.linalg.norm(fw); Ry = Rotation.from_rotvec(np.arctan2(-fw[0], fw[2]) * np.array([0, 1.0, 0])).as_matrix()
                for q in sub(j): W[q] = Ry @ W[q]
    def plant(entries):
        for entry in entries or []:   # a foot rocked until most sole vertices touch, then the flattest; each side on its own
            name, axes_, *opt = (entry, "x") if isinstance(entry, str) else entry; lim = opt[0] if opt else None; one = len(opt) > 1
            for j in [names.index(name)] + ([twin[names.index(name)]] if names.index(name) in twin and not one else []):
                m = np.isin(dom_, sub(j)); W0_ = {q: W[q].copy() for q in sub(j)}
                def sole(ang):
                    R = Rotation.from_rotvec(np.radians(ang[0]) * np.array([1.0, 0, 0])).as_matrix()
                    if len(ang) > 1: R = Rotation.from_rotvec(np.radians(ang[1]) * np.array([0, 0, 1.0])).as_matrix() @ R
                    for q in sub(j): W[q] = R @ W0_[q]
                    y = np.sort(skin_()[1][m, 1]); tol = 0.06 * (y[-1] - y[0])
                    return -np.mean(y - y[0] < tol) + 0.01 * (y[: max(3, len(y) // 5)].mean() - y[0]) / tol
                if axes_ == "x":
                    th = min(np.arange(-70, 70.5, 1.0), key=lambda a: sole([a])); best = [min(th + np.arange(-1, 1.01, 0.1), key=lambda a: sole([a]))]
                else:
                    la, lb = (lim, lim) if lim else (45, 25)
                    a0, b0 = min([(a, b) for a in np.arange(-la, la + 0.5, 2.5) for b in np.arange(-lb, lb + 0.5, 2.5)], key=sole)
                    best = list(min(((a0 + a, b0 + b) for a in np.arange(-2.5, 2.6, 0.5) for b in np.arange(-2.5, 2.6, 0.5)), key=sole))
                sole(best); say(f"{names[j]} turned {np.round(best, 1).tolist()} deg ({axes_}) to plant the sole")
    plant(cfg.get("plant"))
    aim(cfg.get("aim_late"))
    for clip, f, name in cfg.get("clip_paw") or []:
        G, Q = clip_frames(rig, clip, d); G, Q = G[f], Q[f]
        fw = (Q[names.index(cfg["clip_paw_body"][1])] - Q[names.index(cfg["clip_paw_body"][0])]) * [1, 0, 1]
        yaw = Rotation.from_rotvec(-np.arctan2(fw[0], fw[2]) * np.array([0, 1.0, 0])).as_matrix(); j = names.index(name)
        for q in sub(j): W[q] = yaw @ G[q]
        if cfg.get("clip_paw_forward"):
            P_ = place(); tips = [q for q in sub(j) if not any(np.linalg.norm(d["bone"][c]) > 1e-9 for c in kids[q])]
            fw = (P_[tips].mean(0) - P_[j]) * [1, 0, 1]; Ry = Rotation.from_rotvec(np.arctan2(-fw[0], fw[2]) * np.array([0, 1.0, 0])).as_matrix()
            for q in sub(j): W[q] = Ry @ W[q]
    for name in cfg.get("mirror_legs") or []:
        name, _, stop = name.partition(">"); skip = set(sub(names.index(stop))) if stop else set()
        Wk = {q: W[twin[q]].copy() for q in skip if q in twin}
        for q in sub(names.index(name)):
            if q not in twin or q in skip: continue
            r = twin[q]; c = max(kids[q], key=lambda k: np.linalg.norm(d["bone"][k]), default=None)
            if c is None or c not in twin or np.linalg.norm(d["bone"][c]) <= 1e-9: continue
            P_ = place(); tgt = P_[c] * [-1, 1, 1] - P_[r]; v = W[r] @ d["bone"][twin[c]]
            R = Rotation.align_vectors([tgt / np.linalg.norm(tgt)], [v / np.linalg.norm(v)])[0].as_matrix()
            for k in sub(r): W[k] = R @ W[k]
            ax = tgt / np.linalg.norm(tgt); ml, mr = dom_ == q, dom_ == r
            if ml.sum() < 3 or mr.sum() < 3: continue
            target = cKDTree(skin_()[1][ml] * [-1, 1, 1]); W0 = {k: W[k].copy() for k in sub(r)}
            def gap(t):
                Rt = Rotation.from_rotvec(np.radians(t) * ax).as_matrix()
                for k in sub(r): W[k] = Rt @ W0[k]
                return target.query(skin_()[1][mr])[0].mean()
            ts = np.arange(-180, 180, 5.0); t0 = ts[np.argmin([gap(t) for t in ts])]; gap(min(t0 + np.arange(-5, 5.1, 0.5), key=gap))
        for q, Wq in Wk.items(): W[twin[q]] = Wq
    for name in cfg.get("mirror_feet") or []:
        l = names.index(name); r = twin[l]; ml, mr = np.isin(dom_, sub(l)), np.isin(dom_, sub(r))
        target = cKDTree(skin_()[1][ml] * [-1, 1, 1]); W0r = {q: W[q].copy() for q in sub(r)}; jl = [q for q in sub(l) if q in twin and twin[q] in sub(r)]
        def cost(rv):
            R = Rotation.from_rotvec(rv).as_matrix()
            for q in sub(r): W[q] = R @ W0r[q]
            P_, V_ = skin_(); return target.query(V_[mr])[0].mean() + np.linalg.norm(P_[[twin[q] for q in jl]] - P_[jl] * [-1, 1, 1], axis=1).mean()
        x0s = (np.zeros(3), np.array([np.pi / 4, 0, 0]), np.array([-np.pi / 4, 0, 0]), np.array([0, np.pi / 2, 0]), np.array([0, -np.pi / 2, 0]), np.array([0, np.pi, 0]))
        cost(min((minimize(cost, x0, method="Nelder-Mead", options={"xatol": 1e-4, "fatol": 1e-6}) for x0 in x0s), key=lambda o: o.fun).x); msk(name)
    def fit(entries, one=False):
        for upper, lower, axis_name, *rest in entries or []:   # bend th solved so this leg's paw meets the floor the others set
            u, lw = names.index(upper), names.index(lower)
            sign = rest[:1]; foot = names.index(rest[1]) if len(rest) > 1 else None; back = rest[2] if len(rest) > 2 else 2.0
            ax = np.eye(3)["xyz".index(axis_name)]; tw = None if one else twin.get(u)
            both = sub(u) + (sub(tw) if tw is not None else []); mb = np.isin(dom_, both)
            floor = skin_()[1][~mb, 1].min(); W0_ = {q: W[q].copy() for q in both}
            def low(th):
                R = Rotation.from_rotvec(np.radians(th) * ax).as_matrix()
                for a_, b_, Rr in [(u, lw, R)] + ([(tw, twin[lw], M @ R @ M)] if tw is not None else []):
                    for q in sub(a_): W[q] = Rr @ W0_[q]
                    for q in sub(b_): W[q] = (Rotation.from_matrix(Rr).inv() ** back).as_matrix() @ W[q] if foot is not None else Rr.T @ W[q]
                    if foot is not None:
                        for q in sub(foot if a_ == u else twin[foot]): W[q] = W0_[q]
                return skin_()[1][mb, 1].min() - floor
            if low(0.0) >= 0:   # short of the floor: unbend toward the side that lowers the paw
                sg = 1 if low(2.0) < low(-2.0) else -1; ths = np.arange(0, 85.5, 0.5) * sg; lows = np.array([low(t) for t in ths])
                if (lows < 0).any(): lo, hi = ths[np.argmax(lows < 0)], 0.0
                else: th = ths[np.argmin(lows)]; low(th); say(f"{upper} unbent {th:.1f} deg, still short of the floor"); continue
            else: lo, hi = 0.0, 85.0 * (sign[0] if sign else (1 if low(85.0) > low(-85.0) else -1))
            for _ in range(40):
                mid = 0.5 * (lo + hi)
                if low(mid) < 0: lo = mid
                else: hi = mid
            low(hi); say(f"{upper} bent {hi:.1f} deg to meet the floor")
    Sof = {}
    if cfg.get("mirror_rig"):
        for r, l, S in flip_table(d): Sof[l] = S; W[r] = M @ W[l] @ S
        P_ = place()   # limb roots whose vendor offset is not mirrored: aimed so their child joint mirrors too
        for r, l in d["pairs"]:
            r, l = int(r), int(l)
            if parents[l] in twin: continue
            ch = max((k for k in kids[l] if k in twin and np.linalg.norm(d["bone"][k]) > 1e-9), key=lambda k: len(sub(k)), default=None)
            if ch is None: continue
            tgt = P_[ch] * [-1, 1, 1] - P_[r]; v = W[r] @ d["bone"][twin[ch]]
            W[r] = Rotation.align_vectors([tgt / np.linalg.norm(tgt)], [v / np.linalg.norm(v)])[0].as_matrix() @ W[r]
    for entry in cfg.get("flat_toes") or []:   # every digit bone under this paw pitched level (or [paw, deg]: deg below level),
        name, slope = (entry, 0.0) if isinstance(entry, str) else entry   # keeping its heading: toes lie on their pads, not their claws
        for h0 in [names.index(name)] + ([twin[names.index(name)]] if names.index(name) in twin else []):
            for q in kids[h0]:   # each digit's base bone; the claw beyond keeps its own curl (levelled too, its tip would rise)
                c = max(kids[q], key=lambda c: np.linalg.norm(d["bone"][c]), default=None)
                if c is None or np.linalg.norm(d["bone"][c]) < 1e-9: continue
                v = W[q] @ d["bone"][c]; v /= np.linalg.norm(v); hz = v * [1, 0, 1]
                if np.linalg.norm(hz) < 1e-6: continue
                tgt = hz / np.linalg.norm(hz) * np.cos(np.radians(slope)) - np.array([0, np.sin(np.radians(slope)), 0])
                R = Rotation.align_vectors([tgt], [v])[0].as_matrix()
                for k in sub(q): W[k] = R @ W[k]
    for fp in cfg.get("fit_paw") or []:
        h0 = names.index(fp["paw"]); mp = np.isin(dom_, sub(h0)); V_ = skin_()[1]
        toes = [k for k in kids[h0] if np.isin(dom_, sub(k)).sum() >= 3 and V_[np.isin(dom_, sub(k)), 1].min() - V_[mp, 1].min() < 0.3 * np.ptp(V_[mp, 1])]
        def chain_(k):
            c = [k]
            while any(np.linalg.norm(d["bone"][x]) > 1e-9 and kids[x] for x in kids[c[-1]]):
                c.append(max((x for x in kids[c[-1]] if kids[x]), key=lambda x: np.linalg.norm(d["bone"][x])))
            return c
        segs = [s_ for k in toes for s_ in (chain_(k) if fp.get("all_phalanges") else [k] + [c for c in kids[k] if np.linalg.norm(d["bone"][c]) > 1e-9 and kids[c]])]
        tm = [np.isin(dom_, sub(k)) for k in toes] + ([dom_ == h0] if fp.get("palm") else [])
        if fp.get("all_phalanges"): tm += [dom_ == q for q in segs if (dom_ == q).sum() >= 3 and q not in toes]
        W0p = {q: W[q].copy() for q in sub(h0)}; reg = fp.get("reg", 0.0005)
        def setp(x):
            for q in sub(h0): W[q] = W0p[q]
            for i, k in enumerate(segs):   # + tips the digit segment down, about its horizontal across-axis
                c = max(kids[k], key=lambda c: np.linalg.norm(d["bone"][c])); fdir = (W[k] @ d["bone"][c]) * [1, 0, 1]
                R = Rotation.from_rotvec(np.radians(x[3 + i]) * np.cross([0, 1, 0], fdir / (np.linalg.norm(fdir) + 1e-9))).as_matrix()
                for q in sub(k): W[q] = R @ W[q]
            R = Rotation.from_rotvec(np.radians(x[:3])).as_matrix()
            for q in sub(h0): W[q] = R @ W[q]
        pq = fp.get("pad_q")   # pad_q: a digit rests along its length: the mean of its lowest pad_q fraction of vertices counts
        def pads_():   # each pad's 3rd lowest vertex (or lowest pad_q mean) above the paw's lowest point (a hanging claw must come up)
            V_ = skin_()[1]; y0 = V_[mp, 1].min()
            if pq: return np.array([np.sort(V_[m, 1])[: max(3, int(pq * m.sum()))].mean() for m in tm]) - y0
            return np.array([np.sort(V_[m, 1])[2] for m in tm]) - y0
        def cost(x): setp(x); return (pads_() ** 2).sum() + reg * (np.asarray(x) ** 2).sum()
        bounds = [(-15, 15)] * 3 + [(-fp.get("digit_deg", 25), fp.get("digit_deg", 25))] * len(segs)
        best = differential_evolution(cost, bounds, seed=0, tol=1e-6, maxiter=300, polish=True, x0=np.zeros(len(bounds))); setp(best.x)
        if fp.get("mirror"):
            for q in sub(h0): W[twin[q]] = M @ W[q] @ Sof[q if q in Sof else twin[q]]
        say(f"{names[h0]} turned {np.round(best.x[:3], 1).tolist()} deg, digits pitched {np.round(best.x[3:], 1).tolist()}; worst pad {pads_().max():.2f} above the lowest point")
    sym_skin(cfg.get("sym_skin_late"))   # a head re-rolled onto its own mirror after the level / rotate edits
    for name, wmid in cfg.get("sym_hips") or []:   # a midline joint's own turn (3 dof) solved so its twin children (hips) mirror, against
        from scipy.optimize import minimize   # wmid x how far its midline children leave the midline; nothing else turns
        j = names.index(name); W0j = W[j].copy(); tw = [(int(l), int(r)) for r, l in d["pairs"] if parents[int(l)] == j]
        mids = [k for k in kids[j] if k in centre and np.linalg.norm(d["bone"][k]) > 1e-9]; Wk = {q: W[q].copy() for k in kids[j] for q in sub(k)}
        def cst(rv):
            W[j] = Rotation.from_rotvec(rv).as_matrix() @ W0j
            for q, Wq in Wk.items(): W[q] = Wq
            P_ = place(); return sum(np.linalg.norm(P_[l] - P_[r] * [-1, 1, 1]) for l, r in tw) + wmid * sum(abs(P_[k, 0]) for k in mids)
        c0 = cst(np.zeros(3)); b_ = minimize(cst, np.zeros(3), method="Nelder-Mead", options={"xatol": 1e-5, "fatol": 1e-6}); cst(b_.x)
        say(f"{name} turned {np.degrees(np.linalg.norm(b_.x)):.1f} deg: hips mirror + midline cost {c0:.2f} -> {b_.fun:.2f}")
    if cfg.get("sym_chain"):   # a midline chain (root to tip) solved jointly: each joint yawed and rolled about its bone (what hangs below
        from scipy.optimize import minimize   # follows) until every segment's own skin mirrors itself, against 0.5 x how far its joints leave
        ch = [names.index(c) for c in cfg["sym_chain"]]; ms = [dom_ == j for j in ch]; W0c = W.copy()   # the midline (so only tips yaw)
        def turn(x):
            W[:] = W0c
            for i, j in enumerate(ch):
                b = W[j] @ d["bone"][ch[i + 1]] if i + 1 < len(ch) else W[parents[j]] @ d["bone"][j]
                R = Rotation.from_rotvec(np.radians(x[2 * i]) * np.array([0, 1.0, 0])).as_matrix() @ Rotation.from_rotvec(np.radians(x[2 * i + 1]) * b / np.linalg.norm(b)).as_matrix()
                for q in sub(j): W[q] = R @ W[q]
        def gaps(x):
            turn(x); P_, V_ = skin_(); return [cKDTree(V_[m] * [-1, 1, 1]).query(V_[m])[0].mean() for m in ms], sum(abs(P_[k, 0]) for j in ch for k in kids[j])
        def cost(x): g, off = gaps(x); return sum(g) + 0.5 * off
        x0 = np.zeros(2 * len(ch)); g0 = gaps(x0)[0]; best = minimize(cost, x0, method="Powell"); g1 = gaps(best.x)[0]
        say(f"{cfg['sym_chain'][0]}.. yaw/roll {np.round(best.x, 1).reshape(-1, 2).tolist()} deg, skin mirror gaps {np.round(g0, 2).tolist()} -> {np.round(g1, 2).tolist()}")
    for name in cfg.get("sym_roll") or []:   # a midline joint rolled about its own bone until its own skin mirrors itself; only its
        j = names.index(name); m = dom_ == j   # own turn changes (what hangs below keeps its world turn), joints stay on the midline
        c = max((k for k in kids[j] if k in centre and np.linalg.norm(d["bone"][k]) > 1e-9), key=lambda k: len(sub(k)), default=None)
        if m.sum() < 5: continue
        W0j = W[j].copy(); ax = W0j @ d["bone"][c] if c is not None else W[parents[j]] @ d["bone"][j]   # a chain tip: about the bone into it
        ax /= np.linalg.norm(ax)
        others = [k for k in kids[j] if k != c and np.linalg.norm(d["bone"][k]) > 1e-9]
        def gap(t_):
            W[j] = Rotation.from_rotvec(np.radians(t_) * ax).as_matrix() @ W0j; P_, V_ = skin_(); X = V_[m]
            return cKDTree(X * [-1, 1, 1]).query(X)[0].mean() + sum(abs(P_[k, 0]) for k in others if k in centre)
        g0 = gap(0.0); ts = np.arange(-30, 30.5, 0.5); t_ = ts[np.argmin([gap(x) for x in ts])]; gap(t_)
        say(f"{name} rolled {t_:.1f} deg, skin mirror gap {g0:.2f} -> {gap(t_):.2f}")
    fit(cfg.get("fit"))
    plant(cfg.get("plant_final"))
    fit(cfg.get("fit_one"), one=True)
    for name in cfg.get("ground") or []:   # a leg tilted about its root until its lowest skin point is on the floor (twin mirrored)
        P_, V_ = skin_(); floor = V_[:, 1].min(); j0 = names.index(name); tw = twin.get(j0)
        both = sub(j0) + (sub(tw) if tw is not None else []); m = np.isin(dom_, sub(j0)); mb = np.isin(dom_, both); W0_ = {q: W[q].copy() for q in both}
        h = (V_[m][V_[m, 1].argmin()] - P_[j0]) * [1, 0, 1]; axis = np.cross([0, 1, 0], h / np.linalg.norm(h))
        def drop(th):
            R = Rotation.from_rotvec(np.radians(th) * axis).as_matrix()
            for q in sub(j0): W[q] = R @ W0_[q]
            if tw is not None:
                for q in sub(tw): W[q] = M @ R @ M @ W0_[q]
            return skin_()[1][mb, 1].min() - floor
        lo, hi = -45.0, 45.0
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            if drop(mid) > 0: lo = mid
            else: hi = mid
        drop(hi)
    P, V = skin_(); lift = -V[:, 1].min(); P[:, 1] += lift; V[:, 1] += lift
    return d, W, P, V


def flip_table(d):
    """(right, left, S) per twin pair: the axis flip with W_r = M W_l S for a mirror image. The child offsets decide
    (S b_left_child = b_right_child); ties within 1% of their length go to the bind pose's flip."""
    kids = children(d["parents"]); twin = {}
    for r, l in d["pairs"]: twin[int(r)] = int(l); twin[int(l)] = int(r)
    flips = [np.diag(f) for f in ([-1, 1, 1], [1, -1, 1], [1, 1, -1], [-1, -1, -1])]; out = []
    for r, l in d["pairs"]:
        r, l = int(r), int(l); ch = [(k, twin[k]) for k in kids[l] if k in twin and np.linalg.norm(d["bone"][k]) > 1e-9]
        Sb = d["Wbind"][l].T @ M @ d["Wbind"][r]; L = sum(np.linalg.norm(d["bone"][k]) for k, _ in ch) + 1e-9
        out.append((r, l, min(flips, key=lambda S: (round(sum(np.linalg.norm(d["bone"][kr] - S @ d["bone"][k]) for k, kr in ch) / L, 2), np.linalg.norm(Sb - S)))))
    return out


def write(rig, d, W, P, out):
    """The pose as a one-frame BVH on the raw rig (its OFFSET header and channels), re-read and checked."""
    from poseydon.io.bvh import BVH
    from poseydon.core.animation import Animation
    from poseydon.core.skeleton import SkeletonManifest
    from poseydon.build.corpus import rest_source
    from poseydon.core.rotations import matrix_to_quat
    m = SkeletonManifest.load(f"data/truebones/rigs/{rig}/manifest.yaml")
    rest_bvh = BVH.read(rest_source(m, sorted((RAW / rig).glob("*.bvh"))))
    T = d["T"]; parents = d["parents"]; names = [str(x) for x in d["names"]]
    Wr = T.T @ W; local = Wr.copy(); local[1:] = np.swapaxes(Wr[parents[1:]], -1, -2) @ Wr[1:]
    tr = d["bone"][None].copy(); tr[0, 0] = T.T @ P[0]
    anim = Animation(rotations=matrix_to_quat(local)[None], translations=tr, offsets=d["offsets"], parents=parents, names=tuple(names), fps=float(d["fps"]))
    path = Path(out) / f"{rig}_tpose.bvh"; BVH.from_animation(anim, channels=rest_bvh.channels).write(path)
    pos = BVH.read(path).to_animation().global_positions()[0] @ T.T; rel = pos - pos[0]; ext = np.ptp(rel, axis=0).max()
    pairs, centre = d["pairs"], d["centre"]
    across = sum(rel[r] - rel[l] for r, l in pairs) if len(pairs) else np.array([-1.0, 0, 0]); fwd = np.cross([0, 1, 0], across); fwd /= np.linalg.norm(fwd) + 1e-12
    print(json.dumps({"rig": rig, "joint_mirror_err": round(max((np.linalg.norm(M @ rel[r] - rel[l]) for r, l in pairs), default=0.0) / ext, 6),
                      "midline_err": round(max((abs(rel[j, 0]) for j in centre), default=0.0) / ext, 6), "facing_dot_z": round(float(fwd[2]), 5)}), flush=True)
    np.save(Path(out) / f"{rig}_pos.npy", pos)


# ======================================================================================================== views
def sheet(rig, d, P, V, out):
    plt = _plt(); tris = np.load(PROB / f"{rig}_faces.npz")["tris"]; turn = Rotation.from_euler("xy", [20, -40], degrees=True).as_matrix()
    fig, axes = plt.subplots(1, 4, figsize=(26, 6.5), gridspec_kw={"width_ratios": [1, 2, 1, 1.6]})
    for ax, (title, h, v, z) in zip(axes, (("front", 0, 1, 2), ("side", 2, 1, 0), ("top", 0, 2, 1), ("3/4", 0, 1, 2))):
        Vv, Pv = (V @ turn.T, P @ turn.T) if title == "3/4" else (V, P)
        draw(ax, Vv, tris, Pv, d["parents"], h, v, z, floor=title != "3/4"); ax.set_title(f"{rig} {title}", fontsize=8)
    fig.tight_layout(); fig.savefig(Path(out) / f"sheet_{rig}.png", dpi=110); plt.close(fig)


def feet(rig, d, P, V, out, paws):
    """Each given left paw (a hand / foot, or a leg root: its lower limb) beside its twin, front and side, close up."""
    plt = _plt(); names = [str(x) for x in d["names"]]; kids = children(d["parents"]); tris = np.load(PROB / f"{rig}_faces.npz")["tris"]
    dom = d["inf_j"][np.arange(len(V)), d["inf_w"].argmax(1)]; twin = {}
    for r, l in d["pairs"]: twin[int(r)] = int(l); twin[int(l)] = int(r)
    def chain(j):
        c = [j]
        while kids[c[-1]]: c.append(max(kids[c[-1]], key=lambda k: np.linalg.norm(d["bone"][k])))
        return c
    fig, axes = plt.subplots(len(paws), 4, figsize=(16, 3.6 * len(paws)), squeeze=False)
    for row, name in zip(axes, paws):
        for side, j0 in (("L", names.index(name)), ("R", twin[names.index(name)])):
            lower = subtree(kids, j0) if len(kids[j0]) > 1 else subtree(kids, chain(j0)[-4])
            m = np.isin(dom, lower); sel = tris[m[tris].all(1)]; idx = np.unique(sel); remap = -np.ones(len(V), int); remap[idx] = np.arange(len(idx))
            for ax, (t, h, v, z) in zip(row[(0 if side == "L" else 2):], (("front", 0, 1, 2), ("side", 2, 1, 0))):
                draw(ax, V[idx], remap[sel], P, d["parents"], h, v, z); ax.set_title(f"{names[j0]} {t}", fontsize=8)
                ax.set_xlim(V[idx, h].min() - 3, V[idx, h].max() + 3); ax.set_ylim(-2, V[idx, 1].max() + 3)
    fig.tight_layout(); fig.savefig(Path(out) / "feet.png", dpi=90); plt.close(fig)


def paw(rig, d, P, V, out, name):
    """A paw and its twin from four angles (3/4 front-out, front low, side, top): finger placement and grounding."""
    plt = _plt(); names = [str(x) for x in d["names"]]; kids = children(d["parents"]); tris = np.load(PROB / f"{rig}_faces.npz")["tris"]
    dom = d["inf_j"][np.arange(len(V)), d["inf_w"].argmax(1)]; twin = {}
    for r, l in d["pairs"]: twin[int(r)] = int(l); twin[int(l)] = int(r)
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    for row, j0 in zip(axes, (names.index(name), twin[names.index(name)])):
        m = np.isin(dom, subtree(kids, j0)); sel = tris[m[tris].all(1)]; idx = np.unique(sel); remap = -np.ones(len(V), int); remap[idx] = np.arange(len(idx))
        for ax, (t, (ex, ey)) in zip(row, [("3/4 front-out", [10, -40]), ("front low", [-5, 0]), ("side", [0, 90]), ("top", [89, 0])]):
            R = Rotation.from_euler("xy", [ex, ey if j0 == names.index(name) else -ey], degrees=True).as_matrix(); Vv, Pv = V @ R.T, P @ R.T
            draw(ax, Vv[idx], remap[sel], Pv, d["parents"], 0, 1, 2, floor=ex == 0)
            ax.set_xlim(Vv[idx, 0].min() - 2, Vv[idx, 0].max() + 2); ax.set_ylim(Vv[idx, 1].min() - 2, Vv[idx, 1].max() + 2); ax.set_title(f"{names[j0]} {t}", fontsize=8)
    fig.tight_layout(); fig.savefig(Path(out) / f"paw_{name}.png", dpi=80); plt.close(fig)


def skelsym(rig, d, P, out):
    """Pixel-level skeleton symmetry: bones rasterised in the front (x, y) and top (x, z) views, diffed with the mirror
    image (1 px tolerance). 0 px in both views = a symmetric skeleton; the worst bones are printed."""
    from PIL import Image, ImageDraw
    plt = _plt(); names = [str(x) for x in d["names"]]; par = d["parents"]
    segs = [(par[j], j) for j in range(1, len(names)) if np.linalg.norm(P[j] - P[par[j]]) > 1e-6]; S = 600 / np.ptp(P, axis=0).max()
    def raster(ax2, mirror=False, only=None):
        im = Image.new("L", (700, 700)); dr = ImageDraw.Draw(im); c = np.array([350, 350 + S * P[:, ax2].mean()])
        for a, b in (segs if only is None else only):
            pa, pb = P[a][[0, ax2]] * [S, -S], P[b][[0, ax2]] * [S, -S]
            if mirror: pa, pb = pa * [-1, 1], pb * [-1, 1]
            dr.line([tuple(pa + c), tuple(pb + c)], fill=255, width=3)
        return np.array(im) > 0
    grow = lambda X: X | np.roll(X, 1, 0) | np.roll(X, -1, 0) | np.roll(X, 1, 1) | np.roll(X, -1, 1)
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    for ax, (k, title) in zip(axs, [(1, "front (x, y)"), (2, "top (x, z)")]):
        A, B = raster(k), raster(k, True); diff = (A & ~grow(B)) | (B & ~grow(A))
        ax.imshow(np.dstack([A & ~B, B & ~A, A & B]).astype(float)); ax.set_title(f"{title}: {diff.sum()} px differ"); ax.axis("off")
        worst = sorted((((raster(k, only=[s]) & ~grow(B)).sum(), names[s[1]]) for s in segs), reverse=True)[:6]
        print(title, int(diff.sum()), "px differ; worst bones:", [(int(w), nm) for w, nm in worst if w > 0], flush=True)
    fig.savefig(Path(out) / f"skelsym_{rig}.png", dpi=80, bbox_inches="tight"); plt.close(fig)


def belly(rig, d, P, V, out, ref):
    """Twist check: the mesh coloured back (red) / belly (blue) as in a reference pose ("bind" or "clip:frame", lying flat),
    seen from above, below and the side. A twisted joint shows as blue on top."""
    plt = _plt(); from matplotlib.collections import PolyCollection
    n = len(d["names"]); par = d["parents"]
    if ref == "bind":
        Wr = d["Wbind"]; Pr = np.zeros((n, 3))
        for j in range(1, n): Pr[j] = Pr[par[j]] + Wr[par[j]] @ d["bone"][j]
    else: f_, k_ = ref.split(":"); G, Q = clip_frames(rig, f_, d); Wr, Pr = G[int(k_)], Q[int(k_)]
    Vr = (d["inf_w"][..., None] * (np.einsum("vkab,vkb->vka", Wr[d["inf_j"]], d["inf_u"]) + Pr[d["inf_j"]])).sum(1)
    dom = d["inf_j"][np.arange(len(V)), d["inf_w"].argmax(1)]; up = (Vr - Pr[dom])[:, 1]; up = np.clip(up / (np.abs(up).max() * 0.5), -1, 1)
    tris = np.load(PROB / f"{rig}_faces.npz")["tris"]; fig, axes = plt.subplots(3, 1, figsize=(22, 9))
    for ax, (title, h, v, z, sg) in zip(axes, [("from above", 0, 2, 1, 1), ("from below", 0, 2, 1, -1), ("side", 2, 1, 0, 1)]):
        tt = V[tris]; order = np.argsort(sg * tt[:, :, z].mean(1)); cols = plt.cm.coolwarm(0.5 + 0.5 * up[tris].mean(1)[order])
        ax.add_collection(PolyCollection(tt[order][:, :, [v, h]] if v == 2 else tt[order][:, :, [h, v]], facecolors=cols, edgecolors="none"))
        ax.autoscale_view(); ax.set_aspect("equal"); ax.set_title(f"{rig} {title} (red: back, blue: belly, as in {ref})", fontsize=8)
    fig.tight_layout(); fig.savefig(Path(out) / f"belly_{rig}.png", dpi=80); plt.close(fig)


def contact(rig, d, V):
    """Per leg: its lowest skin point (% of the extent), left and right; all ~0 means every foot is on the floor."""
    names = [str(x) for x in d["names"]]; kids = children(d["parents"]); dom = d["inf_j"][np.arange(len(V)), d["inf_w"].argmax(1)]; ext = np.ptp(V, axis=0).max()
    twin = {}
    for r, l in d["pairs"]: twin[int(r)] = int(l); twin[int(l)] = int(r)
    for l in d["legs"]:
        lows = [V[np.isin(dom, subtree(kids, j)), 1].min() / ext * 100 for j in (int(l), twin.get(int(l))) if j is not None and np.isin(dom, subtree(kids, j)).any()]
        print(f"  {rig} {names[int(l)]}: lowest {', '.join(f'{x:.1f}%' for x in lows)}", flush=True)


# ========================================================================================================== cli
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0]); sp = ap.add_subparsers(dest="cmd", required=True)
    b = sp.add_parser("build", help="(fbxsdk) problem + faces into tmp/tposes4"); b.add_argument("rigs", nargs="+"); b.add_argument("--out", default=str(PROB))
    p = sp.add_parser("pose", help="pose from the config, write BVH + sheet + skelsym (+ feet)"); p.add_argument("out"); p.add_argument("rig")
    p.add_argument("--paws", nargs="*", default=[], help="left paws / leg roots for the feet close-up"); p.add_argument("--base", help="override: show a base pose only")
    r = sp.add_parser("ref", help="a clip frame (or a saved pose: __TPOSE_authored.bvh 0), rendered like a pose"); r.add_argument("out"); r.add_argument("rig"); r.add_argument("clip"); r.add_argument("frame", type=int)
    r.add_argument("--paws", nargs="*", default=[])
    bl = sp.add_parser("belly", help="twist check: back / belly colouring from a flat reference pose"); bl.add_argument("out"); bl.add_argument("rig")
    bl.add_argument("ref", nargs="?", default="bind", help="bind or clip:frame lying flat"); bl.add_argument("--base")
    w = sp.add_parser("paw", help="a paw and its twin from four angles"); w.add_argument("out"); w.add_argument("rig"); w.add_argument("paw"); w.add_argument("--base")
    f = sp.add_parser("flips", help="the left -> right axis flip chosen per twin pair, with the bind's"); f.add_argument("rig")
    t = sp.add_parser("tree", help="the joint tree with positions in the current pose"); t.add_argument("rig"); t.add_argument("--base")
    s = sp.add_parser("save", help="copy OUT/<Rig>_tpose.bvh next to the clips as __TPOSE_authored.bvh, checked"); s.add_argument("out"); s.add_argument("rig")
    a = ap.parse_args()
    if a.cmd == "build":
        for rig in a.rigs: build(rig, a.out)
        return
    if a.cmd == "flips":
        d = load(a.rig); names = [str(x) for x in d["names"]]
        for rr, l, S in flip_table(d): print(f"{names[l]:32s} S {np.diag(S).astype(int)}  bind {np.round(np.diag(d['Wbind'][l].T @ M @ d['Wbind'][rr]), 2)}")
        return
    if a.cmd == "save":
        from poseydon.io.bvh import BVH
        src = Path(a.out) / f"{a.rig}_tpose.bvh"; clip = next(p_ for p_ in sorted((RAW / a.rig).glob("__*.bvh")) if "TPOSE" not in p_.name)
        x, y = BVH.read(src).to_animation(), BVH.read(clip).to_animation()
        ch = lambda p_: re.findall(r"CHANNELS.*", p_.read_text())
        assert list(x.names) == list(y.names) and ch(src) == ch(clip), f"joints / channels differ from {clip.name}"
        dst = RAW / a.rig / "__TPOSE_authored.bvh"; shutil.copy(src, dst); print(f"saved {dst} ({len(x.names)} joints, channels as {clip.name})"); return
    out = Path(getattr(a, "out", "tmp/tpose_tree")); out.mkdir(parents=True, exist_ok=True)
    base = f"clip:{a.clip}:{a.frame}" if a.cmd == "ref" else getattr(a, "base", None)
    d, W, P, V = pose(a.rig, config(a.rig, base))
    if a.cmd == "tree":
        names = [str(x) for x in d["names"]]
        for j in range(len(names)):
            k, dep = j, 0
            while k > 0: k = d["parents"][k]; dep += 1
            print(f"{j:3d} {'. ' * dep}{names[j]:30s} {np.round(P[j], 1)}")
        return
    if a.cmd == "paw": paw(a.rig, d, P, V, out, a.paw); return
    if a.cmd == "belly": belly(a.rig, d, P, V, out, a.ref); return
    contact(a.rig, d, V); sheet(a.rig, d, P, V, out)
    paws = a.paws or [str(d["names"][int(l)]) for l in d["legs"]]
    if paws: feet(a.rig, d, P, V, out, paws)
    skelsym(a.rig, d, P, out)
    if a.cmd == "pose": write(a.rig, d, W, P, out)


if __name__ == "__main__":
    main()
