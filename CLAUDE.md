# CLAUDE.md — tcgDataSynth project state & working notes

> This file is the durable memory for this project. Everything outside
> `workspace/output/` is deleted when the Docker container exits, so decisions,
> findings, and progress live HERE (and in code), not in scratch or agent memory.
> Keep it terse and current. Update it at the end of every working session.

## What this is
Blender 5.0 synthetic-data generator for a trading-card **corner-detection** model
(YOLO-pose, 4 keypoints/card). Author works in a headless Docker container with **no
Blender**; all `bpy` code is delivered as numbered `tests/tXX_*.py` scripts the USER
runs in Blender 5.0 and reports back. Pure-Python modules are unit-tested in Docker.

## blender install location

- "C:\Program Files\Blender Foundation\Blender 5.0\blender.exe" use this when providing commands or in scripts as needed

## Locked decisions (from user, 2026-07-19)
- **Orchestration:** modal-timer operator (one scene per timer tick; Blender + assets
  stay warm; pause = finish current tick then stop). NOT subprocess-per-scene.
- **Slab label:** procedural placeholder (white/colored rect + dark text-like marks).
- **Keypoint order:** `TL, TR, BR, BL` (clockwise, card's own upright frame). Spec default;
  user didn't override. v-flag = 2 for all.
- **Card image source:** recursively search
  `C:\Code\React\CollectiblesApp\src\ai_dev\datasets\pokemon\data\images`
  (USER's machine — NOT visible from Docker). Any image found is selectable at random;
  **filename stem = card ID** used in labels. Path is configurable in `config.py`.
- **Color management:** AgX (Phase 0), locked.
- **Render engine:** switched to **CYCLES + GPU/CUDA** (2026-07-19). EEVEE-Next rendered
  clear sleeves SOLID (Principled transmission didn't refract even with use_raytracing).
  Project is transmission-heavy, user configured Cycles+CUDA → Cycles is now primary.
  `blender/render_setup.py` = single engine knob (`config.RENDER_ENGINE`); EEVEE kept
  selectable for the Phase 8 A/B comparison. Cycles: samples=128, denoise on, device=GPU.

## Open items (need user input, non-blocking for current work)
- Card-back image: **RESOLVED** — user added `back.png` to the images root. Excluded
  from face discovery; will be the back texture when backs face camera. Rule added:
  **a card whose back faces the camera is NOT labeled** (front-face-visible test).
- Per-card picture-region coords (§3.4): **RESOLVED** — user will not supply metadata;
  use defaults for all cards (override hook `picture_regions.json` still available).
- Per-card picture-region coords (§3.4): no metadata source given. Default region
  x∈[0.080,0.920], y∈[0.098,0.471] used for all cards; optional override via a central
  `picture_regions.json` ({card_id: [x0,y0,x1,y1]}) if the user drops one in. Treated as
  a per-card INPUT with a default, never hardcoded in Blender code.

## Dev environment (Docker) — gotchas
- Container `/tmp` is a **100M tmpfs**; pip build/cache overflow it → "No space left on
  device". Fix: `export TMPDIR=<scratchpad>/piptmp; export PIP_CACHE_DIR=$TMPDIR/cache`
  and use `pip install --break-system-packages --no-cache-dir`. Overlay `/` has ~950G.
- `python3 -m venv` fails (no ensurepip); using system Python + `--break-system-packages`.
- Installed for Docker tests: numpy 2.4.6, opencv 5.0.0 (headless), pillow 12.3.0.
- Pure-python unit tests: `python3 tests/unit/test_*.py` (also pytest-compatible).

## Phase status
- [x] Scaffolding: dir layout, README, config.py, this file, requirements-dev.txt.
- [x] Card-sourcing module `texturegen/cardsource.py` (recursive discovery, seed-driven
      selection, per-card region w/ optional `picture_regions.json` override).
      Unit-tested green: `tests/unit/test_cardsource.py` (6/6).
- [x] **Phase 0 — API ground truth.** DONE. Report at `out/phase0_api_report.txt`.
      Findings recorded below.
- [x] **Phase 1 — bare card + labels.** Labeling math VALIDATED by user (viz corners
      correct across head-on/30°/50°/wide/long; back_facing & half_out → 0 labels).
      Pure-python Docker-tested: `labeltools/{yolo_pose,visualize,frustum}.py`.
      Blender `tests/t01_bare_card_label.py`. **Bug fixed:** cam near-clip was 0.1m and
      clipped the close wide-lens card → set `clip_start=0.001`. User must re-run t01 once
      to confirm `wide_15mm` now shows the card (labeling already correct).
- [x] **Phase 2 — card factory + protection assets. DONE (per user: don't wait for
      review after slab connector changes; consolidate + go Phase 3).**
      Slab connector changes applied: unified full-thickness CONNECTOR material (slot 3,
      transmission 0.7 + roughness = slightly less see-through) for outer edge + label
      outline + separating line; recess = 4 straight bars with EMPTY corners (gap=4mm, no
      diagonals); back = frosted. Fixes the black thin-ridge issue (now full-depth solid).
      CONSOLIDATED: `blender/assets_build.py` builds all card-independent protection meshes
      (sleeve 1mm/2.5mm, toploader, semirigid, slab) into assets/protection_lib.blend;
      `blender/asset_lib.py` loads/shares them at gen time w/ in-process fallback.
      `build_sleeve_mesh(size)` extracted for reuse. (t05 not user-verified — user waived.)
- [~] **Phase 2 (historical detail) — card factory + protection assets:**
      - Docker: `rules/combinations.py` sampler + `validate_scene_config`; tests 20/20
        incl. 500-sample audit w/ full coverage.
      - Blender modules created & promoted from validated Phase 1:
        `blender/labeling.py` (label_card = projection + frustum.classify),
        `blender/card_factory.py` (build_card_mesh + build_card_unit: 0.45mm thick
        rounded box, slot0=front image, slot1=back.png, slot2=mid-grey edge
        (0.18,0.18,0.18); OBJECT-linked front slot enables shared-mesh instancing).
      - `tests/t02_card_unit.py` — **CONFIRMED by user** (console correct + visuals good:
        front image, back.png correct orientation, grey edge, sharing check all True).
      - Edge color decision: **mid-grey** (user, 2026-07-19).
      - **SLEEVES (t03) DELIVERED — awaiting user run.** `texturegen/plastic_warp.py`
        (Docker-tested 4/4) pre-bakes warp normal maps to `assets/plastic_warp_{0,1,2}.png`
        (Blender's Python has NO cv2, so warp maps MUST be pre-baked in Docker & loaded).
        `blender/protection.py` = build_sleeve (2 layers, 0.05mm off, curve-to-meet on
        L/R/bottom via smoothstep, open top, +1/+2.5mm margin) + clear/opaque plastic
        mats using Principled Transmission (needs scene.eevee.use_raytracing=True).
        `tests/t03_sleeve.py` renders clear/opaque × 1mm/2.5mm + prints an EEVEE/material
        transparency API PROBE (to nail refraction settings from ground truth).
      - t03 run1 (EEVEE): clear sleeve SOLID → pivoted to Cycles.
      - t03 run2 (Cycles): clear sleeves GOOD; opaque-back front too milky → FIX: front
        stays transmission=1.0, 'matte'=roughness bump only (front_finish param).
      - t03 run3 (Cycles): black corner + spots on every sleeve. Weld (remove_doubles)
        did NOT fix → not a crack. User: spots everywhere on sleeve_only.
        LIKELY CAUSE: refractive glass TIR at curved sealed seams. Diagnostics delivered:
        `t03d_sleeve_diagnose.py` (geometry-vs-view) and `t03e_material_isolate.py`
        (glass-vs-normalmap + candidate fix). **Candidate fix: model thin plastic as
        straight-through ALPHA transparency (Transmission=0, Alpha<1) + glossy + warp
        normal — no refraction, no TIR.**
      - t03e CONFIRMED: glass = black spots (TIR), normal map fine, ALPHA = clean.
        Residual "slightly darker" corner = geometry pinch where sealed edges converge.
        FIXES APPLIED: make_clear_plastic rewritten to alpha model (Transmission=0,
        Alpha~0.15, Coat gloss, warp normal) — this is now the standard for ALL plastic
        (toploader/slab/binder/display-case will reuse it).
      - **SLEEVES CONFIRMED GOOD (user).** Final: Principled TRANSMISSION glass (IOR 1.5)
        + warp normal; all-edges-sealed welded geometry, 0.5mm mesh, 0.12mm standoff.
        Lesson: transmission glass works in CYCLES (not EEVEE); flat/parallel glass is
        clean, tightly-curved converging seams cause TIR artifacts. (History below.)
      - t03 run5 (alpha): still slightly visible + residual poke. Changes:
        (1) CLARITY: make_clear_plastic rewritten to Transparent+Glossy mixed by FRESNEL
            (near-invisible head-on, reflective at grazing) — no alpha film. Glossy node
            has a Principled-mirror fallback if the node id differs in 5.0.
        (2) SIMPLER GEOM: sleeve now sealed on ALL 4 edges (open-top dropped per user);
            d_edge includes top.
        (3) POKE-THROUGH: standoff 0.05→0.12mm AND grid 1.5mm→0.5mm cells (coarse quad
            straddled the card edge and dipped below it). Sleeve mesh is card-independent
            → Phase 4 must SHARE one mesh per size (perf; ~23k verts/layer at 0.5mm).
        **Awaiting user re-run of t03.** If Glossy fallback triggers, console is silent;
        reflections still work via Principled mirror.
      - **t04 (semi-rigid + toploader) DELIVERED — awaiting user run.** `protection.py`:
        build_toploader (70x98, 1mm gap), build_semirigid (81x108 + 12mm back lip),
        _flat_pocket_mesh (flat parallel sheets + L/R/bottom rim, open top). Rigid clear
        glass. Card inside always sleeved + offset ±2mm/±2° (from combinations). New
        `blender/scene_common.py` consolidates reset/world/camera/lights/framing.
        FIXED bug: build_sleeve set matrix_parent_inverse=card.matrix_world.inverted()
        which pinned the sleeve at origin for OFFSET cards → removed it (sleeve now
        inherits card transform via default identity parent-inverse). t03 unaffected.
      - t04 run1: alright. User asks: (1) opaque spine needs real ~2mm WIDTH (not just
        1mm gap-thickness), (2) warp = fewer/larger deflections, (3) toploaders need
        micro-scratches+dust+wear, (4) add a side point light for reflections.
        DONE: (1) _flat_pocket_mesh now = full clear sheets + opaque U-spine of 2mm width
        (3 boxes via _add_box, adjacent no-overlap); (2) plastic_warp base_cell 160→320,
        octaves 2→1, regenerated assets; (3) new texturegen/surface_wear.py (scratch+dust
        grayscale, tests 2/2) → assets/toploader_wear_{0,1,2}.png → make_toploader_plastic
        modulates Roughness via MapRange from the wear map; (4) scene_common.add_lights
        gained a lateral side_light. **Awaiting user re-run of t04.**
      - t04 run2 changes: spine on TOPLOADER only (semi-rigid = clear rim), spine darker
        blue + ~35% transparent (alpha 0.65), toploader plastic slight random grey→blue
        tint, scratch presence randomized (wear_rough base→0.35), 3× scratch density,
        toploader 70→74mm wide, rigid rotation ±2°→±1° (HOLDER_MAX_ROT_DEG).
      - t04 run3: **per-instance texture variation** added (see Texture variation section)
        — random base map (6 each) + random UV crop/zoom/flip so reflections/scratches
        don't repeat across the dataset. **Awaiting user re-run.**
      - t04 run4: UV variation seam FIXED (scale was >1 = tiling; now crops a <1 sub-window
        kept within [0,1] + EXTEND). User confirmed holders GOOD.
      - **SLAB (t05) DELIVERED — awaiting user run.** `texturegen/slab_label.py` (procedural
        grading label, tests 2/2) → assets/slab_label_{0..5}.png. `protection.build_slab`
        (80x135x6.7mm; label 20x68@4mm-from-top; line 4mm below; recess 64.2x89.70 outline;
        card in recess NO sleeve). Reuses toploader clear surface (slot0) + spine material
        for edges & ridge outlines (slot1) + label material (slot2), per user. `tests/t05_slab.py`.
      - t05 run2 changes (user): edges+ridges now SAME clear slab material (physical
        embossed, not spine); recess outline NOTCHED (4mm off each corner, octagon via
        _add_bar/_octagon_segments); back sheet = make_frosted_back (rough 0.85, transmission
        0.5, strong bump warp) to diffuse light; front unchanged. Slots: 0 clear(front+
        edges+ridges), 1 frosted back, 2 label. **Awaiting user re-run.**
      - **PENDING SLAB CHANGES (user, scheduled ~1h45m delay before doing them; then DO
        NOT wait for review — consolidate + go to Phase 3):**
        1. Recess rectangle NOT continuous: remove the 45° diagonal corner bars (the
           `_octagon_segments` diagonals) — corners must be EMPTY. So recess outline =
           only the 4 straight edge bars, stopping short so corners are open gaps.
        2. Internal label-outline rectangle + separating line render BLACK (thin raised
           clear transmission ridges = dark refraction). FIX: make them PHYSICAL geometry
           spanning FULL thickness (z from -hz to +hz, back→front), and use a material
           SLIGHTLY LESS see-through than the front (e.g. transmission ~0.7 / a bit of
           roughness) so they read as solid frosted connectors, not black.
        3. Outer edge of the slab: use the SAME connecting bits/material as the interior
           connectors (full-thickness, slightly-less-see-through), i.e. one unified
           "connector" material+geometry for outer edge + label outline + line + recess.
        Plan: add make_connector_material (transmission ~0.7, slight roughness, maybe faint
        tint); connectors = full-depth bars/walls (z -hz..+hz). Front=clear, back=frosted,
        label=label, connectors=new slot. Recess = 4 straight full-depth bars, corners open.
      - THEN: bundle all protection into assets_build.py → Phase 2 done → Phase 3.
- [~] **Phase 3 — finishes & damage (texturegen).** Docker pipeline DONE & self-verified:
      - `texturegen/holo.py`: region_mask (entire/picture/reverse from per-card picture
        coords) + patterns (none/cosmos/horizontal_lines) + masked_pattern. Tests 4/4.
      - `texturegen/damage.py`: dirt, scratches, surface_damage (edge/corner-biased white
        blobs) + composite_overlays. Tests 4/4.
      - `texturegen/physical_texture.py`: structure-tensor flow_field → LIC etched lines →
        height_to_normal. Tests 3/3. Preview out/unit_physical_lines.png confirms lines
        follow art contours + fill flat areas parallel (spec §3.4). 
      - numpy-2 gotcha: use np.ptp(a) not a.ptp() (method removed in numpy 2.x).
      - Previews for user review: out/unit_holo_*, unit_damage_*, unit_physical_*.
      - User tuning applied: physical texture less noisy + evenly spaced (LIC length 30,
        band-limit noise sigma 1.5); flow_field only follows HIGH-CONTRAST edges
        (contrast_percentile=80, weak gradients zeroed, orientation via tensor smoothing).
      - surface_damage v2: edge blobs now placed ON the nearest edge (touching, not just
        biased near it); added triangular RIPS anchored to an edge (n_rips); 2/3 blobs
        spread. IMPORTANT: damage (dirt/scratches/surface) MUST use a fresh PER-INSTANCE
        seed at gen time (NOT cached per card ID) to avoid overfitting — noted in
        damage.py docstring; Phase 4 gen loop must honor this.
      - **HOLO SHADER (t06) DELIVERED — awaiting user run.** `blender/finishes.py`
        make_front_material(finish, region_mode, pattern, ...): holo via Principled
        Thin Film Thickness[nm]/IOR, spatially modulated by region_mask*pattern; region
        mask = procedural UV-box math (picture coords, top-down v=1-uv.y); patterns =
        procedural (sine lines / noise cosmos / uniform); optional physical normal input.
        NO cv2 needed at render time. `tests/t06_finishes.py` renders 8 variants
        (normal + holo entire/picture/reverse x none/cosmos/lines) at a tilt w/ side light.
      - ARCHITECTURE NOTE (cv2 availability): holo region/pattern/iridescence = procedural
        (no cv2). DAMAGE overlays + physical-texture NORMAL still need cv2+numpy. Plan:
        install opencv into Blender's bundled Python so texturegen runs at gen time (damage
        per-instance, physical normal per card), OR pre-bake a damage POOL (card-independent)
        + a per-card precompute for physical normals. Decide with user before Phase 4.
      - **cv2 DECISION: user will install opencv-python-headless into Blender's bundled
        Python** (`...\Blender 5.0\5.0\python\bin\python.exe -m pip install
        opencv-python-headless`, admin). So texturegen runs at GEN TIME: damage per-instance,
        physical-texture normal per card. No pre-bake/precompute path needed.
      - t06 lighting reworked (user: overexposed): scene_common.add_lights is now §3.6-style
        — sun 90% dimmer (energy 0.2), cold SPOT beside camera (3.0), 2 random point lights
        (warm↔cold, 1.0-2.5W), rng param. t06 varies card angle per case (±18° X/Y).
      - **HOLO v2 (angle-dependent COLOR) DELIVERED — awaiting user run.** User: holo needs
        goniochromism, not just reflection intensity. Two versions in finishes.py to compare:
        make_holo_thinfilm (metallic + Thin Film Thickness pattern-driven 100-1500nm, IOR
        1.45) and make_holo_spectral (dot(perturbedNormal, Incoming)→spectral ColorRamp,
        angle perturbed per-pixel by pattern, FRACT-cycled; ramp→BaseColor+faint Emission).
        BOTH: pattern normal map perturbs shading normal (bends rainbow); lines get
        Anisotropic + Anisotropic Rotation (=0.0 for horizontal; may need tuning). Region
        mask mixes foil vs plain card. `tests/t06_finishes.py` = 2 versions × 4 patterns
        (none/cosmos/lines/water_web, region=entire) × 3 CAMERA azimuths (0/25/45, same card,
        fixed lights) = 24 renders → compare hue-shift-with-angle. REQUIRES cv2 in Blender.
      - New patterns (texturegen/holo.py, tests 5/5): water_web (wavy level-set web), cosmos
        redone as varied-size circles + smaller PIXELATED circles; pattern_normal() derives
        normal from pattern. water_web added to combinations.HOLO_PATTERNS too.
      - HOLO DECISION (user): thin-film reflects only WHITE (metallic reflects light color,
        not per-angle spectrum) → DROPPED. Spectral rainbow look = GOOD but was OPAQUE
        (rainbow was in metallic Base Color, hid the art). FIX: make_holo_spectral now keeps
        CARD ART as diffuse base (visible/translucent) + rainbow as EMISSIVE overlay (ramp)
        masked to region, emission=mask*0.7; pattern normal still perturbs shading normal +
        hue. t06 now spectral-only (4 patterns × 3 az = 12 renders). **Awaiting user run.**
      - Spectral v2 fixes (user): (1) WASHOUT — rainbow was additive emission over whole
        region → moved rainbow into BASE COLOR as a blend with art (mask*rainbow_blend=0.5,
        energy-neutral, no brightening) + faint emission 0.12. (2) FLAT holo — pattern
        normal was embossing cosmos/water_web; now the pattern perturbs the shading normal
        ONLY for horizontal_lines ("physical line layer"); cosmos/water_web/none are flat
        (color only via the pattern-value hue perturbation). **Awaiting user run.**
      - Spectral v3 = DIFFRACTION-FLASH model (user insight: holo is angle-SELECTIVE, not a
        uniform glow — only aligned regions flash, they shift with angle, between = slight
        darken). finishes.make_holo_spectral: phase P = dot(N,Incoming)*ANGLE_GAIN(12) +
        pattern*PATTERN_GAIN(4); flash f = pow(0.5+0.5cos(P*2π), SHARPNESS(4)); hue=FRACT(P)
        →ramp; base=art darkened by mask*DARKEN(0.18)*(1-f); emission=ramp*mask*f*EMIT(1.6).
        Localised flashes → no washout; shifts with camera/card angle. LIMITATION: surface
        shader can't read light dirs → responds to view/card angle only (offer specular
        variant for light-angle later). Tuning consts at top of finishes.py. **Awaiting run.**
      - Holo flash magic numbers moved to hand-editable `holo_tuning.json` (project root);
        `config.load_holo_tuning()` merges over DEFAULT_HOLO_TUNING (bad/missing → defaults);
        finishes.make_holo_spectral loads it at build time (edit JSON, re-run, no code change).
        Tested test_config.py 3/3.
      - PHYSICAL TEXTURE + DAMAGE WIRED (t07, awaiting user run). `texturegen/cardprep.py`
        (bpy-free, cv2, tests 3/3): physical_normal_path() = §3.4 etched normal from card art
        (cached per card ID); damaged_card_path() = dirt/scratches/surface composited (per
        INSTANCE via seed, cached per (card,seed)). finishes.make_holo_spectral gained
        physical_normal_path (etched foil = main RAISED layer; supersedes line-pattern normal;
        flat patterns stay flat). make_normal_material already applies physical normal.
        `tests/t07_finish_damage.py`: normal_plain/physical/damage + holo_cosmos_phys_damage +
        holo_lines. Requires cv2 in Blender.
      NOTE: t07 overrides the front material manually; PHASE 4 scene builder will orchestrate
        finish+damage+physical per SceneConfig (finish/damage come from rules/combinations).
      **Phase 3 essentially COMPLETE** pending t07 visual OK. 67/67 unit tests.
      - Tuning round (user): (1) dirt now has a LARGE-scale exclusion mask (whole regions
        clean, not uniform). (2) physical texture = clean evenly-spaced streamlines
        (traced_lines, Jobard-Lefebvre style: 1px AA lines, spacing=3 → 1px line + 2px gap,
        following high-contrast contours) — replaced the noisy LIC. (3) NEW RULE:
        physical_texture ALWAYS pairs with holo_pattern 'none' (combinations.py _sample_finish
        + validate + test). t07 holo case updated to none+physical accordingly.
- [~] **Phase 4 — layouts (table→floating→binder→display case→hand). STARTED.**
      Integration core `blender/scene_builder.py`: build_card_instance(CardConfig) assembles
      full card = damaged front + physical normal (cardprep) + finish (finishes normal/holo) +
      base unit + protection (sleeve/toploader/semirigid/slab, nested with inner offset). Returns
      CardInstance(root=what layout positions, card=what gets labeled). cv2-in-Blender required
      (falls back to plain texture). `blender/layouts.py` build_table: bg plane (noisy mat) +
      clutter rects + cards laid flat. `tests/t08_table.py`: samples table SceneConfig, builds,
      labels all, LAST card shoved half-out-of-frame for frustum-rule test. **Awaiting user run.**
      - t08 run1: labeling/frustum/backface CORRECT. Fix (user): slab clipped neighbors
        (grid was card-sized). Added scene_builder.protection_footprint() +
        protection_half_thickness(); build_table now spaces by max footprint DIAGONAL + gap
        (protection-aware, ~non-overlapping, small jitter) and lifts each object by half its
        thickness so a 6.7mm slab rests ON the table. Camera subject_h 0.30→0.40.
      - t08 run2 GOOD. Added config.json scene params (max_cards, allow_overlap,
        out_of_frustum keep/remove); t08 reads them.
      - HOLO TAG added to labels (|<id>|<tag>). t08 now builds a DELIBERATE demo scene:
        4 in-frustum cards one of each tag (none/full/holo/reverse) + 1 shoved out.
        **Re-run t08 + visualizer to confirm tags in the label file.**
      - FLOATING (t09) DELIVERED — awaiting user run. layouts.build_floating: bg plane far
        back + scattered prism(cube)/cylinder props (bpy.ops primitives) behind cards; cards
        float at varied depth (z -0.10..0.04) + tilt (rx,ry ±0.8, rz full), back_to_camera
        flips. tests/t09_floating.py samples floating config, labels all (holo tags),
        respects config scene params. Camera front-ish.
      - t09 run1: layout fine. Changes (user): config now PER-LAYOUT (floating gets
        max_cards/max_shapes/allow_overlap/out_of_frustum). PURPLE holo investigation: t09
        rebuilt as ALL-holo cards each in sleeve OR toploader, VARIED region/pattern, at
        floating angles. Hypothesis: t08's uniform purple = 'none' pattern → uniform phase →
        one hue whole card (not plastic). t09 uses cosmos/lines/water_web → should show varied
        flashes; if still uniform-tinted only on TOPLOADER cards → the toploader tint
        (0.9,0.92,0.95 in scene_builder._add_protection) is the culprit.
      - t09 run2 diagnosis (user): sleeve/toploader REFRACTIVE glass (IOR 1.5) TIR-traps the
        holo card's EMISSION → amplified saturation + whole-card reflect at angles (only over
        self-lit holo; fine over lit-diffuse cards = why unsleeved t06 was good). FIX:
        make_clear_plastic + make_toploader_plastic rewritten NON-REFRACTIVE = Transparent
        BSDF (straight-through) mixed by Fresnel with Glossy reflection (_clear_plastic_graph
        + _reflect_node w/ Glossy→Principled-mirror fallback); wear still drives glossy
        roughness. scene_builder holder tint softened to (0.97,0.98,1.0). CHANGES SLEEVE/
        TOPLOADER EVERYWHERE (re-verify t03/t04 normal-card look). **Awaiting user run.**
      - THIN-WALLED PLASTIC FIX (user's precise diagnosis): single-sided transmission
        permanently refracts camera ray → corrupts holo dot(N,Incoming) + Cycles kills the
        caustic → dark. FIX: make_clear_plastic + make_toploader_plastic = thin-walled
        (Fresnel(IOR1.5,warp) mixing Transparent + Glossy[warp only], + LightPath IsShadowRay
        → pure Transparent so lamp light passes). Slab keeps real transmission (make_slab_surface,
        reduced warp 0.08 + shadow branch). render_setup transparent_max_bounces=32. t10
        verify (bare/sleeve/toploader/both) CONFIRMED GOOD by user.
      - Holo pattern tweaks (user): cosmos base flat black (circles only); water_web thicker/
        less-intense(peak0.65)/smoother(blur2.2); horizontal_lines amp 0.5→0.35 (~30% less);
        pattern_normal strength 2.0→1.2 + pre-blur (smoother). Patterns gen at runtime (no assets).
      - BINDER (t11) DELIVERED — awaiting user run. layouts.build_binder: grid (rows x cols
        from "RxC"), content footprint per type (_BINDER_CONTENT sleeved/toploader/slab),
        slot_gap + padding, filled_slots (empty slots), clear thin-walled front pocket sheet
        + clear/solid back, hard-cover board (_solid_material) + 30mm spine, one/two offset
        pages. Cards = build_card_instance placed in grid, face +Z, labeled through front.
        Returns (instances, extent) for framing. binder config block added (max_cards 12).
        tests/t11_binder.py. v1 — expect board/spine/scale tuning.
      - t11 v2 (user): binder pages warped/loose+reflective → make_clear_plastic gained
        warp_strength (pages use 0.5 vs sleeve 0.18). Mirror-reflection test:
        layouts.scatter_reflectors (colorful prisms/cylinders) + non-sun point lights placed
        BEHIND the camera (setup_reflection_lighting) so they show only as reflections. t11
        now renders 4 scenes covering ALL grids(1x1/2x2/3x3/4x3) × page(clear/solid) ×
        content(slab/toploader/sleeved).
      - t11 v3 (user): (1) WELDED slot dividers — thin frosted bars (_weld_material) at slot
        midpoints/edges (vertical cols+1, horizontal rows+1) making separate slots. (2) Spine
        now CENTERED with TWO cover halves (_empty pivots at spine edges); content page on the
        configured side. (3) Halves TILT INWARD up to 10° about the spine (pivot rot_y=-sign*tilt;
        all half objects parented to pivot in local coords). (4) Reused table backdrop
        (_plane+_noisy_material behind, no clutter). build_binder now returns (instances,extent).
        **Awaiting user run.**
      NEXT layouts: display case, hand. Then Phase 5 (lighting/camera random),
      6 (postfx), 7 (GUI), 8 (throughput).
- [ ] Phase 5 — lighting & camera randomization
- [ ] Phase 6 — post effects (postfx)
- [ ] Phase 7 — GUI + orchestration
- [ ] Phase 8 — throughput + 50-image pilot

## Phase 0 findings (from out/phase0_api_report.txt, 2026-07-19)
- bpy.app.version = (5,0,0). Bundled Python 3.11.13, **numpy 1.26.4** (Docker has 2.4.6 —
  keep pure-python numpy-1.26-compatible; avoid numpy-2-only APIs).
- **Render engines: only `BLENDER_EEVEE`** registered (EEVEE Next). `scene.cycles` exists
  but CYCLES not in engine enum → Phase 8 Cycles compare needs the addon enabled.
- Principled BSDF socket names (Blender 5.0): `Base Color`, `Metallic`, `Roughness`, `IOR`,
  `Alpha`, `Normal`, `Specular IOR Level`, `Specular Tint`, `Anisotropic`,
  `Anisotropic Rotation`, `Tangent`, `Transmission Weight`, `Coat Weight/Roughness/IOR/
  Tint/Normal`, `Sheen Weight/Roughness/Tint`, `Emission Color`, `Emission Strength`,
  **`Thin Film Thickness`**, **`Thin Film IOR`** (=holo/iridescence). Subsurface split into
  `Subsurface Weight/Radius/Scale/IOR/Anisotropy`. distribution ∈ {GGX, MULTI_GGX}.
- Camera: `cam.lens`(mm), `sensor_width=36`, `sensor_height=24`, `sensor_fit`. DoF:
  `cam.dof.use_dof`, `cam.dof.focus_distance`(m), `cam.dof.aperture_fstop`, `focus_object`.
- `bpy_extras.object_utils.world_to_camera_view(scene, obj, coord)` AVAILABLE. Returns
  (0,0)=bottom-left → (1,1)=top-right, **z<0 = behind camera**. → YOLO: `y=1-ndc.y`,
  frustum = `0≤x≤1 & 0≤y≤1 & z>0`.
- EEVEE: final samples = `scene.eevee.taa_render_samples` (default 64). `use_raytracing`
  default False (enable for holo/reflections later). Legacy bloom/SSR toggles GONE (EEVEE
  Next) — bloom must be compositor/postfx.
- Color mgmt: default **view_transform = 'AgX'** (LOCKED). Static enum introspection only
  showed ['NONE'] because it's a dynamic OCIO enum; AgX/Standard/Filmic exist at runtime.
- Default render was 1920x1080 → we force 1280x1280.

## Central config — config.json (single source of tunable params)
`config.json` (project root) holds ALL runtime-tunable params; `config.load_config()`
deep-merges over `DEFAULT_CONFIG` with type coercion + fallback (bad edit never crashes).
Sections: `holo` (angle_gain/pattern_gain/sharpness/emit/darken — finishes.make_holo_spectral
via `load_holo_tuning()`) and `scene` (max_cards, allow_overlap, out_of_frustum
'keep'|'remove' — via `load_scene_params()`). Renamed from holo_tuning.json (2026-07-20).
Now PER-LAYOUT under `layouts` (table, floating, ...). `load_layout_params(name)` returns
that layout's params (recursive deep-merge, per-layout enum validation). floating adds
`max_shapes`. Wired: sample_scene_config(max_cards=), build_table/build_floating(allow_overlap,
max_shapes), t08/t09 read load_layout_params + honor out_of_frustum. ADD FUTURE TUNABLES HERE
(new layout = new entry under "layouts").

## Texture variation (IMPORTANT for dataset diversity)
Pre-baked warp/wear maps are FEW shared datablocks. To avoid identical
reflection/scratch patterns across the dataset, every plastic instance samples a
random cropped/zoomed/flipped sub-region via a Mapping node
(`protection.random_uv_xform(rng)` -> `_apply_uv_mapping`), AND picks a random base
map (6 each: assets/plastic_warp_{0..5}, toploader_wear_{0..5}). Mapping applies
result=win*uv+loc, so win<1 (0.3-0.6) zooms IN to a sub-window; the window is kept
fully within [0,1] (incl. flips) + texture EXTEND so it NEVER crosses a tile
boundary (fixes a hard reflection seam). Offset/scale/flip only (no rotation) so
normal-map vectors stay valid. Layout builders (Phase 4+)
MUST pass a per-instance uv_xform from the scene rng to build_sleeve/build_toploader/
build_semirigid, else patterns repeat. Images still load once (check_existing=True)
so memory stays flat regardless of instance count.

## Conventions (do not drift)
- Meters, real-world scale. Card = 0.063 × 0.088 × 0.00045 m. Corner radius 3mm.
- Single seeded `numpy.random.Generator` threaded everywhere; no bare `random.*`.
- Every image reproducible from one int seed (seed in filename/manifest).
- Labels: YOLO-pose, class 0, `class cx cy w h (xi yi vi)*4 |<card_id>|<holo_tag>`; Y
  flipped to top-left origin; points with camera-space z<0 (behind camera) = OUTSIDE
  frustum. holo_tag: none|full(entire)|holo(picture)|reverse — from finish region via
  scene_builder.holo_tag_for_finish; threaded label_card→frustum.classify→CardLabel.
  Standard (no-suffix) variant omits both bars.
- Docker-side pure-python tests live in `tests/unit/`; Blender scripts in `tests/tXX_*.py`.
