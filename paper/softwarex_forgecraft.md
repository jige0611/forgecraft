# ForgeCraft: An open-source framework for manufacturing-aware co-evolution of robot morphology and control

**作者**：【待填：姓名】<sup>a, *</sup>

<sup>a</sup> 【待填：单位 / 地址，例如 Department of ..., University of ..., City, Country】

<sup>*</sup> 通讯作者。E-mail: 【待填：通讯邮箱】

---

## Abstract

ForgeCraft is an open-source Python framework for the co-evolution of robot morphology and control, with an explicit emphasis on the transition from simulation to manufacturing. Most morphology-evolution research terminates at simulated locomotion: the resulting designs are evaluated in physics engines but are rarely manufacturable, because part geometry, tolerance stack-ups, standard fasteners and assembly constraints are absent from the search space. ForgeCraft couples four components into a single pipeline. A graph-based morphology representation builds bodies from a configurable parts catalogue. A co-evolution engine combines genetic operators, quality-diversity search (MAP-Elites) and multi-objective selection with a graph-neural-network morphology encoder and PPO control training. A physics evaluation layer supports MuJoCo and, where available, the Genesis backend. Finally, a manufacturing-aware export chain converts evolved bodies into parametric mechanical parts and produces STEP AP242 assemblies (tessellated geometry rather than analytic B-rep), 2D DXF drawings with tolerance annotation, interference reports, bills of materials and one-click production packages. Manufacturability is treated as a first-class objective rather than a post-hoc filter. The framework ships with 21 parts catalogues, 8 task definitions, 9 parametric part generators, 10 PBR material presets, and 424 automated tests executed in continuous integration on Python 3.10–3.12. We describe the architecture, illustrate the complete workflow on a recorded design, and outline the research applications the framework is intended to enable. Section 2.5 states the current limitations of the release explicitly.

**Keywords:** robot morphology; co-evolution; reinforcement learning; evolutionary computation; design for additive manufacturing; open-source software

---

## Required metadata

### Table 1 — Code metadata

| Nr. | Code metadata description | Please fill in this column |
|-----|---------------------------|----------------------------|
| C1 | Current code version | v0.4.0 |
| C2 | Permanent link to code / repository used for this code version | https://github.com/jige0611/forgecraft/releases/tag/v0.4.0 |
| C3 | Permanent link to Reproducible Capsule | None |
| C4 | Legal Code License | MIT License |
| C5 | Code versioning system used | git |
| C6 | Software code languages, tools, and services used | Python (≥3.10); PyTorch; NumPy; NetworkX; PyYAML; Pydantic; MuJoCo; Gymnasium; Trimesh; SciPy; FastAPI |
| C7 | Compilation requirements, operating environments & dependencies | OS-independent (Windows, Linux, macOS). No compilation required. Install with `pip install -e ".[all]"`; core install with `pip install -e .`. CPU execution supported; optional CUDA acceleration for policy training. |
| C8 | If available, link to developer documentation / manual | https://github.com/jige0611/forgecraft/tree/main/docs |
| C9 | Support email for questions | 【待填：你的邮箱】 |

### Table 2 — Software metadata

| Nr. | (Executable) software metadata description | Please fill in this column |
|-----|--------------------------------------------|----------------------------|
| S1 | Current software version | v0.4.0 |
| S2 | Permanent link to executables of this version | https://github.com/jige0611/forgecraft/releases/tag/v0.4.0 |
| S3 | Legal Software License | MIT License |
| S4 | Computing platforms / Operating Systems | Windows, Linux, macOS |
| S5 | Installation requirements & dependencies | Python ≥3.10; see C6. No system-level dependencies beyond a Python environment. |
| S6 | If available, link to user manual | https://github.com/jige0611/forgecraft#readme |
| S7 | Support email for questions | 【待填：你的邮箱】 |

---

## 1. Motivation and significance

The morphology of a robot largely determines what it can do, yet in practice morphology is usually fixed by human designers before control policies are learned. Co-design — the joint optimisation of body and controller — has therefore attracted sustained interest since the early work on evolving virtual creatures [1], and has been revisited with modern reinforcement learning in a number of influential systems, including graph-grammar-based design [2], unsupervised evolution of universal creatures [3], and transformer-based controllers for co-optimised agents [4]. Concurrent work has explored self-assembling morphologies [5], collection-level quality-diversity [6], and differentiable simulation for design [7].

A persistent limitation runs through this literature. Evaluated designs are almost always *simulated* bodies: abstract link-and-joint assemblies whose geometric realisation is left to the reader. In practice, an evolved morphology cannot be built without answering questions the simulation never asks. Which of the abstract links correspond to printable parts? Do the parts fit a build volume? Are walls thick enough? Do adjacent parts collide once real bearings, fasteners and clearances are introduced? Can a joint be assembled at all? What is the bill of materials?

These are not cosmetic concerns. They are precisely the constraints that determine whether an evolved design leaves the computer. When manufacturability is absent from the objective, evolutionary search is free to exploit shapes that are optimal in simulation and unbuildable in reality, and post-hoc filtering then discards the very designs the search converged on.

Two further gaps compound the problem. First, **reproducibility**: morphology-evolution results are frequently reported for bespoke, unavailable codebases, so comparisons across papers are difficult. Second, **fragmentation**: the required capabilities — morphology generation, physics evaluation, control learning, geometric modelling and manufacturing preparation — are typically distributed across separate tools with incompatible data models, forcing manual re-engineering between stages.

**Target audience.** ForgeCraft is intended for three groups. (i) Researchers in evolutionary robotics and morphology optimisation who need a reproducible, extensible testbed with manufacturing realism in the loop. (ii) Roboticists and mechanical designers who want to generate candidate mechanisms for additive manufacturing from a task-level specification. (iii) Educators who need a self-contained platform for teaching co-design, since the complete pipeline — from parts catalogue to machine-readable drawings — runs on a laptop.

**Relation to existing work.** ForgeCraft does not compete with morphology-evolution algorithms; it is complementary. Existing frameworks and benchmarks provide morphology encodings, physics engines or robot-learning environments, and ForgeCraft integrates these ecosystems rather than reimplementing them (Section 2). Its distinctive contribution is the *manufacturing-aware* portion of the pipeline: the explicit representation of evolved bodies as parametric mechanical parts, the treatment of manufacturability as an optimisation objective, and the automated production of manufacturing deliverables — STEP assemblies (tessellated), annotated 2D drawings, interference reports and bills of materials. To our knowledge, publicly available morphology co-evolution frameworks do not implement this export stage; we should be clear, however, that implementing the stage is not the same as validating it, and no evolved design from this framework has yet been fabricated and tested (Sections 2.5 and 3).

**The gap this software fills.** ForgeCraft provides a single, reproducible pipeline in which a task specification and a parts catalogue are transformed into an evolved design *together with* the documentation required to build it. Every stage is instrumented, every artefact is serialisable, and the entire workflow is covered by automated tests that run on accessible, non-proprietary software.

---

## 2. Software description

### 2.1 Software architecture

ForgeCraft is organised as a layered Python package of 192 modules (~60,000 lines) with a deliberately narrow interface between layers, so that individual research questions can be addressed without modifying the rest of the stack.

**Core representation.** The central data structure is `MechanicalBody`, a directed graph of `Part` and `Joint` objects. Parts carry a type and a parameter dictionary, so a body is a typed graph rather than a fixed topology. The representation is memory-conscious by design (`__slots__` on both classes), because populations of thousands of bodies are kept in memory during search. Parts are drawn from a *catalogue*: a declarative YAML description of available part types, their geometric primitive, parameter ranges, mass, friction and actuation properties. Because the catalogue is data rather than code, new design spaces are defined without changing the framework.

**Evaluation layer.** Evolved bodies are compiled to MuJoCo MJCF models for rigid-body simulation, with an optional Genesis backend for GPU-accelerated evaluation. To keep search tractable, the evaluation path uses deliberately simplified collision geometry (convex proxies), decoupling the cost of physics evaluation from the cost of geometric detail. Environments follow the Gymnasium API, and task specifications (reward composition, episode budget, termination criteria) are themselves declarative YAML.

**Search layer.** Co-evolution is orchestrated by an evolution loop that alternates between population-level search over morphologies and reinforcement-learning updates of the controllers conditioned on those morphologies. Morphologies are encoded by a message-passing graph neural network [8] into fixed-length embeddings; controllers are PPO [9] actors conditioned on those embeddings. Conditioning the policy on the morphology embedding allows a controller to be trained for an arbitrary topology, but we emphasise that ForgeCraft trains a *separate* controller per evaluated individual within that individual's evaluation budget (optionally warm-started from the previous generation's trainer state): the framework does not currently learn a single policy that generalises zero-shot across body topologies, and we do not claim such generalisation. Selection combines elitism, tournament and Pareto-based multi-objective selection (NSGA-II/III [10,11]); a MAP-Elites archive [12] maintains behavioural diversity. Search is made affordable by adaptive evaluation budgets that grow with generation index, and the framework supports checkpoint/resume.

**Geometry and manufacturing layer.** This layer is what distinguishes ForgeCraft from morphology-only frameworks. A parametric generator produces concrete mechanical geometry for each abstract part — chassis with ribs and counterbores, motors with cooling fins and end bells, bearings with shields, tubes with threaded ends, springs, feet and battery packs — from the same parameter dictionary used during search. Because the search-time bodies and export-time geometry share parameters, the exported artefacts correspond to the design that was actually evaluated.

**Auxiliary layers.** Optional modules provide finite-element and isogeometric analysis, additive-manufacturing pre-processing (slicing, infill patterns, G-code generation), knowledge storage, and LLM-assisted design agents. These are isolated so that the core pipeline has a small dependency footprint.

### 2.2 Software functionalities

#### 2.2.1 Morphology representation and generation

Design spaces are defined by catalogues (21 shipped, spanning primitive parts, legged robots, grippers, hoppers and parameterised part libraries). Bodies can be generated randomly, assembled from locomotion templates, or constructed by the genetic operators. Catalogue constraints (tree depth, branching factor, part budget, symmetry) are enforced during generation, so search never produces invalid graphs.

#### 2.2.2 Morphology–control co-evolution

The `EvolutionLoop` implements the full co-evolution cycle: initialisation, UCB-scheduled evaluation, breeding (selection, crossover, mutation), elite injection from the MAP-Elites archive, and checkpointing. Genetic operators include both parametric and topological mutation, with adaptive mutation rates driven by population diversity. Multi-objective tasks return Pareto fronts rather than scalar fitness. Optional features include encoder pre-training by contrastive learning, curriculum over task difficulty, and distributed evaluation over Ray or multiprocessing; both distributed paths dispatch to the same rollout-and-PPO worker used in-process, so their results are consistent with the single-machine path by construction, although the distributed paths are not exercised in the results reported here (Section 2.5).

#### 2.2.3 Physics-based evaluation

Fitness is computed from rollouts in MuJoCo, combining task rewards (locomotion speed, climbing, acceleration, efficiency) with stability and survivability terms. Evaluation is parallelised across worker processes with progressive episode budgets, and observation normalisation statistics are checkpointed so that resumed runs remain consistent. One modelling detail determines whether these terms carry any signal at all, and we therefore state it explicitly: the MJCF builder attaches a `<freejoint/>` to the root body, so every compiled model has a full six-degree-of-freedom floating base. An earlier revision of the builder omitted that element, which welded the root to the world frame, reduced the model's degrees of freedom to the number of internal joints, held the centre-of-mass sensor constant, and made displacement-, speed- and fall-based terms identically zero — with the consequence that evolutionary search received no locomotion signal whatsoever. That defect is fixed in the present release; Section 2.5 records the measured before/after effect rather than only the repair.

#### 2.2.4 Parametric geometry and product-level visualisation

The parametric generator produces nine classes of mechanical parts at four quality levels (vertex-density presets from interactive to presentation grade). Parts are assigned physically based rendering materials from ten presets (carbon fibre, 7075 aluminium, chrome steel, spring steel, LiPo enclosures, TPU and others). The pipeline exports interactive glTF 2.0 scenes in both assembled and exploded configurations, and renders multi-view figures with configurable viewpoints — useful because morphology papers depend heavily on visual communication. The renderer is deliberately lightweight: it rasterises meshes through matplotlib with per-face Lambert diffuse plus Blinn-Phong specular and Fresnel terms under a three-light rig, which is adequate for schematic figures but is not a physically based path tracer and should not be used for material comparisons.

#### 2.2.5 Manufacturing-aware export

`export_all_enhanced` performs the manufacturing stage end to end:

1. **Parametric solid generation** for every part in the evolved body.
2. **Connection features** — bolt holes, hinge bearing seats and snap fits generated from joint type, with clearances derived from FDM tolerance profiles.
3. **Tolerance and shrinkage compensation** for the selected print profile.
4. **Support and orientation planning**, with support-volume reporting.
5. **Interference checking** between mating part pairs, classifying fits as clearance, transition or interference.
6. **STEP AP242 assembly** export for CAD interchange. The exported entities are tessellated (`TESSELLATED_ITEM` with `COORDINATES_LIST` and `TRIANGULATED_FACE` under the `AP242_MANAGED_MODEL_BASED_3D_ENGINEERING_MIM_LF` schema): the file is valid AP242 and readable by tessellation-aware viewers, but it carries triangle meshes rather than analytic NURBS surfaces and topological B-rep edges, so it cannot be edited parametrically on import.
7. **2D engineering drawings** (DXF) with tolerance annotation.
8. **Bill of materials** with mass roll-up and material recommendations.
9. **Manufacturability scoring**, decomposing into build-volume, wall-thickness, assembly-complexity, torque-budget and cantilever checks.
10. **Production packaging** into a single archive that can be handed to a fabricator, together with a URDF for downstream robotics tooling and an optional ONNX export of the trained policy for deployment.

Manufacturability scores can be consumed by the search layer, allowing buildability to be optimised rather than merely reported.

### 2.3 Design decisions and trade-offs

Four architectural decisions shape the framework and are worth stating explicitly, since each resolves a tension rather than being an obvious default.

**Dual geometry channels.** Geometric fidelity and evaluation throughput pull in opposite directions. High-fidelity part geometry is essential for manufacturing output, but using it in physics simulation would make the evaluation of thousands of candidate bodies prohibitively expensive; conversely, simplified collision proxies make search tractable but cannot be exported. ForgeCraft therefore maintains two representations derived from the *same* parameter set: simplified convex proxies for simulation, and parametric solids for export. The deliberate cost of this design is that the two representations can disagree, so a body may be collision-valid in simulation while its detailed geometry interferes. We therefore treat interference checking as a required stage on the export channel rather than an internal assumption, and report its outcome as first-class output.

**Declarative design spaces.** Catalogues, task definitions, material assignments and print profiles are expressed as data (YAML) rather than code. This introduces an indirection layer and requires up-front validation, but it means a new design space is a configuration file rather than a fork of the framework — which we consider decisive for ForgeCraft's intended role as a *shared* testbed, since design spaces can be exchanged and version-controlled independently of the implementation.

**Quality as a discrete dial.** Vertex density, rendering quality and export completeness are selected through a small number of presets rather than exposed as per-call parameters. This trades fine-grained control for reproducibility: any figure or mesh reported at a given quality level can be regenerated exactly, which matters when geometric output is itself a reported result.

**Isolation of heavy optional dependencies.** Finite-element analysis, GPU simulation, distributed evaluation and LLM-assisted design are isolated behind optional installation extras and import guards. This keeps the core installation small and, more importantly, allows the test suite to execute in environments without those dependencies. This is the mechanism by which continuous integration remains green across three Python versions without specialised hardware — and the absence of such isolation was the source of several real defects found while preparing this release.

### 2.4 Verification

The repository contains 424 automated tests: 253 in the core suite and 171 module-level tests. Coverage includes morphology graph invariants, catalogue loading, genetic operators, selection schemes, checkpoint round-trip, physics model construction, geometry generation, tolerance and connector calculation, interference checking and export formats. Continuous integration runs the suite on Python 3.10, 3.11 and 3.12 at every commit; GPU-dependent tests are skipped automatically when no CUDA device is present.

### 2.5 Known limitations

We list the known defects and scope boundaries of v0.4.0 explicitly. Every item below is reproducible from the repository, and each is covered by the "Known limitations" section of the README, so the software description and the paper cannot drift apart.

**Fixed: the root body was welded to the world frame.** In the previous revision `forgecraft/simulation/builder.py` emitted `<body name="..." pos="...">` for the root without a `<freejoint/>`, so the compiled model's degrees of freedom equalled the number of internal joints (measured then: a six-part, five-joint body compiled to `nq = nv = 5`, i.e. with no floating base), the centre-of-mass sensor was constant, termination criteria of the form `_com_z < fall_height` never fired, and the `displacement`, `speed` and `velocity` reward terms were identically zero. This was the most consequential defect in the release, because it removed the locomotion signal from the evolutionary objective altogether. It is now repaired: the root carries a `<freejoint/>`, and the same six-part body compiles to `nq = 9, nv = 8` (the free joint contributes seven positions and six degrees of freedom) with joints ordered `free` at `qposadr = 0`, hinges at `qposadr = 7` and `8`. In a 300-step zero-action rollout the centre of mass descends from 0.476 m to 0.191 m (Δ = −0.286 m), translates 0.064 m along x, and fires the fall termination at step 299; all of these quantities were exactly zero before the repair. Reproducing the fix also exposed two latent errors in `forgecraft/rl/env.py`, which are fixed with it: `step()` returned `np.bool_` rather than a Python `bool` for `truncated` (a Gymnasium contract violation, masked previously because the sensor branch was never reached), and `reset()` perturbed `qpos` by joint index, which with a floating base corrupts the base position and quaternion; it now addresses `qpos` through `jnt_qposadr` and skips free and ball joints. The repair was validated end to end by re-running the full evolutionary experiment reported in Section 3 (`--catalog speedster --task speed --population 32 --generations 100 --seed 42`): the resulting best individual attains a composite fitness of 12954.87 whose movement terms are non-zero (`speed` = 2497.80, `displacement` = 10452.36), and it contains **eight actuated joints** — whereas before the repair the only non-zero component of any individual was `upright` and the search never produced an actuated body.

**Control training is per-individual.** A separate `PPOTrainer` is constructed for each evaluated individual and updated within that individual's evaluation budget, optionally warm-started from the previous generation. No single policy is shared across bodies, and no zero-shot transfer across topologies is claimed.

**The shipped example body is a prototype, not an engineered design.** As detailed in Section 3, the demo individual now does move — it carries eight actuated joints and its fitness contains non-zero `speed` and `displacement` terms — but the search exploited the objective in ways the objective does not penalise: it contains 120 colliding part pairs and twelve degenerate joint ranges (seven collapsed to `[0, 0]`, five unbounded, the largest spanning 888 rad), at 6.233 kg and USD 442.11. Self-collision penalties and physically meaningful joint limits are therefore not yet part of the fitness function, which is future work. The example demonstrates that the pipeline runs end to end and that search receives a locomotion signal; it is not evidence of a high-performing design.

**Two manufacturability aggregates coexist.** `manufacturability.json` reports `overall = 0.9925`, a blend of five geometric-feasibility dimensions, alongside `manufacturability_score = 0.70`, which is driven by collision detection. The two are not reconciled in code.

**STEP output is tessellated.** Exports conform to AP242 but carry triangle meshes rather than analytic B-rep geometry (see Section 2.2.5).

**Rendering is not physically based.** Figures are rasterised through matplotlib with per-face Lambert/Blinn-Phong/Fresnel shading under a fixed three-light rig; this is adequate for schematic views but is not a path tracer.

**No performance benchmarks are shipped.** The repository contains no FPS or speed-up measurements. The optional Genesis backend must be installed and benchmarked locally; throughput figures that appeared in early documentation drafts have been removed.

**Degenerate joint ranges are reported, not rejected.** Independent noise on `range_min` and `range_max` used to invert joint limits; range ordering is now normalised in the genetic operators and the joint optimiser, and the regenerated demonstration snapshot no longer contains any `hi < lo` range. Degenerate limits nevertheless persist in a different form, because nothing constrains how wide a range may become: the shipped individual has seven joints collapsed to `[0, 0]` and five with unbounded ranges (up to 888 rad). The exporter surfaces all such cases as warnings rather than refusing to export, so a physically implausible limit reaches the delivered files unless the reader inspects the report.

**Distributed evaluation is implemented but not exercised in the reported results.** The Ray and multiprocessing paths in `forgecraft/evolution/distributed.py` dispatch to the same rollout-and-PPO worker used in-process (`_evaluate_body_worker_ucb`), so they agree with the single-machine path by construction; the reported experiments nevertheless use the in-process parallel evaluator and no cluster measurement is included. The `RedisTaskQueue` helper supplies queue primitives only and is not wired into `evaluate_population`.

**The finite-element solver discretises the bounding box, not the part.** `surface_to_tetrahedralize` in `forgecraft/analysis/fea.py` derives its grid spacing from the part's bounding box (`extents = bmax - bmin`) and fills that box with regular volume elements; it does not tetrahedralise the part's surface geometry. Results therefore describe a solid block of the part's outer dimensions and must not be read as a stress or safety-factor check of the actual part. The wider `forgecraft/analysis/` package (isogeometric analysis, multiphysics, fracture) is likewise not wired into the evolutionary or manufacturing paths, and none of it is used in the results reported here.

**The topology optimiser does not solve a mechanical problem.** The SIMP loop in `forgecraft/analysis/topology.py` never assembles or solves `Ku = f`; `scipy.sparse.linalg.spsolve` is imported but never called. Its "sensitivity" is `dc = -p·x^(p-1)`, which depends only on the density field and contains neither displacements nor element stiffness, so the loop redistributes material under a volume constraint without any structural mechanics. The `compliance` field of `TopologyResult` is hard-coded to `0.0` and is not a compliance value.

---

## 3. Illustrative examples

The complete workflow is exercised by an example shipped in the repository (`examples/demo_robot/`), which records the artefacts of a completed evolution run (generation 79, individual 007).

**Invocation.** A run is a single command:

```
python -m forgecraft.main --catalog speedster --task speed \
    --population 32 --generations 100 --export --export-dir design_output
```

The evolution loop prints per-generation statistics. The CLI `--export` flag drives the lightweight export path (STL, STEP, 3MF, URDF, BOM and a markdown/JSON review report). The fuller presentation chain, which additionally produces DXF drawings, glTF scenes, rendered views and a ZIP production archive, is `export_all_enhanced` and is reached through the Python API; the presentation artefacts in the example come from that chain. Specifically, the committed scenes and renders were produced by Phase 3 of the chain invoked directly as `export_presentation_suite(body_data, outdir, prefix="gen89", quality="high", explode_distance=0.12)`, so that artefact filenames carry the individual's generation tag; `export_all_enhanced` derives the same prefix from the body name.

**Resulting artefacts and their interpretation.** The recorded design comprises 25 parts and 24 joints with a stored composite fitness of 12954.87. Unlike the previous release, whose only non-zero component was `upright`, this fitness is dominated by movement: `displacement` = 10452.36 and `speed` = 2497.80, alongside `upright` = 46.41 and a manufacturability term of 0.9925. The body has **eight actuated joints**, so the review report no longer emits the "no actuated joints, the body cannot move under its own power" warning that the previous snapshot carried. We nevertheless report its manufacturing diagnostics in full, because they remain more informative about the state of the release than the fitness value is: 120 colliding part pairs and twelve joints with implausible parameter ranges (seven collapsed to `[0, 0]`, five unbounded, the largest spanning 888.18 rad). Its manufacturability aggregate is 0.9925 while a separate `manufacturability_score` is 0.70, the latter driven by the collision result. The review report estimates 6.233 kg and USD 442.11 (material 186.98, printing 255.13) at 85.0 h and 2862 g of PLA, whereas `bom.json` — which derives mass from bounding-box volume and unit cost from a separate price table — reports 5.244 kg and USD 157.31 for the same design.

Two caveats follow, and we state them rather than omit them. First, a stored fitness of 12954.87 is **not** evidence of a high-performing robot: it is a weighted sum dominated by movement terms, and the search reaches it while the body's parts interpenetrate in 120 pairs and its joints admit physically meaningless travel, neither of which the objective penalises. The honest reading is that the objective now carries a locomotion signal — which, before the repair described in Section 2.2.3, it did not — and that self-collision and joint-limit terms must be added before evolved bodies can be read as engineering designs. Second, the aggregate figures disagree in several places for reasons that are not reconciled in code: `overall` (0.9925) and `manufacturability_score` (0.70) differ because the former blends five geometric-feasibility dimensions (build volume, wall thickness, assembly complexity, torque budget and cantilever) and excludes collisions, while the mass and cost reported by `bom.json` differ from those of the review report because the two compute volume and unit cost by different formulas. We report all of them rather than the most flattering one; reconciling them is future work. We therefore present this example as evidence that the pipeline runs end to end, that evolutionary search now receives a locomotion signal, and that the export chain emits inspectable, machine-readable deliverables — not as evidence that it discovered a high-performing robot.

The artefacts committed under `examples/demo_robot/` are the body description, the evolution history, the review report, the bill of materials, the manufacturability JSON, two interactive glTF scenes (assembled and exploded), 13 rendered views (seven canonical viewpoints plus three paper-oriented views in normal and exploded states), three part-showcase figures and a single-file HTML viewer. STL, STEP, DXF, URDF and ZIP archive outputs are produced by the export stage but are deliberately not committed, to keep the repository small; they regenerate from `best_body.json`.

**Reproducibility.** Runs are seeded, task and catalogue definitions are declarative, and checkpoints capture both evolutionary and training state, so an interrupted run resumes to an identical configuration.

---

## 4. Impact

ForgeCraft is a newly released framework; as such this section describes the impact it is designed to enable rather than a body of completed external use.

**Enabling reproducible comparison.** Morphology co-evolution results are often reported on bespoke codebases that cannot be re-run. By providing a tested, documented and permissively licensed implementation with declarative design spaces, ForgeCraft allows design spaces and tasks to be shared as configuration files, making comparisons between methods concrete rather than approximate.

**Making manufacturability a research variable.** Because buildability can enter both the objective (a weighted fitness term with a minimum-score constraint, enabled per task) and the reporting, the framework permits a question that is currently difficult to study: how does the structure of evolutionary search change when designs must be buildable? Treating part standardisation, minimum feature size and assembly complexity as objectives rather than filters turns a post-hoc constraint into an experimental factor. We note that the framework makes this study *possible*; we have not yet carried it out.

**Bridging two communities.** The output formats target the tools that mechanical engineers already use — STEP AP242 for CAD interchange, DXF for drawing review, bills of materials for procurement. A researcher can therefore carry an evolved design into a mechanical workflow without re-modelling it from scratch, though since the STEP geometry is tessellated rather than analytic, a fabricator will typically need to rebuild parametric features from the drawings and the bill of materials. Closing that gap — and ultimately returning a physical robot, enabling simulation-to-reality studies of evolved morphologies — remains an open objective rather than a demonstrated capability.

**Education and accessibility.** The full pipeline runs on commodity laptops with no proprietary software, and the repository includes a browser-viewable interactive model and rendered outputs. This supports teaching co-design as an end-to-end engineering activity — from fitness function to drawing sheet — rather than as an isolated optimisation exercise.

**Extensibility.** Catalogues, tasks, part generators, materials, fitness functions and export stages are all defined through narrow interfaces or declarative data, so domain-specific extensions (new actuator types, new manufacturing processes, alternative encodings) do not require modifying the core.

---

## 5. Conclusions

ForgeCraft is an open-source framework that couples morphology–control co-evolution with a manufacturing-aware export chain, so that evolved robot designs are delivered together with the geometry, drawings, tolerances and documentation needed to build them. The framework integrates established components — graph-based morphology representation, GNN-encoded PPO control, quality-diversity and multi-objective search, MuJoCo physics — and adds an explicit representation of evolved bodies as parametric mechanical parts, with manufacturability available as an optimisation objective. Distributed under the MIT licence with 424 automated tests and continuous integration, it is intended as reusable infrastructure for research at the boundary between evolutionary robotics and design for manufacturing.

Future work is led by repair. The first priority is the root-free-joint defect of Section 2.5, which currently welds the root body to the world and zeroes every displacement-based reward; until it is fixed, the search chain cannot optimise locomotion and the framework's numbers should not be read as performance results. After that, the empirical questions the framework is built to answer come into reach: quantifying how manufacturability constraints reshape evolved morphologies, reconciling the two manufacturability aggregates, and validating the loop by fabricating and testing evolved designs in hardware.

We state the scope of this release plainly: v0.4.0 is a research prototype whose manufacturing-aware export chain is functional and exercised end to end, and whose co-evolution search chain is not yet trustworthy because of the modelling defect above. The reference example is a pipeline-completeness demonstration, not a motile design. We prefer this description to the more attractive claims that an unexamined reading of the outputs might invite.

---

## CRediT authorship contribution statement

【待填：作者贡献声明，例如】
**姓名**: Conceptualization, Methodology, Software, Validation, Writing – original draft.

## Declaration of competing interest

The authors declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

## Acknowledgements

【待填：若有资助/致谢请填写；无则删除本节】

## References

【注意：以下文献均为该领域的真实代表性工作，但**投稿前必须逐条核对作者、年份、卷期与 DOI**。】

1. K. Sims, "Evolving virtual creatures," in *Proc. 21st Annual Conf. on Computer Graphics and Interactive Techniques (SIGGRAPH)*, 1994, pp. 15–22.
2. A. Zhao, J. Xu, M. Konaković-Luković, J. Hughes, A. Spielberg, D. Rus, W. Matusik, "RoboGrammar: Graph grammar for terrain-optimized robot design," *ACM Transactions on Graphics*, vol. 39, no. 6, 2020.
3. A. Gupta, S. Savarese, S. Ganguli, L. Fei-Fei, "Embodied intelligence via learning and evolution," *Nature Communications*, vol. 12, 2021.
4. Y. Yuan, Y. Song, Z. Luo, W. Sun, C. Kitani, "Transform2Act: Learning a transform-and-control policy for efficient agent design," in *Proc. Int. Conf. on Learning Representations (ICLR)*, 2022.
5. D. Pathak, C. Lu, T. Darrell, P. Isola, A. A. Efros, "Learning to control self-assembling morphologies: A study of generalization via modularity," in *Advances in Neural Information Processing Systems (NeurIPS)*, 2019.
6. J.-B. Mouret, J. Clune, "Illuminating search spaces by mapping elites," arXiv:1504.04909, 2015.
7. Y. Hu, L. Anderson, T.-M. Li, Q. Sun, N. Carr, J. Ragan-Kelley, F. Durand, "DiffTaichi: Differentiable programming for physical simulation," in *Proc. ICLR*, 2020.
8. J. Gilmer, S. S. Schoenholz, P. F. Riley, O. Vinyals, G. E. Dahl, "Neural message passing for quantum chemistry," in *Proc. Int. Conf. on Machine Learning (ICML)*, 2017.
9. J. Schulman, F. Wolski, P. Dhariwal, A. Radford, O. Klimov, "Proximal policy optimization algorithms," arXiv:1707.06347, 2017.
10. K. Deb, A. Pratap, S. Agarwal, T. Meyarivan, "A fast and elitist multiobjective genetic algorithm: NSGA-II," *IEEE Transactions on Evolutionary Computation*, vol. 6, no. 2, pp. 182–197, 2002.
11. K. Deb, H. Jain, "An evolutionary many-objective optimization algorithm using reference-point-based nondominated sorting approach, part I," *IEEE Transactions on Evolutionary Computation*, vol. 18, no. 4, pp. 577–601, 2014.
12. J.-B. Mouret, J. Clune, "Quality diversity: A new frontier for evolutionary computation," *Frontiers in Robotics and AI*, vol. 3, 2016.
13. E. Todorov, T. Erez, Y. Tassa, "MuJoCo: A physics engine for model-based control," in *Proc. IEEE/RSJ Int. Conf. on Intelligent Robots and Systems (IROS)*, 2012, pp. 5026–5033.
14. G. Brockman et al., "OpenAI Gym," arXiv:1606.01540, 2016.
15. N. Heess et al., "Emergence of locomotion behaviours in rich environments," arXiv:1707.02286, 2017.
16. X. B. Peng, P. Abbeel, S. Levine, M. van de Panne, "DeepMimic: Example-guided deep reinforcement learning of physics-based character skills," *ACM Transactions on Graphics*, vol. 37, no. 4, 2018.
17. A. Cully, J. Clune, D. Tarapore, J.-B. Mouret, "Robots that can adapt like animals," *Nature*, vol. 521, pp. 503–507, 2015.
18. J. Lehman, K. O. Stanley, "Abandoning objectives: Evolution through the search for novelty alone," *Evolutionary Computation*, vol. 19, no. 2, pp. 189–223, 2011.
19. P. Grbic et al., "EvoCraft: A new challenge for open-endedness," arXiv:2012.04751, 2020.
20. G. Gibson, D. Rosen, B. Stucker, *Additive Manufacturing Technologies*, 3rd ed., Springer, 2021.
21. G. Boothroyd, P. Dewhurst, W. Knight, *Product Design for Manufacture and Assembly*, 3rd ed., CRC Press, 2010.
22. ISO 10303-242, *Industrial automation systems and integration — Product data representation and exchange — Part 242: Managed model-based 3D engineering*, ISO, 2022.
23. ISO 2768-1, *General tolerances — Part 1: Tolerances for linear and angular dimensions without individual tolerance indications*, ISO, 1989.
24. A. Vaswani et al., "Attention is all you need," in *Advances in Neural Information Processing Systems (NeurIPS)*, 2017.
25. P. Veličković, G. Cucurull, A. Casanova, A. Romero, P. Liò, Y. Bengio, "Graph attention networks," in *Proc. ICLR*, 2018.
26. S. Fujimoto, H. van Hoof, D. Meger, "Addressing function approximation error in actor-critic methods," in *Proc. ICML*, 2018.

---

## 投稿前必做清单（非论文正文）

- [ ] **补充作者信息**：姓名、单位、通讯邮箱（文中标了【待填】的位置）
- [ ] **打版本标签**：`git tag v0.4.0 && git push origin v0.4.0`，否则 C2/S2 的链接会 404
- [ ] **核对全部参考文献**的作者/年份/卷期/DOI——草稿中的条目按领域记忆整理，必须逐条核实
- [ ] **确认 `src/` 布局要求**：SoftwareX 建议源码放在 `src/` 目录；当前是仓库根目录下的 `forgecraft/` 包。建议先与编辑部确认，或评估迁移到 src-layout 的成本
- [ ] **转换为官方模板**：用 SoftwareX 的 DOC 或 LaTeX 模板重排（本文件为内容草稿）
- [ ] **补一张架构图与一张流程图**（论文通常需要，可从 `examples/demo_robot/` 的渲染图改造）
- [ ] **AI 使用声明**：如期刊要求，需声明本文与代码撰写中生成式 AI 的使用范围
