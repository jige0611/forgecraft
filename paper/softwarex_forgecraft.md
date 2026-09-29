# ForgeCraft: An open-source framework for manufacturing-aware co-evolution of robot morphology and control

**作者**：【待填：姓名】<sup>a, *</sup>

<sup>a</sup> 【待填：单位 / 地址，例如 Department of ..., University of ..., City, Country】

<sup>*</sup> 通讯作者。E-mail: 【待填：通讯邮箱】

---

## Abstract

ForgeCraft is an open-source Python framework for the co-evolution of robot morphology and control, with an explicit emphasis on the transition from simulation to manufacturing. Most morphology-evolution research terminates at simulated locomotion: the resulting designs are evaluated in physics engines but are rarely manufacturable, because part geometry, tolerance stack-ups, standard fasteners and assembly constraints are absent from the search space. ForgeCraft couples four components into a single pipeline. A graph-based morphology representation builds bodies from a configurable parts catalogue. A co-evolution engine combines genetic operators, quality-diversity search (MAP-Elites) and multi-objective selection with a graph-neural-network morphology encoder and PPO control training. A physics evaluation layer supports MuJoCo and, where available, the Genesis backend. Finally, a manufacturing-aware export chain converts evolved bodies into parametric mechanical parts and produces STEP AP242 assemblies (tessellated geometry rather than analytic B-rep), 2D DXF drawings with tolerance annotation, interference reports, bills of materials and one-click production packages. Manufacturability is treated as a first-class objective rather than a post-hoc filter. The framework ships with 21 parts catalogues, 8 task definitions, 9 parametric part generators, 10 PBR material presets, and 424 automated tests executed in continuous integration on Python 3.10–3.12. We describe the architecture, illustrate the complete workflow on a recorded design, and outline the research applications the framework is intended to enable. Section 2.5 states the current limitations of the release explicitly, including a simulation-modelling defect that disables displacement- and fall-based reward terms under the default configuration.

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

Fitness is computed from rollouts in MuJoCo, combining task rewards (locomotion speed, climbing, acceleration, efficiency) with stability and survivability terms. Evaluation is parallelised across worker processes with progressive episode budgets, and observation normalisation statistics are checkpointed so that resumed runs remain consistent. One caveat must be stated here because it affects how the reported numbers should be read: in the current default configuration the MJCF builder attaches no free joint to the root body, so the root is welded to the world frame. Degrees of freedom therefore equal the count of internal joints, which makes displacement-, speed- and fall-based terms identically zero under that configuration. This is a modelling defect, documented in Section 2.5 and tracked as a fix for the next release, not a deliberate simplification.

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

**Root body is welded to the world frame.** `forgecraft/simulation/builder.py` emits `<body name="..." pos="...">` for the root without a `<freejoint/>`; the free joint appears only in the fallback builder `_build_fallback_xml()`. The compiled model's degrees of freedom therefore equal the number of internal joints (measured: a six-part, five-joint body compiles to `nq = nv = 5`, i.e. no six-DoF floating base), the centre-of-mass sensor is constant, termination criteria of the form `_com_z < fall_height` never fire, and `displacement`, `speed` and `velocity` reward terms are identically zero. This is the most consequential defect in the release. We deliberately did not fix it in this version because doing so invalidates the committed demonstration artefacts and the numerical expectations of several existing tests; it is the first item on the roadmap.

**Control training is per-individual.** A separate `PPOTrainer` is constructed for each evaluated individual and updated within that individual's evaluation budget, optionally warm-started from the previous generation. No single policy is shared across bodies, and no zero-shot transfer across topologies is claimed.

**The shipped example body is not a performance result.** As detailed in Section 3, the demo individual has zero actuated joints, 76 colliding pairs and six degenerate joint ranges, and its only non-zero fitness component is `upright`. It demonstrates pipeline completeness, not design quality.

**Two manufacturability aggregates coexist.** `manufacturability.json` reports `overall = 0.9954`, a blend of five geometric-feasibility dimensions, alongside `manufacturability_score = 0.70`, which is driven by collision detection. The two are not reconciled in code.

**STEP output is tessellated.** Exports conform to AP242 but carry triangle meshes rather than analytic B-rep geometry (see Section 2.2.5).

**Rendering is not physically based.** Figures are rasterised through matplotlib with per-face Lambert/Blinn-Phong/Fresnel shading under a fixed three-light rig; this is adequate for schematic views but is not a path tracer.

**No performance benchmarks are shipped.** The repository contains no FPS or speed-up measurements. The optional Genesis backend must be installed and benchmarked locally; throughput figures that appeared in early documentation drafts have been removed.

**Degenerate joint ranges are reported, not rejected.** Independent noise on `range_min` and `range_max` used to invert joint limits; range ordering is now normalised in the genetic operators and the joint optimiser. The committed demonstration snapshot predates that fix, so it still exhibits `hi < lo` ranges, and the exporter surfaces such cases as warnings rather than refusing to export.

**Distributed evaluation is implemented but not exercised in the reported results.** The Ray and multiprocessing paths in `forgecraft/evolution/distributed.py` dispatch to the same rollout-and-PPO worker used in-process (`_evaluate_body_worker_ucb`), so they agree with the single-machine path by construction; the reported experiments nevertheless use the in-process parallel evaluator and no cluster measurement is included. The `RedisTaskQueue` helper supplies queue primitives only and is not wired into `evaluate_population`.

---

## 3. Illustrative examples

The complete workflow is exercised by an example shipped in the repository (`examples/demo_robot/`), which records the artefacts of a completed evolution run (generation 79, individual 007).

**Invocation.** A run is a single command:

```
python -m forgecraft.main --catalog speedster --task speed \
    --population 32 --generations 100 --export --export-dir design_output
```

The evolution loop prints per-generation statistics. The CLI `--export` flag drives the lightweight export path (STL, STEP, 3MF, URDF, BOM and a markdown/JSON review report). The fuller presentation chain, which additionally produces DXF drawings, glTF scenes, rendered views and a ZIP production archive, is `export_all_enhanced` and is reached through the Python API; the presentation artefacts in the example come from that chain.

**Resulting artefacts and their interpretation.** The recorded design comprises 16 parts and 15 joints with a stored composite fitness of 0.8498. We report its manufacturing diagnostics in full, because they are more informative about the state of the release than the fitness value is. The design has **zero actuated joints** — the review report emits the explicit warning "no actuated joints, the body cannot move under its own power" — 76 colliding part pairs, and six joints with degenerate parameter ranges (four where `hi < lo`, two collapsed to `[0, 0]`). Its manufacturability aggregate is 0.9954 while a separate `manufacturability_score` is 0.70; the bill of materials totals USD 325.78 (material 137.78, printing 188.00) at an estimated 62.7 h and 2109 g of PLA.

Two caveats follow, and we state them rather than omit them. First, the stored fitness is **not** evidence of locomotion: because of the root-free-joint defect described in Section 2.2.3, the only non-zero fitness component of this body is `upright` (5.2209), while every displacement-, speed- and energy-related component is exactly zero. Second, the two manufacturability figures disagree because they are computed from different inputs — the 0.9954 aggregate blends five geometric-feasibility dimensions (build volume, wall thickness, assembly complexity, torque budget and cantilever) but does not fold in the collision result that drives the 0.70 score. We report both rather than the more flattering one; reconciling them is future work. We therefore present this example as evidence that the pipeline runs end to end and emits inspectable, machine-readable deliverables — not as evidence that it discovered a high-performing robot.

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
