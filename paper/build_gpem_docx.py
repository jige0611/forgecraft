"""Build the GPEM (Genetic Programming and Evolvable Machines) submission source.

GPEM accepts .docx or LaTeX source files. This script renders the original-research
manuscript to ``paper/gpem_forgecraft.docx`` using python-docx, so the file can be
uploaded to Editorial Manager without any local TeX installation.

Run:
    .venv/Scripts/python.exe paper/build_gpem_docx.py
"""

from __future__ import annotations

import re
from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "paper" / "gpem_forgecraft.docx"
RENDERS = ROOT / "examples" / "demo_robot" / "renders"

BODY_FONT = "Times New Roman"
BODY_SIZE = Pt(10)
MONO_FONT = "Consolas"
LINE_SPACING = 0.98
SPACE_AFTER = Pt(2.5)

RICH_RE = re.compile(r"(\*\*[^*]+\*\*|\*[^*]+\*|`[^`]+`)")


# --------------------------------------------------------------------------- #
# document scaffolding
# --------------------------------------------------------------------------- #
def new_document() -> Document:
    doc = Document()
    normal = doc.styles["Normal"]
    normal.font.name = BODY_FONT
    normal.font.size = BODY_SIZE
    normal.element.rPr.rFonts.set(qn("w:eastAsia"), "SimSun")

    for section in doc.sections:
        section.top_margin = Inches(0.75)
        section.bottom_margin = Inches(0.75)
        section.left_margin = Inches(0.75)
        section.right_margin = Inches(0.75)
    return doc


def add_rich(paragraph, text: str) -> None:
    """Write text into a paragraph, honouring **bold**, *italic* and `mono`."""
    for chunk in RICH_RE.split(text):
        if not chunk:
            continue
        if chunk.startswith("**") and chunk.endswith("**"):
            run = paragraph.add_run(chunk[2:-2])
            run.bold = True
        elif chunk.startswith("*") and chunk.endswith("*") and len(chunk) > 2:
            run = paragraph.add_run(chunk[1:-1])
            run.italic = True
        elif chunk.startswith("`") and chunk.endswith("`"):
            run = paragraph.add_run(chunk[1:-1])
            run.font.name = MONO_FONT
            run.font.size = Pt(9)
        else:
            paragraph.add_run(chunk)


def title(doc: Document, text: str) -> None:
    par = doc.add_paragraph()
    par.alignment = WD_ALIGN_PARAGRAPH.CENTER
    par.paragraph_format.space_after = Pt(8)
    run = par.add_run(text)
    run.bold = True
    run.font.size = Pt(14)


def centred(doc: Document, text: str, size: float = 10.5, italic: bool = False) -> None:
    par = doc.add_paragraph()
    par.alignment = WD_ALIGN_PARAGRAPH.CENTER
    par.paragraph_format.space_after = Pt(3)
    add_rich(par, text)
    for run in par.runs:
        run.font.size = Pt(size)
        if italic:
            run.italic = True


def h1(doc: Document, text: str) -> None:
    par = doc.add_paragraph()
    par.paragraph_format.space_before = Pt(10)
    par.paragraph_format.space_after = Pt(3)
    par.paragraph_format.keep_with_next = True
    run = par.add_run(text)
    run.bold = True
    run.font.size = Pt(12)


def h2(doc: Document, text: str) -> None:
    par = doc.add_paragraph()
    par.paragraph_format.space_before = Pt(7)
    par.paragraph_format.space_after = Pt(2)
    par.paragraph_format.keep_with_next = True
    run = par.add_run(text)
    run.bold = True
    run.italic = True
    run.font.size = Pt(11)


def p(doc: Document, text: str, indent: bool = False, size: float = 10) -> None:
    par = doc.add_paragraph()
    par.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    par.paragraph_format.space_after = SPACE_AFTER
    par.paragraph_format.line_spacing = LINE_SPACING
    if indent:
        par.paragraph_format.left_indent = Inches(0.25)
        par.paragraph_format.right_indent = Inches(0.25)
    add_rich(par, text)
    for run in par.runs:
        run.font.size = Pt(size)


def bullet(doc: Document, text: str) -> None:
    par = doc.add_paragraph(style="List Bullet")
    par.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    par.paragraph_format.space_after = Pt(2)
    par.paragraph_format.line_spacing = LINE_SPACING
    add_rich(par, text)


def numbered(doc: Document, items: list[str]) -> None:
    for item in items:
        par = doc.add_paragraph(style="List Number")
        par.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        par.paragraph_format.space_after = Pt(2)
        par.paragraph_format.line_spacing = LINE_SPACING
        add_rich(par, item)


def caption(doc: Document, text: str) -> None:
    par = doc.add_paragraph()
    par.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    par.paragraph_format.space_after = Pt(8)
    par.paragraph_format.line_spacing = 1.0
    add_rich(par, text)
    for run in par.runs:
        run.font.size = Pt(9)


def side_by_side_figure(doc: Document, images: list[Path], width_in: float, caption_text: str) -> None:
    available = [img for img in images if img.exists()]
    if not available:
        p(doc, "【图缺失】")
        return
    tbl = doc.add_table(rows=1, cols=len(available))
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    for cell, img in zip(tbl.rows[0].cells, available):
        cell.text = ""
        par = cell.paragraphs[0]
        par.alignment = WD_ALIGN_PARAGRAPH.CENTER
        par.paragraph_format.space_after = Pt(0)
        par.add_run().add_picture(str(img), width=Inches(width_in))
    caption(doc, caption_text)


def table(doc: Document, header: list[str], rows: list[list[str]], font_size: float = 9.0):
    tbl = doc.add_table(rows=1, cols=len(header))
    tbl.style = "Table Grid"
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    tbl.autofit = True

    def keep_together(row, repeat_as_header: bool = False) -> None:
        """Prevent a row from breaking across pages; optionally repeat it."""
        tr_pr = row._tr.get_or_add_trPr()
        tr_pr.append(OxmlElement("w:cantSplit"))
        if repeat_as_header:
            tr_pr.append(OxmlElement("w:tblHeader"))

    def fill(cell, text: str, bold: bool = False) -> None:
        cell.text = ""
        par = cell.paragraphs[0]
        par.paragraph_format.space_after = Pt(1)
        par.paragraph_format.line_spacing = 1.0
        add_rich(par, text)
        for run in par.runs:
            run.font.size = Pt(font_size)
            if bold:
                run.bold = True

    for idx, name in enumerate(header):
        fill(tbl.rows[0].cells[idx], name, bold=True)
    keep_together(tbl.rows[0], repeat_as_header=True)
    for row in rows:
        cells = tbl.add_row().cells
        for idx, value in enumerate(row):
            fill(cells[idx], value)
        keep_together(tbl.rows[-1])
    doc.add_paragraph().paragraph_format.space_after = Pt(2)
    return tbl


# --------------------------------------------------------------------------- #
# manuscript
# --------------------------------------------------------------------------- #
def build() -> None:
    doc = new_document()

    title(
        doc,
        "ForgeCraft: closing the gap between evolved robot morphology "
        "and machine-readable manufacturing deliverables",
    )
    centred(doc, "Zhenxin Liu¹*, ORCID: 0009-0005-0773-647X")
    centred(doc, "¹ Beijing Jiaotong University, Beijing, China", size=9.5)
    centred(doc, "* Corresponding author. E-mail: liuzhenxin0611@qq.com", size=9.5, italic=True)

    # ---- abstract -------------------------------------------------------- #
    h1(doc, "Abstract")
    p(
        doc,
        "Morphology–control co-design has produced a large body of simulated robot "
        "designs, yet the field has no shared route from an evolved body to a buildable "
        "artefact. Evaluated morphologies are abstract link-and-joint assemblies, and the "
        "part geometry, tolerances, fasteners and clearances that decide whether a design "
        "can be built are absent from the search space. We present ForgeCraft, an "
        "open-source Python framework that carries an evolved morphology through to "
        "machine-readable manufacturing deliverables. ForgeCraft couples a graph-based "
        "morphology representation assembled from declarative part catalogues with a "
        "co-evolution engine (graph-neural-network morphology encoding, PPO controllers, "
        "NSGA-II/III and MAP-Elites search), MuJoCo physics evaluation, and a parametric "
        "geometry and manufacturing layer that emits tessellated STEP AP242 assemblies, "
        "tolerance-annotated DXF drawings, interference reports, bills of materials, a "
        "manufacturability score and a single production archive. We tested the "
        "framework's central design claim — that buildability is optimised rather than "
        "filtered — by counterfactual ablation of the manufacturability term, and the first "
        "measurement falsified it: removing the term left the top-5 and top-10 of the "
        "population unchanged. Chasing that null result exposed two instrumentation defects, "
        "both of which we report and repair. The scoring function was threshold-triggered and "
        "capped, so its entire range lay inside [0.80, 1.00] and 7 of 21 shipped catalogues "
        "produced an identically constant score; and the weight lookup read the "
        "reward-component namespace while the task files declared the weight elsewhere, so "
        "every task silently fell back to a hard-coded 0.05. After repair — graded checks with "
        "renormalised weights, and an explicit weight namespace that warns instead of falling "
        "back — the mean within-catalogue range of the score rises from 0.024 to 0.225 and no "
        "catalogue is constant. We then report what the repair does and does not buy. Selection "
        "begins to change only when the product of weight and score spread is comparable to the "
        "spread of fitness itself, a diagnostic ratio w·Δmfg/σ(fitness) ≳ 0.4–0.5 that two "
        "catalogues whose fitness spreads differ by a factor of 7.8 both reproduce. Crossing "
        "that threshold did not, however, measurably improve the evolved population: over three "
        "seeds the mean change in population manufacturability is −0.03 with a cross-seed range "
        "of 0.24, and a power analysis puts the number of seeds required to resolve an effect of "
        "that size at 27–143, i.e. 14–72 hours of computation. We therefore present the "
        "criterion as a result and the population-level improvement as an open, "
        "under-powered question. What does hold is the export stage: in every run reported "
        "here, all individuals in the final population contain colliding part pairs, and the "
        "interference report is the only component that surfaces them. We "
        "describe the architecture, position it against three widely used "
        "evolutionary-robotics platforms whose published feature sets stop at simulation, "
        "and report a complete worked run together with its diagnostics — including the "
        "points at which those diagnostics disagree with one another. We state the "
        "limitations of the release explicitly: the reference body is a prototype with 120 "
        "self-colliding part pairs, mass and cost are computed by three unreconciled code "
        "paths, and no evolved design has yet been fabricated.",
        indent=True,
    )
    p(
        doc,
        "**Keywords:** evolutionary robotics · morphology optimisation · design for "
        "additive manufacturing · STEP AP242 · reproducibility · open-source software",
        indent=True,
    )

    # ---- 1 ---------------------------------------------------------------- #
    h1(doc, "1 Introduction")
    p(
        doc,
        "The body of a robot largely determines what that robot can do, and in practice "
        "the body is normally fixed by human designers before any controller is learned. "
        "Co-design — the joint optimisation of morphology and control — has been studied "
        "since the first work on evolving virtual creatures [1] and has been revisited "
        "repeatedly with modern machinery: graph-grammar design spaces [2], learned "
        "transform-and-control policies [4], self-assembling modular morphologies [5], and "
        "quality-diversity search over collections of designs [6, 7].",
    )
    p(
        doc,
        "A persistent limitation runs through this literature. Evaluated designs are "
        "almost always *simulated* bodies: abstract link-and-joint assemblies whose "
        "geometric realisation is left to the reader. An evolved morphology cannot be "
        "built, however, without answering questions the simulation never asks. Which "
        "abstract links correspond to printable parts? Do they fit a build volume? Are the "
        "walls thick enough? Do adjacent parts collide once real bearings, fasteners and "
        "clearances are introduced? Can the joint be assembled at all, and what does the "
        "bill of materials cost? These are not cosmetic concerns: they decide whether an "
        "evolved design leaves the computer. When manufacturability is absent from the "
        "objective, search is free to exploit shapes that are optimal in simulation and "
        "unbuildable in reality, and a post-hoc buildability filter then discards exactly "
        "the designs the search converged on.",
    )
    p(
        doc,
        "Two further gaps compound the problem. **Reproducibility:** morphology-evolution "
        "results are frequently reported for bespoke codebases that cannot be re-run, so "
        "comparisons between methods remain approximate. **Fragmentation:** the "
        "capabilities required for a complete study — morphology generation, physics "
        "evaluation, control learning, geometric modelling and manufacturing preparation — "
        "are normally distributed across separate tools with incompatible data models, so "
        "every stage boundary is crossed by hand.",
    )
    p(
        doc,
        "**Contribution.** ForgeCraft adds the missing stage to this pipeline. Its "
        "distinctive content is not a new co-evolution algorithm — genetic operators, "
        "quality-diversity archives and graph-encoded policies are deliberately taken from "
        "the existing literature — but (i) an explicit parametric representation of "
        "evolved bodies as mechanical parts sharing parameters with the bodies that were "
        "actually evaluated, (ii) an export chain that converts such a body into the "
        "deliverables a fabricator can consume, and (iii) a manufacturing score that is "
        "wired into the objective — together with a two-round measurement, in Section 7, that "
        "shows the shipped release could not actually act on that term, identifies the two "
        "instrumentation defects responsible, and derives from the repaired instrument a "
        "quantitative criterion for when such a term can influence selection. "
        "We also report, in Section 8, the defects and inconsistencies found in our own "
        "release, because a framework making manufacturing claims is only as useful as its "
        "account of where those claims currently stop.",
    )

    # ---- 2 ---------------------------------------------------------------- #
    h1(doc, "2 Related work and the position of this work")
    p(
        doc,
        "Modern co-design systems encode a body as a graph and optimise it with "
        "evolutionary search while a learned controller is fitted to each candidate; "
        "ForgeCraft follows that template and differs in the treatment of the output. The "
        "body graph is not only a simulation input but also the specification from which "
        "concrete mechanical geometry is generated, so the artefact evaluated and the "
        "artefact exported derive from one shared parameter set (Section 3.4).",
    )
    p(
        doc,
        "Three platforms are close enough in ambition to warrant direct comparison, and "
        "Table 1 summarises it. **Evolution Gym** [8] is a large-scale benchmark for "
        "co-optimising the design and control of soft robots, in which a body is a 2D "
        "lattice of material cells simulated as a mass-spring system. **ARIEL** "
        "(https://ci-group.github.io/ariel/), the successor to the Revolve simulator [9], "
        "covers genome encoding, evolution, MuJoCo simulation and result analysis for "
        "modular robots. **JBotEvolver** [10] is a long-established Java platform for "
        "evolutionary robotics research and teaching, built around a 2D differential-drive "
        "engine.",
    )
    p(
        doc,
        "All three are strong at what they set out to do, and ForgeCraft integrates the "
        "same class of components rather than reimplementing them. The relevant difference "
        "is the terminal stage: in the released feature sets of all three, the pipeline ends "
        "once a simulation result has been produced and logged, and none describes a route "
        "from an evolved body to manufacturing deliverables. For Evolution Gym this follows "
        "from the representation itself — a 2D voxel matrix has no corresponding fabrication "
        "output. For ARIEL and JBotEvolver, whose robots are built from physical modules or "
        "wheeled bases, the morphologies are realisable in hardware by construction, but the "
        "mapping from an evolved graph to machine-readable part geometry, drawings and bills "
        "of materials is not part of the framework.",
    )
    p(
        doc,
        "Design for additive manufacturing [11] and design for manufacture and assembly "
        "[12] supply the criteria a buildable part must satisfy — minimum feature size, "
        "wall thickness, build-volume fit, support and orientation, assembly complexity — "
        "and product data exchange between CAD systems is standardised in ISO 10303, whose "
        "AP242 part [13] covers managed model-based 3D engineering, with general "
        "tolerances given by ISO 2768-1 [14]. This body of work is mature, but it does not "
        "address a design generated by a search free to exploit any unmodelled degree of "
        "freedom — precisely the case ForgeCraft is built for.",
    )

    table(
        doc,
        ["Platform", "Morphology representation", "Co-design search", "Physics", "Manufacturing deliverables", "Reference"],
        [
            [
                "Evolution Gym",
                "2D lattice of material voxels (soft / rigid / actuator)",
                "Yes — design and control co-optimisation; task suite and baseline algorithms",
                "Own C++ mass-spring engine",
                "None in the released benchmark",
                "Bhatia et al. [8]",
            ],
            [
                "ARIEL (formerly Revolve)",
                "Modular robot graph decoded from NDE / CPPN / tree / vector genotypes",
                "Yes — body–brain co-design with EA operators",
                "MuJoCo",
                "None in the published feature set",
                "[9]",
            ],
            [
                "JBotEvolver",
                "2D differential-drive kinematics; controller evolution on a fixed or scripted body",
                "Controller evolution; body variation is not the focus",
                "Own 2D engine",
                "None",
                "[10]",
            ],
            [
                "**ForgeCraft**",
                "Typed graph of parametric parts drawn from declarative YAML catalogues",
                "Yes — NSGA-II/III and MAP-Elites over bodies; PPO controller conditioned on a GNN body embedding",
                "MuJoCo; optional Genesis backend",
                "**Tessellated STEP AP242 assembly, tolerance-annotated DXF drawings, interference report, bill of materials, manufacturability score, production archive**",
                "This paper",
            ],
        ],
    )
    caption(
        doc,
        "**Table 1** Feature comparison against three widely used evolutionary-robotics "
        "platforms. The manufacturing column records what each project's released "
        "documentation and repository describe at the time of writing; it does not rule "
        "out unpublished extensions. Rows are ordered by breadth of published scope.",
    )

    # ---- 3 ---------------------------------------------------------------- #
    h1(doc, "3 System overview")
    p(
        doc,
        "ForgeCraft is a layered Python package of 192 modules (approximately 60,000 "
        "lines) with a deliberately narrow interface between layers, so that individual "
        "research questions can be addressed without modifying the rest of the stack.",
    )

    h2(doc, "3.1 Core morphology representation")
    p(
        doc,
        "The central data structure is `MechanicalBody`, a directed graph of `Part` and "
        "`Joint` objects. Parts carry a type and a parameter dictionary, so a body is a "
        "typed graph rather than a fixed topology; the representation is memory-conscious "
        "by design (`__slots__` on both classes) because populations of thousands of "
        "bodies are held in memory during search. Parts are drawn from a *catalogue*: a "
        "declarative YAML description of the available part types with their geometric "
        "primitive, parameter ranges, mass, friction and actuation properties. Because the "
        "catalogue is data rather than code, a new design space is a configuration file "
        "rather than a fork; 21 catalogues ship with the release, spanning primitive "
        "parts, legged robots, grippers, hoppers and parameterised part libraries.",
    )

    h2(doc, "3.2 Physics evaluation")
    p(
        doc,
        "Evolved bodies are compiled to MuJoCo MJCF models for rigid-body simulation, "
        "with an optional Genesis backend for GPU-accelerated evaluation. To keep search "
        "tractable, the evaluation path uses deliberately simplified collision geometry "
        "(convex proxies), which decouples the cost of physics evaluation from the cost of "
        "geometric detail. Environments follow the Gymnasium API [15], and task "
        "specifications — reward composition, episode budget, termination criteria — are "
        "themselves declarative YAML.",
    )
    p(
        doc,
        "One modelling detail determines whether the reward terms carry any signal at "
        "all: the MJCF builder attaches a `<freejoint/>` to the root body, so every "
        "compiled model has a full six-degree-of-freedom floating base. An earlier revision "
        "omitted that element and welded the root to the world frame, which made the "
        "displacement-, speed- and fall-based reward terms identically zero. Section 6 "
        "records the measured before-and-after effect of the repair.",
    )

    h2(doc, "3.3 Search")
    p(
        doc,
        "Co-evolution alternates between population-level search over morphologies and "
        "reinforcement-learning updates of the controllers conditioned on those "
        "morphologies. Morphologies are encoded by a message-passing graph neural network "
        "[16, 17] into fixed-length embeddings, and controllers are PPO [18] actors "
        "conditioned on those embeddings. Conditioning the policy on a morphology "
        "embedding allows a controller to be trained for an arbitrary topology, but "
        "ForgeCraft trains a *separate* controller per evaluated individual within that "
        "individual's evaluation budget, optionally warm-started from the previous "
        "generation: it does not learn a single policy that generalises zero-shot across "
        "body topologies, and we make no such claim. Selection combines elitism, "
        "tournament and Pareto-based multi-objective selection (NSGA-II/III [19, 20]) "
        "while a MAP-Elites archive [6] maintains behavioural diversity. Search is made "
        "affordable by adaptive evaluation budgets that grow with the generation index, "
        "and runs support checkpoint and resume.",
    )

    h2(doc, "3.4 Geometry and manufacturing layer")
    p(
        doc,
        "A parametric generator produces concrete mechanical geometry for each abstract "
        "part — chassis with ribs and counterbores, motors with cooling fins and end "
        "bells, bearings with shields, tubes with threaded ends, springs, feet and battery "
        "packs — from the same parameter dictionary used during search. Nine classes of "
        "part are generated at four vertex-density presets, and parts receive physically "
        "based rendering materials from ten presets. Because search-time bodies and "
        "export-time geometry share parameters, the exported artefacts correspond to the "
        "design that was actually evaluated. Optional modules provide finite-element and "
        "isogeometric analysis, additive-manufacturing pre-processing, knowledge storage "
        "and LLM-assisted design agents, isolated behind optional installation extras so "
        "that the core pipeline keeps a small dependency footprint; Section 8 records that "
        "these analysis modules are incomplete and unused in every result reported here.",
    )

    # ---- 4 ---------------------------------------------------------------- #
    h1(doc, "4 The manufacturing-aware export chain")
    p(
        doc,
        "The export chain is the contribution claimed in this paper. A single entry "
        "point, `export_all_enhanced`, takes an evolved body plus the catalogue it was "
        "drawn from and performs the following stages end to end:",
    )
    numbered(
        doc,
        [
            "**Parametric solid generation** for every part in the evolved body.",
            "**Connection features** — bolt holes, hinge bearing seats and snap fits "
            "generated from the joint type, with clearances derived from FDM tolerance "
            "profiles.",
            "**Tolerance and shrinkage compensation** for the selected print profile.",
            "**Support and orientation planning**, with support-volume reporting.",
            "**Interference checking** between mating part pairs, classifying each fit as "
            "clearance, transition or interference.",
            "**STEP AP242 assembly export** for CAD interchange, subject to the "
            "qualification below.",
            "**2D engineering drawings** in DXF with tolerance annotation.",
            "**Bill of materials** with mass roll-up and material recommendations.",
            "**Manufacturability scoring**, decomposed into build-volume, wall-thickness, "
            "assembly-complexity, torque-budget and cantilever checks.",
            "**Production packaging** into one archive that can be handed to a "
            "fabricator, together with a URDF for downstream robotics tooling and an "
            "optional ONNX export of the trained policy.",
        ],
    )
    p(
        doc,
        "Two properties matter more than the stage list. First, the manufacturability score "
        "described in stage 9 is wired into the search layer — a manufacturability term and "
        "a minimum-score constraint are both available per task — and Section 7 measures "
        "what that wiring currently achieves. Second, the STEP output carries a "
        "qualification we will not bury: the exported entities are tessellated "
        "(`TESSELLATED_ITEM` with `COORDINATES_LIST` and `TRIANGULATED_FACE`, under the "
        "`AP242_MANAGED_MODEL_BASED_3D_ENGINEERING_MIM_LF` schema). The file is valid AP242 "
        "and readable by tessellation-aware viewers, but it carries triangle meshes rather "
        "than analytic NURBS surfaces and topological B-rep edges, so it cannot be edited "
        "parametrically on import. A fabricator receiving this package would typically "
        "rebuild parametric features from the DXF drawings and the bill of materials rather "
        "than from the solid. Closing that gap is future work.",
    )

    # ---- 5 ---------------------------------------------------------------- #
    h1(doc, "5 Worked example: one complete run")
    p(
        doc,
        "The workflow is exercised by an example committed to the repository "
        "(`examples/demo_robot/`), recording the artefacts of a finished evolution run "
        "(generation 89, individual 005), started with a single command:",
    )
    p(
        doc,
        "`python -m forgecraft.main --catalog speedster --task speed "
        "--population 32 --generations 100 --seed 42 --export --export-dir design_output`",
        indent=True,
    )
    p(
        doc,
        "The CLI `--export` flag drives the lightweight export path (STL, STEP, 3MF, "
        "URDF, BOM and a review report). The fuller presentation chain, which additionally "
        "produces DXF drawings, glTF scenes, rendered views and the ZIP production "
        "archive, is `export_all_enhanced` and is reached through the Python API.",
    )
    p(
        doc,
        "The recorded design comprises 25 parts and 24 joints with a stored composite "
        "fitness of 12954.87, dominated by movement terms (`displacement` = 10452.36, "
        "`speed` = 2497.80, `upright` = 46.41). The body has eight actuated joints, so the "
        "review report no longer emits the \"no actuated joints, the body cannot move "
        "under its own power\" warning that the previous snapshot carried. We nevertheless "
        "report the manufacturing diagnostics in full, because they say more about the "
        "state of the release than the fitness value does (Table 2, Fig. 1).",
    )

    table(
        doc,
        ["Quantity", "Value", "Where it comes from"],
        [
            ["Parts / joints", "25 / 24", "`best_body.json`"],
            ["Composite fitness", "12954.87", "`evolution_history.json`"],
            ["Movement components", "`speed` 2497.80; `displacement` 10452.36; `upright` 46.41", "`evolution_history.json`"],
            ["Actuated joints", "8", "`best_body.json`"],
            ["Colliding part pairs", "120", "interference check in the review report"],
            [
                "Implausible joint ranges",
                "11 total: 6 collapsed to [0, 0]; 5 spanning more than 4π rad (widest 888.18 rad)",
                "joint-range warnings in the review report",
            ],
            ["Manufacturability aggregates", "`overall` 0.9925 and `manufacturability_score` 0.70", "`manufacturability.json`"],
            [
                "Mass of the same body",
                "5.244 kg, 6.233 kg, 7.005 kg",
                "`bom.json`; review report; torque-budget check in `manufacturability.json`",
            ],
            [
                "Cost of the same body",
                "USD 157.31 and USD 442.11 (material 186.98 + printing 255.13)",
                "`bom.json`; review report",
            ],
            ["Print estimate", "85.0 h, 2862 g PLA", "review report"],
        ],
        font_size=8.5,
    )
    caption(
        doc,
        "**Table 2** Diagnostics for the reference run. The last four rows are not "
        "defects of the pipeline but of its bookkeeping: the same body is measured "
        "differently by different code paths, and nothing reconciles them.",
    )

    side_by_side_figure(
        doc,
        [
            RENDERS / "gen89_normal_paper_main.png",
            RENDERS / "gen89_exploded_paper_main_exploded.png",
        ],
        2.9,
        "**Fig. 1** Assembled (left) and exploded (right) views of the evolved body of "
        "Section 5, rendered by the framework's lightweight rasteriser. The 25 parametric "
        "parts shown are the ones from which the STEP assembly, DXF drawings and bill of "
        "materials are produced. The body is a prototype produced by the pipeline, not a "
        "validated design: it contains 120 colliding part pairs and eleven joints with "
        "implausible parameter ranges (Table 2).",
    )

    p(
        doc,
        "Three caveats follow, and we state them rather than omit them. First, a stored "
        "fitness of 12954.87 is **not** evidence of a high-performing robot: it is a "
        "weighted sum dominated by movement terms, reached while the body's parts "
        "interpenetrate in 120 pairs and its joints admit physically meaningless travel, "
        "neither of which the objective penalises. The honest reading is that the objective "
        "now carries a locomotion signal — which, before the repair described in Section 6, "
        "it did not — and that self-collision and joint-limit terms must be added before "
        "evolved bodies can be read as engineering designs. Second, the aggregate figures "
        "disagree: `overall` (0.9925) and `manufacturability_score` (0.70) are two distinct "
        "definitions rather than an arithmetic error (Section 7), while the mass of this "
        "single body is reported as 5.244 kg, 6.233 kg and 7.005 kg by three sites, and cost "
        "as USD 157.31 against USD 442.11. Each site derives volume and unit cost from a "
        "different formula — bounding-box extents, per-part parameters, or a separate price "
        "table — and we report all of these figures rather than the most flattering one. "
        "Third, the committed snapshot is a stored artefact: STL, STEP, DXF, URDF and the "
        "ZIP archive regenerate from `best_body.json` and are deliberately not committed, to "
        "keep the repository small.",
    )
    p(
        doc,
        "We therefore present this example as evidence for three narrow claims only: that "
        "the pipeline runs end to end from a task specification to a production package, "
        "that evolutionary search now receives a locomotion signal, and that the export "
        "chain emits inspectable, machine-readable deliverables. It is not evidence that "
        "the framework discovered a high-performing robot.",
    )

    # ---- 6 ---------------------------------------------------------------- #
    h1(doc, "6 Verification and reproducibility")
    p(
        doc,
        "The repository contains 424 automated tests — 253 in the core suite and 171 "
        "module-level — covering morphology graph invariants, catalogue loading, genetic "
        "operators, selection schemes, checkpoint round-trip, physics model construction, "
        "geometry generation, tolerance and connector calculation, interference checking "
        "and export formats. Continuous integration runs the suite on Python 3.10–3.12 at "
        "every commit, skipping GPU-dependent tests when no CUDA device is present. Runs "
        "are seeded, task and catalogue definitions are declarative, and checkpoints "
        "capture both evolutionary and training state, so an interrupted run resumes to an "
        "identical configuration.",
    )
    p(
        doc,
        "The most consequential defect found while preparing this release illustrates why "
        "a green test suite is not sufficient evidence. In the previous revision, "
        "`forgecraft/simulation/builder.py` emitted the root body without a "
        "`<freejoint/>`, so a six-part, five-joint body compiled with no floating base: the "
        "centre-of-mass sensor was constant, termination criteria of the form "
        "`_com_z < fall_height` never fired, and the `displacement`, `speed` and `velocity` "
        "reward terms were identically zero, removing the locomotion signal from the "
        "objective altogether. The repair is verified by measurement rather than assertion: "
        "the same body now compiles to `nq = 9, nv = 8`, with the free joint at "
        "`qposadr = 0` and hinges at `qposadr = 7` and `8`; in a 300-step zero-action "
        "rollout the centre of mass descends from 0.476 m to 0.191 m, translates 0.064 m "
        "along x and fires the fall termination at step 299 — all exactly zero before the "
        "repair. Two latent Gymnasium-contract errors in `forgecraft/rl/env.py` were exposed "
        "by the same investigation and are fixed with it: `step()` returned `np.bool_` "
        "rather than `bool`, and `reset()` perturbed `qpos` by joint index, which with a "
        "floating base corrupts the base position and quaternion.",
    )

    # ---- 7 ---------------------------------------------------------------- #
    h1(doc, "7 What the manufacturing layer actually contributes")

    p(
        doc,
        "The framework's stated design goal is that buildability is optimised rather than "
        "filtered. We tested that claim twice. The first measurement, against the release as "
        "shipped (v0.4.0), returned a clean negative. Investigating why it returned a negative "
        "exposed two instrumentation defects, which we report and repair in this section. The "
        "second measurement, against the repaired instrument, replaces the negative with a "
        "quantitative criterion for when a manufacturability term can influence selection at all "
        "— and with an account of the experimental budget at which that influence stops being "
        "observable.",
    )

    h2(doc, "7.1 First measurement: the term does not change selection")
    p(
        doc,
        "**Method.** Comparing two independently evolved arms is unreliable in this codebase: "
        "evaluation runs in worker processes, and on Windows those workers are spawned rather "
        "than forked, so they do not inherit the parent seed and two same-seed arms are not "
        "strictly paired. We therefore measured *within* a single run. Each individual carries a "
        "fitness that includes the manufacturability term and a directly measurable value of "
        "that term, so the counterfactual fitness obtained by removing it is "
        "`fitness_without = fitness_with − w_mfg · mfg`, with no re-running required. The "
        "effective weight is `w_mfg = 0.05`: the task file lists `manufacturability` among its "
        "fitness components but declares no matching reward component, so `get_weight` returns "
        "zero and the aggregator's fallback applies (defect D2 in Section 7.2).",
    )
    p(
        doc,
        "**The term does not change selection.** Across 20 generations and a population of 16, "
        "removing the term never reverses the ordering of any pair of individuals, and the set "
        "that selection acts on is unchanged (Table 3):",
    )
    table(
        doc,
        ["Catalogue", "σ(fitness)", "mfg range Δmfg", "w·Δmfg/σ",
         "Kendall τ_b", "discordant pairs", "top-5 changes", "top-10 changes"],
        [
            ["`primitives`", "1.2737", "0.2536", "0.010", "0.8845", "0 / 120", "0", "0"],
            ["`parameterized_parts_v7`", "0.1637", "0.4322", "0.132", "0.9399", "0 / 120", "0", "0"],
        ],
        font_size=8.5,
    )
    caption(
        doc,
        "**Table 3** Counterfactual removal of the manufacturability term from the objective, "
        "measured within a single 20-generation run (population 16, seed 42, speed task). A "
        "discordant pair is a pair of individuals whose order is genuinely reversed; order "
        "changes inside a tie are not counted, because they cannot change which individuals "
        "selection keeps. The two catalogues differ by a factor of 7.8 in fitness spread and "
        "agree that at w = 0.05 the term changes nothing that selection acts on. The term spans "
        "at most 13% of one fitness standard deviation.",
    )
    p(
        doc,
        "A naive rank statistic reads this differently: counting positions after re-sorting gives "
        "2 of 16 individuals moved in `primitives` and 8 of 16 in `parameterized_parts_v7` under "
        "the unrepaired release. Inspection of the sorted fitness values shows why those counts "
        "are misleading. The final populations contain only nine distinct fitness values, and "
        "31–38% of individuals sit at exactly 0.0, so a large share of \"rank changes\" are "
        "permutations inside a tie — permutations that selection resolves by identifier, not by "
        "score. Kendall's τ-b and the discordant-pair count, both of which exclude ties, are the "
        "quantities that answer the question actually being asked, and both are zero. We report "
        "this because the tie structure is itself an independent property of the release: a "
        "population in which a third of individuals share the score 0.0 has limited resolution "
        "for any added objective term, however well designed.",
    )
    h2(doc, "7.2 Two instrumentation defects")

    p(
        doc,
        "A null result invites the question of whether the instrument was working. It was not, "
        "in two independent ways.",
    )
    p(
        doc,
        "**D1 — the scoring function had no gradient.** The five checks (build volume, wall "
        "thickness, assembly complexity, torque budget, cantilever) were each written as a "
        "threshold test with a capped penalty. A part either passed, losing nothing, or failed, "
        "losing a fixed amount, so the theoretical range of the composite score was [0.80, 1.00] "
        "and in practice almost nothing triggered. Sampling 100 randomly generated bodies from "
        "each of the 21 shipped catalogues, the score was *identically constant* for 7 of the 21 — "
        "no body differed from any other body. A further defect sat inside the torque check: when "
        "a body had no actuated joints the check returned 1.0, so a morphology that cannot move "
        "under its own power received full marks for its torque budget.",
    )
    p(
        doc,
        "**D2 — the weight was never connected.** `aggregate_fitness_generic` obtained the "
        "manufacturability weight through `get_weight`, which reads `reward.components` — the "
        "namespace that configures the reinforcement-learning reward. The task files declared "
        "`manufacturability` in `fitness_components` and `fitness_formula` instead, and "
        "`fitness_formula` was used only to compose a description string. The lookup therefore "
        "always missed, and the aggregator silently fell back to a hard-coded 0.05. The weights "
        "written in the task files — 0.03 for `multi_objective`, 0.01 for `speed_density` — never "
        "took effect. This is the more consequential of the two defects, because it is invisible: "
        "a declared weight that is silently replaced by a default produces a run that looks "
        "correct and is not.",
    )

    h2(doc, "7.3 Repair")
    p(
        doc,
        "Both defects are fixed in the release accompanying this paper.",
    )
    bullet(
        doc,
        "All five checks are now graded rather than binary: a design threshold maps to a score of "
        "0.5, with headroom above it and progressive loss below it. The torque check reports "
        "*not applicable* for bodies without hinge joints and is dropped from the weighted sum, "
        "which is then renormalised over the applicable checks; a body with actuated joints and "
        "zero available torque now scores 0.0 rather than 1.0.",
    )
    bullet(
        doc,
        "A separate `fitness_weights` namespace is introduced, distinct from the reward "
        "configuration. The aggregator resolves weights in the order explicit fitness weight, "
        "then reward component, then fallback — and the fallback emits a warning instead of "
        "passing silently.",
    )
    bullet(
        doc,
        "The internal weights are redistributed according to the resolving power of each check: "
        "assembly complexity 0.35, torque budget 0.25, cantilever 0.15, wall thickness 0.15, "
        "build volume 0.10.",
    )
    p(
        doc,
        "The effect on the instrument is measurable. Repeating the same sampling procedure "
        "(21 catalogues × 100 bodies) gives (Table 4):",
    )
    table(
        doc,
        ["Quantity", "v0.4.0", "Repaired"],
        [
            ["Cross-catalogue mean range", "0.9760 – 1.0000", "0.8147 – 0.9838"],
            ["Mean within-catalogue range", "0.0240", "0.2254"],
            ["Catalogues with zero range", "7 / 21", "0 / 21"],
            ["`primitives` within-catalogue range", "0.0000", "0.0958"],
            ["`parameterized_parts_v7` within-catalogue range", "0.0857", "0.4627"],
        ],
        font_size=8.5,
    )
    caption(
        doc,
        "**Table 4** Dynamic range of the fitness-side manufacturability score before and after "
        "repair, measured on 100 randomly generated bodies per catalogue for all 21 shipped "
        "catalogues. The mean within-catalogue range grows by a factor of 9.4, and no catalogue "
        "is constant after the repair.",
    )
    p(
        doc,
        "A gradient, however, is not the same thing as influence, and it is worth being precise "
        "about why. The score now discriminates between bodies. Whether that discrimination "
        "reaches selection depends on how its spread compares with the spread of the rest of the "
        "objective, which the repair does not change.",
    )

    h2(doc, "7.4 A diagnostic criterion")
    p(
        doc,
        "The relevant quantity is the ratio of the term's own contribution range to the standard "
        "deviation of fitness within the population: `r = w · Δmfg / σ(fitness)`, where Δmfg is "
        "the within-population range of the score and w is the effective weight. This ratio is "
        "diagnostic: it can be evaluated on a single generation, before any long run, and it "
        "predicts whether the term can reorder individuals at all.",
    )
    p(
        doc,
        "Sweeping the weight over an order of magnitude on the repaired instrument, with the "
        "discordant-pair count recomputed from the stored fitness values, gives (Table 5):",
    )
    table(
        doc,
        ["Catalogue", "w", "r = w·Δmfg/σ", "discordant pairs", "top-5 changes", "top-10 changes"],
        [
            ["`primitives`", "0.05", "0.010", "0", "0", "0"],
            ["`primitives`", "0.2", "0.040", "1", "0", "0"],
            ["`primitives`", "1.0", "0.199", "2", "0", "0"],
            ["`primitives`", "2.0", "0.398", "2", "0", "0"],
            ["`parameterized_parts_v7`", "0.05", "0.132", "0", "0", "0"],
            ["`parameterized_parts_v7`", "0.2", "0.528", "3", "1", "0"],
            ["`parameterized_parts_v7`", "0.5", "1.320", "13", "2", "0"],
            ["`parameterized_parts_v7`", "1.0", "2.640", "27", "3", "1"],
        ],
        font_size=8.5,
    )
    caption(
        doc,
        "**Table 5** Weight sweep on the repaired instrument. `primitives` and "
        "`parameterized_parts_v7` differ by a factor of 7.8 in fitness standard deviation, so "
        "the same weight corresponds to very different diagnostic ratios. Both catalogues begin "
        "to reverse individual orderings at r ≈ 0.4–0.5: `parameterized_parts_v7` at w = 0.2 and "
        "`primitives` never within the swept range, reaching only r = 0.398 at w = 2.0. Top-5 "
        "and top-10 counts refer to set membership, not to order.",
    )
    p(
        doc,
        "The criterion we draw from this is that a manufacturability term begins to influence "
        "selection when `w · Δmfg / σ(fitness)` reaches roughly 0.4–0.5, and that below that "
        "value the term is present in the objective without being able to change which "
        "individuals survive. Two catalogues whose fitness spreads differ by nearly an order of "
        "magnitude produce the same threshold, which is what makes the ratio a criterion rather "
        "than a coincidence. It also corrects a natural but wrong inference from the repair: "
        "giving the score a gradient does not by itself make the objective consequential, because "
        "σ(fitness) is set by the task rather than by the metric's design.",
    )

    h2(doc, "7.5 Does crossing the threshold improve the population?")
    p(
        doc,
        "The criterion concerns a single generation. A separate claim would be that raising the "
        "weight above the threshold causes evolution to produce more manufacturable populations "
        "over many generations. We tested that claim directly, because it does not follow from "
        "the criterion and it is the claim a reader would actually want.",
    )
    p(
        doc,
        "Using `parameterized_parts_v7` and the speed task, we ran paired arms at w = 0.05 "
        "(baseline, r = 0.13) and w = 0.50 (treatment, r = 1.32) for 20 generations at population "
        "16, over three seeds. Each arm takes about fifteen minutes, and runs are serial because "
        "concurrent evaluation exceeds available memory. The measured quantity is the mean "
        "manufacturability of the final population (Table 6).",
    )
    table(
        doc,
        ["Seed", "Baseline mean", "Treatment mean", "Δ mean", "Δ median",
         "Baseline mfg < 0.8", "Treatment mfg < 0.8"],
        [
            ["7", "0.8147", "0.6394", "−0.1753", "−0.3024", "6 / 16", "11 / 16"],
            ["13", "0.7704", "0.7834", "+0.0130", "+0.0875", "9 / 16", "8 / 16"],
            ["42", "0.7360", "0.8049", "+0.0689", "+0.1300", "12 / 16", "8 / 16"],
            ["**Aggregate**", "—", "—", "**−0.0311**", "−0.0283", "—", "—"],
        ],
        font_size=8.5,
    )
    caption(
        doc,
        "**Table 6** Verification run at r = 1.32, above the diagnostic threshold. Two of three "
        "seeds move in the expected direction and one moves sharply against it; the aggregate is "
        "negative. The cross-seed range of the effect (0.244) is larger than the effect itself, "
        "and the baseline arm alone varies by 0.079 across seeds.",
    )
    p(
        doc,
        "**The effect is not detectable at this budget.** Three observations support that "
        "reading rather than a reading of genuine harm. First, the sign is not consistent: two "
        "seeds improve and one degrades sharply, which is what a small, noisy effect looks like "
        "rather than what a systematic penalty looks like. Second, the variability exceeds the "
        "signal: the spread of the per-seed differences (0.244) is eight times the absolute "
        "aggregate, and the baseline arm's own seed-to-seed spread of 0.079 is comparable to the "
        "treatment effect. Third, the population is small: sixteen individuals evaluated on a "
        "stochastic physics task cannot resolve a shift of a few hundredths in a mean.",
    )
    p(
        doc,
        "To state how much evidence would be needed, we ran a power calculation on the observed "
        "paired differences (two-sided α = 0.05, 80% power, observed standard deviation of the "
        "paired difference 0.128): resolving an effect of 0.10 would require 13 seeds (26 runs, "
        "about 6.5 hours), 0.07 would require 27 seeds (54 runs, about 13.5 hours), 0.05 would "
        "require 52 seeds (104 runs, about 26 hours), and 0.03 would require 143 seeds. The "
        "observed effect is in the 0.03–0.07 band. The honest conclusion is therefore not that "
        "raising the weight fails, but that establishing whether it succeeds requires between "
        "one and three days of computation per question — and that this estimate is itself "
        "optimistic, because the evaluation workers do not inherit the parent seed, so same-seed "
        "arms are not strictly paired.",
    )
    p(
        doc,
        "Reducing variance is more economical than adding seeds, and the ordering of remedies is "
        "clear from where the variance comes from: make evaluation reproducible and genuinely "
        "paired so that a seed identifies a run rather than merely a starting point; run long "
        "enough that final populations have converged, so that trajectory stochasticity stops "
        "contributing; and only then enlarge the population and add seeds.",
    )

    h2(doc, "7.6 Two different quantities share one name")
    p(
        doc,
        "`compute_manufacturability` applies the five geometric checks described above and is "
        "what the search layer consumes. `score_manufacturability` instead scores structure, "
        "materials, tolerances and cost, with the structure term driven by the number of "
        "colliding part pairs, and is what the exported report quotes. This is the origin of an "
        "inconsistency recorded in the README, where the reference body reports `overall` 0.9925 "
        "beside `manufacturability_score` 0.70 in the same file: the two numbers answer different "
        "questions. The export-side score is itself close to constant, returning exactly 0.70 for "
        "20 of the 21 catalogues.",
    )

    h2(doc, "7.7 What does work: the export layer")
    p(
        doc,
        "The defects the objective cannot see are detected downstream, and they are the norm "
        "rather than the exception. In every run reported here, all 16 individuals in the final "
        "population contain colliding part pairs, with up to 151 colliding pairs in a single "
        "body, while the fitness-side score of those same bodies remains above 0.97. The "
        "reference body of Section 5 carries 120 such pairs and 11 out-of-range joint limits. "
        "Self-collision is therefore not a tail case that a filter would catch at the end; it is "
        "the typical state of the population, and the interference report is the only component "
        "of the framework that surfaces it.",
    )

    h2(doc, "7.8 Consequence for the contribution")
    p(
        doc,
        "We therefore do not claim, in this release, that manufacturability functions as an "
        "effective first-class optimisation objective. The plumbing that would let it drive "
        "search is in place and the instrument now discriminates, but the criterion in "
        "Section 7.4 shows how far the weight must move before selection responds, and "
        "Section 7.5 shows that moving it that far buys an effect too small to measure at the "
        "reported budget. What the framework does contribute to manufacturing is the export and "
        "verification chain of Section 4: a design is carried to STEP, DXF, bill of materials "
        "and an interference report, and the defects that search ignores are reported there. "
        "Making the objective term consequential is not, as we had assumed, a matter of "
        "recalibrating the checks; it requires choosing a weight from the diagnostic ratio and "
        "then budgeting enough replication to see the effect. Both are now specified, and neither "
        "is done.",
    )

    # ---- 8 ---------------------------------------------------------------- #
    h1(doc, "8 Limitations of the current release")
    p(
        doc,
        "Every item below is reproducible from the repository, and each is mirrored in "
        "the README so that the software description and the paper cannot drift apart.",
    )
    bullet(
        doc,
        "**The reference body is a prototype, not an engineered design.** It contains 120 "
        "colliding part pairs and eleven joints with implausible parameter ranges (six "
        "collapsed to [0, 0], five spanning more than 4π rad). Self-collision penalties "
        "and physically meaningful joint limits are not yet part of the fitness function.",
    )
    bullet(
        doc,
        "**Control training is per-individual.** A separate trainer is constructed for "
        "each evaluated individual and updated within that individual's budget; no single "
        "policy is shared across bodies and no zero-shot transfer is claimed.",
    )
    bullet(
        doc,
        "**The repaired objective is under-powered at the reported budget.** The criterion in "
        "Section 7.4 can be met by raising the weight, but the population-level benefit of doing "
        "so is not detectable with 16 individuals, 20 generations and three seeds; a power "
        "calculation puts the requirement at 13 to 143 seeds per comparison. Claims about how "
        "manufacturability objectives reshape evolution must therefore wait for a cheaper "
        "evaluation path, not for a better metric.",
    )
    bullet(
        doc,
        "**Population resolution is limited by ties.** Final populations contain about nine "
        "distinct fitness values, and 31–38% of individuals score exactly 0.0. Any added "
        "objective term competes for differences that the evaluator does not currently resolve, "
        "so the diagnostic criterion of Section 7.4 is an upper bound on what a weight can "
        "achieve.",
    )
    bullet(
        doc,
        "**Manufacturability and cost aggregates coexist without reconciliation.** The "
        "two manufacturability aggregates and the three mass and two cost estimates of "
        "Table 2 come from independent code paths that nothing reconciles. The two "
        "manufacturability aggregates are distinct definitions rather than an arithmetic "
        "error (Section 7); the mass and cost discrepancies remain unexplained.",
    )
    bullet(
        doc,
        "**STEP output is tessellated.** Exports conform to AP242 but carry triangle "
        "meshes rather than analytic B-rep geometry (Section 4).",
    )
    bullet(
        doc,
        "**Degenerate joint ranges are reported, not rejected.** Range ordering is now "
        "normalised in the genetic operators and the joint optimiser, so the reference "
        "snapshot contains no inverted range; but nothing constrains how wide a range may "
        "become, and the exporter surfaces the resulting warnings rather than refusing to "
        "export, so a physically implausible limit reaches the delivered files unless the "
        "reader inspects the report.",
    )
    bullet(
        doc,
        "**The finite-element solver discretises the bounding box, not the part.** "
        "`surface_to_tetrahedralize` in `forgecraft/analysis/fea.py` derives its grid "
        "spacing from the part's bounding box and fills that box with regular volume "
        "elements; it does not tetrahedralise the part's surface geometry. Results "
        "describe a solid block of the part's outer dimensions and must not be read as a "
        "stress or safety-factor check.",
    )
    bullet(
        doc,
        "**The topology optimiser does not solve a mechanical problem.** The SIMP loop in "
        "`forgecraft/analysis/topology.py` never assembles or solves Ku = f; `spsolve` is "
        "imported but never called, and its sensitivity term depends only on the density "
        "field. The `compliance` field of `TopologyResult` is hard-coded to 0.0.",
    )
    bullet(
        doc,
        "**Rendering is not physically based.** Figures are rasterised through matplotlib "
        "with per-face Lambert/Blinn-Phong/Fresnel shading under a fixed three-light rig — "
        "adequate for schematic views, not a path tracer, and not intended for material "
        "comparisons.",
    )
    bullet(
        doc,
        "**No performance benchmarks are shipped, and distributed evaluation is not "
        "exercised in the reported results.** The repository contains no FPS or speed-up "
        "measurements; throughput figures from early documentation drafts were removed "
        "rather than replaced with estimates. The Ray and multiprocessing paths dispatch "
        "to the same rollout-and-PPO worker used in-process and therefore agree with the "
        "single-machine path by construction, but the experiments above use the in-process "
        "evaluator.",
    )
    bullet(
        doc,
        "**No evolved design has been fabricated.** The manufacturing deliverables are "
        "generated and internally checked, but no design from this framework has been "
        "printed, assembled and tested. This is the largest gap between what the framework "
        "produces and what it demonstrates, and it is the next milestone rather than a "
        "completed result.",
    )

    # ---- 9 ---------------------------------------------------------------- #
    h1(doc, "9 Discussion")
    p(
        doc,
        "**Manufacturability as an experimental variable.** The measurement in Section 7 "
        "reframes this question rather than closing it. Buildability is reported but not "
        "effectively optimised, and the reason is now specific to the point of being "
        "quantitative: a manufacturability term can only reorder candidates when its "
        "contribution range is comparable to the spread of fitness it is added to, and the "
        "weight required to reach that ratio produces a change too small to detect at the "
        "budget we can afford. Both halves of the problem are engineering decisions rather "
        "than mysteries — choose the weight from the ratio, then buy the replication the "
        "power calculation asks for, or trade replication for a lower-variance evaluator. "
        "Once that is done the interesting question becomes available: how does the structure "
        "of evolutionary search change when designs must be buildable? Treating part "
        "standardisation, minimum feature size and assembly complexity as objectives rather "
        "than filters would then turn a post-hoc constraint into an experimental factor. The "
        "framework makes this study *possible*; we have not yet carried it out, and Section 7 "
        "is an account of why the naive version of it does not work.",
    )
    p(
        doc,
        "**Reproducible comparison and a bridge between two communities.** By providing a "
        "tested, documented and permissively licensed implementation with declarative "
        "design spaces, ForgeCraft allows design spaces and tasks to be shared as "
        "configuration files, making comparisons between methods concrete rather than "
        "approximate. Its output formats target the tools mechanical engineers already use — "
        "STEP for CAD interchange, DXF for drawing review, bills of materials for "
        "procurement — so an evolved design can enter a mechanical workflow without being "
        "re-modelled from scratch, subject to the tessellation caveat above. The full "
        "pipeline also runs on commodity laptops with no proprietary software, which "
        "supports teaching co-design as an end-to-end engineering activity — from fitness "
        "function to drawing sheet — rather than as an isolated optimisation exercise.",
    )

    # ---- 10 --------------------------------------------------------------- #
    h1(doc, "10 Conclusion")
    p(
        doc,
        "ForgeCraft couples morphology–control co-evolution with a manufacturing-aware "
        "export chain, so that evolved robot designs are delivered together with the "
        "geometry, drawings, tolerances and documentation needed to build them. It "
        "integrates established components — graph-based morphology representation, "
        "GNN-encoded PPO control, quality-diversity and multi-objective search, MuJoCo "
        "physics — and adds an explicit representation of evolved bodies as parametric "
        "mechanical parts, together with an export and verification chain that reports the "
        "defects the search objective does not see. A two-round measurement reported here shows "
        "that the manufacturing term was not, as shipped, able to change selection: the scoring "
        "function had no gradient and the weight was never connected. Both are repaired, and the "
        "repaired instrument yields a quantitative criterion — a diagnostic ratio of weight and "
        "score spread against fitness spread — for when such a term can matter at all; whether "
        "meeting that criterion improves evolved populations remains an open question that this "
        "budget cannot answer. The contribution we therefore claim is the deliverable chain, its "
        "diagnostics, and the measurement method that exposed the defects, not an improvement in "
        "search. ForgeCraft is distributed under the MIT "
        "licence with 424 automated tests and continuous integration, and is intended as "
        "reusable infrastructure for research at the boundary between evolutionary robotics "
        "and design for manufacturing.",
    )
    p(
        doc,
        "Future work is ordered by dependency. First, the search objective needs the "
        "terms it currently lacks — self-collision penalties and physically meaningful "
        "joint limits — and the mass and cost models must be unified into a single "
        "reconciled source. Second, the empirical questions the framework exists to answer "
        "come into reach: how manufacturability constraints reshape evolved morphologies, "
        "and how search behaves when buildability is an objective rather than a filter. "
        "Third, and decisively, the loop must be closed in hardware: fabricating and "
        "testing an evolved design is the only evidence that will establish whether the "
        "manufacturing stage reported here is a real capability or an elaborate "
        "serialisation format. We state the scope of this release plainly: it is a "
        "research prototype whose export chain is functional and exercised end to end, and "
        "whose search chain is now free of the defect that silenced its locomotion signal "
        "but still optimises an objective that permits self-intersecting bodies. The "
        "reference example is a pipeline-completeness demonstration, not a motile design.",
    )

    # ---- declarations ----------------------------------------------------- #
    h1(doc, "Declarations")
    p(
        doc,
        "**Funding.** This research received no specific grant from any funding agency in the "
        "public, commercial or not-for-profit sectors.",
    )
    p(doc, "**Competing interests.** The author declares no competing interests.")
    p(
        doc,
        "**Author contributions.** Zhenxin Liu — Conceptualization, Methodology, "
        "Software, Validation, Formal analysis, Investigation, Data curation, "
        "Writing – original draft, Writing – review & editing, Visualization.",
    )
    p(
        doc,
        "**Data and code availability.** The framework is released under the MIT licence "
        "at https://github.com/jige0611/forgecraft. The reference run is committed in "
        "full under `examples/demo_robot/`; STL, STEP, DXF, URDF and the production "
        "archive regenerate from the committed `best_body.json`. The shipped release "
        "measured in Section 7.1 is https://github.com/jige0611/forgecraft/releases/tag/v0.4.0; "
        "the repair described in Sections 7.2–7.3 and the experiments of Sections 7.4–7.5 are "
        "on the default branch of the same repository and are not part of that tag.",
    )
    p(
        doc,
        "**Use of generative AI.** Generative AI tools — specifically the TRAE coding "
        "assistant — were used for code assistance, experiment scripting and language editing "
        "during preparation of this manuscript and the associated software. All scientific "
        "claims, results and references were verified by the author, who takes full "
        "responsibility for the content.",
    )

    # ---- references ------------------------------------------------------- #
    h1(doc, "References")
    references = [
        "Sims K (1994) Evolving virtual creatures. In: Proceedings of the 21st Annual "
        "Conference on Computer Graphics and Interactive Techniques (SIGGRAPH '94), "
        "pp 15–22. https://doi.org/10.1145/192161.192167",
        "Zhao A, Xu J, Konaković-Luković M, Hughes J, Spielberg A, Rus D, Matusik W "
        "(2020) RoboGrammar: graph grammar for terrain-optimized robot design. ACM "
        "Transactions on Graphics 39(6):188:1–188:16. "
        "https://doi.org/10.1145/3414685.3417831",
        "Gupta A, Savarese S, Ganguli S, Fei-Fei L (2021) Embodied intelligence via "
        "learning and evolution. Nature Communications 12(1):5721. "
        "https://doi.org/10.1038/s41467-021-25874-z",
        "Yuan Y, Song Y, Luo Z, Sun W, Kitani KM (2022) Transform2Act: learning a "
        "transform-and-control policy for efficient agent design. In: International "
        "Conference on Learning Representations (ICLR). arXiv:2110.03659",
        "Pathak D, Lu C, Darrell T, Isola P, Efros AA (2019) Learning to control "
        "self-assembling morphologies: a study of generalization via modularity. In: "
        "Advances in Neural Information Processing Systems 32 (NeurIPS)",
        "Mouret J-B, Clune J (2015) Illuminating search spaces by mapping elites. "
        "arXiv:1504.04909",
        "Pugh JK, Soros LB, Stanley KO (2016) Quality diversity: a new frontier for "
        "evolutionary computation. Frontiers in Robotics and AI 3:40. "
        "https://doi.org/10.3389/frobt.2016.00040",
        "Bhatia JS, Jackson H, Tian Y, Xu J, Matusik W (2021) Evolution Gym: a "
        "large-scale benchmark for evolving soft robots. In: Advances in Neural "
        "Information Processing Systems 34 (NeurIPS). arXiv:2201.09863",
        "Hupkes E, Jelisavcic M, Eiben AE (2018) Revolve: a versatile simulator for "
        "online robot evolution. In: Applications of Evolutionary Computation "
        "(EvoApplications 2018), LNCS 10784. Springer, Cham, pp 687–702. "
        "https://doi.org/10.1007/978-3-319-77538-8_46",
        "Duarte M, Silva F, Rodrigues T, Oliveira SM, Christensen AL (2014) JBotEvolver: "
        "a versatile simulation platform for evolutionary robotics. In: Proceedings of "
        "the 14th International Conference on the Synthesis and Simulation of Living "
        "Systems (ALIFE 14). MIT Press, Cambridge, MA, pp 210–211. "
        "https://doi.org/10.7551/978-0-262-32621-6-CH035",
        "Gibson I, Rosen D, Stucker B, Khorasani M (2021) Additive Manufacturing "
        "Technologies, 3rd edn. Springer, Cham. https://doi.org/10.1007/978-3-030-56127-7",
        "Boothroyd G, Dewhurst P, Knight WA (2010) Product Design for Manufacture and "
        "Assembly, 3rd edn. CRC Press, Boca Raton. "
        "https://doi.org/10.1201/9781420089288",
        "ISO (2022) ISO 10303-242:2022 Industrial automation systems and integration — "
        "product data representation and exchange — part 242: application protocol: "
        "managed model-based 3D engineering, 3rd edn. International Organization for "
        "Standardization, Geneva",
        "ISO (1989) ISO 2768-1:1989 General tolerances — part 1: tolerances for linear "
        "and angular dimensions without individual tolerance indications, 1st edn. "
        "International Organization for Standardization, Geneva",
        "Brockman G, Cheung V, Pettersson L, Schneider J, Schulman J, Tang J, Zaremba W "
        "(2016) OpenAI Gym. arXiv:1606.01540",
        "Gilmer J, Schoenholz SS, Riley PF, Vinyals O, Dahl GE (2017) Neural message "
        "passing for quantum chemistry. In: Proceedings of the 34th International "
        "Conference on Machine Learning (ICML), PMLR 70:1263–1272",
        "Veličković P, Cucurull G, Casanova A, Romero A, Liò P, Bengio Y (2018) Graph "
        "attention networks. In: International Conference on Learning Representations "
        "(ICLR). arXiv:1710.10903",
        "Schulman J, Wolski F, Dhariwal P, Radford A, Klimov O (2017) Proximal policy "
        "optimization algorithms. arXiv:1707.06347",
        "Deb K, Pratap A, Agarwal S, Meyarivan T (2002) A fast and elitist multiobjective "
        "genetic algorithm: NSGA-II. IEEE Transactions on Evolutionary Computation "
        "6(2):182–197. https://doi.org/10.1109/4235.996017",
        "Deb K, Jain H (2014) An evolutionary many-objective optimization algorithm using "
        "reference-point-based nondominated sorting approach, part I: solving problems "
        "with box constraints. IEEE Transactions on Evolutionary Computation "
        "18(4):577–601. https://doi.org/10.1109/TEVC.2013.2281535",
        "Todorov E, Erez T, Tassa Y (2012) MuJoCo: a physics engine for model-based "
        "control. In: IEEE/RSJ International Conference on Intelligent Robots and Systems "
        "(IROS), pp 5026–5033. https://doi.org/10.1109/IROS.2012.6386109",
    ]
    for idx, ref in enumerate(references, start=1):
        par = doc.add_paragraph()
        par.paragraph_format.space_after = Pt(1.5)
        par.paragraph_format.line_spacing = 0.97
        par.paragraph_format.left_indent = Inches(0.28)
        par.paragraph_format.first_line_indent = Inches(-0.28)
        add_rich(par, f"{idx}. {ref}")
        for run in par.runs:
            run.font.size = Pt(8.5)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc.save(OUT)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    build()