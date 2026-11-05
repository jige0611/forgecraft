"""Build the Computer-Aided Design (Elsevier, ISSN 0010-4485) submission files.

CAD accepts .docx source files and expects: a research paper of no more than 7500
words excluding references; square-bracket numeric citations numbered in order of
appearance; an Elsevier-style numbered reference list; and a separate ``Highlights``
file of 3-5 bullets, each at most 85 characters including spaces.

CAD is also reviewed **double anonymously**, so Editorial Manager asks for the
manuscript and the author details as two separate uploads. This script therefore
renders three files:

``paper/cad_title_page.docx``
    Title, author, affiliation, corresponding e-mail, acknowledgements, CRediT,
    funding, competing-interest and data-availability statements. Uploaded as
    "Title page with author details".
``paper/cad_manuscript_anonymized.docx``
    Title, abstract, body, AI declaration and references, with every identifying
    string removed. Uploaded as "Manuscript without author details".
``paper/cad_forgecraft.docx``
    The two concatenated, for local record only; do not upload this one.

plus ``paper/Highlights.docx``. Everything is produced with python-docx, so no local
TeX installation is needed.

Run:
    .venv/Scripts/python.exe paper/build_cad_docx.py
"""

from __future__ import annotations

import re
from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Inches, Pt

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "paper" / "cad_forgecraft.docx"
TITLE_OUT = ROOT / "paper" / "cad_title_page.docx"
MANUSCRIPT_OUT = ROOT / "paper" / "cad_manuscript_anonymized.docx"
HIGHLIGHTS_OUT = ROOT / "paper" / "Highlights.docx"
RENDERS = ROOT / "examples" / "demo_robot" / "renders"

TITLE_TEXT = (
    "ForgeCraft: closing the gap between evolved robot morphology "
    "and machine-readable manufacturing deliverables"
)

# Double-anonymous review: none of these may survive into the anonymised manuscript.
IDENTIFYING = ("Zhenxin", "Beijing Jiaotong", "jige0611", "liuzhenxin0611", "ORCID")

# Elsevier: "include 3 to 5 bullet points (maximum 85 characters, including
# spaces, per bullet point)"
HIGHLIGHTS = [
    "Evolved bodies are exported as STEP AP242 assemblies, DXF drawings and BOMs.",
    "Bodies are represented as parametric part graphs from declarative catalogues.",
    "The manufacturability objective does not change selection at shipped weight.",
    "Self-collision occurs in every evolved individual; only the export layer reports it.",
    "Framework ships 424 tests and MIT licence; no evolved design is yet fabricated.",
]

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
    for row in rows:
        cells = tbl.add_row().cells
        for idx, value in enumerate(row):
            fill(cells[idx], value)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)
    return tbl


# --------------------------------------------------------------------------- #
# manuscript
# --------------------------------------------------------------------------- #
def build(anonymised: bool, out: Path) -> None:
    doc = new_document()

    title(doc, TITLE_TEXT)
    if not anonymised:
        centred(doc, "Zhenxin Liu¹*, ORCID: 0009-0005-0773-647X")
        centred(doc, "¹ Beijing Jiaotong University, Beijing, China", size=9.5)
        centred(doc, "* Corresponding author. E-mail: liuzhenxin0611@qq.com", size=9.5, italic=True)

    # ---- abstract -------------------------------------------------------- #
    h1(doc, "Abstract")
    p(
        doc,
        "Morphology–control co-design has produced many simulated robot designs, yet the "
        "field has no shared route from an evolved body to a buildable artefact. Evaluated "
        "morphologies are abstract link-and-joint assemblies: the part geometry, assembly "
        "configuration, tolerances, fasteners and clearances that decide whether a design "
        "can be built are absent from the search space. We present ForgeCraft, an "
        "open-source Python framework that carries an evolved morphology through to "
        "machine-readable manufacturing deliverables, with the body's topology, part "
        "geometry and assembly configuration derived from one shared parameter set. "
        "ForgeCraft couples a graph-based morphology representation assembled from "
        "declarative part catalogues with a co-evolution engine (graph-neural-network "
        "encoding, PPO controllers, NSGA-II/III and MAP-Elites search), MuJoCo physics "
        "evaluation, and a parametric geometry and manufacturing layer that emits "
        "tessellated STEP AP242 assemblies, tolerance-annotated DXF drawings, interference "
        "reports, bills of materials, a manufacturability score and a production archive. "
        "We tested the framework's central design claim — that buildability is optimised "
        "rather than filtered — and report that it does not hold at the shipped weight: a "
        "counterfactual removal of the manufacturability term leaves the top-5 and top-10 "
        "of the population unchanged, because the score it consumes is nearly constant "
        "across the shipped design spaces. What does hold is the export stage: in every run "
        "reported here, all individuals in the final population contain colliding part "
        "pairs, and the interference report is the only component that surfaces them. We "
        "describe the architecture, position it against three widely used "
        "evolutionary-robotics platforms whose published feature sets stop at simulation, "
        "and report a complete worked run with its diagnostics. Limitations are stated "
        "explicitly: the reference body is a prototype with 120 self-colliding part pairs, "
        "mass and cost come from three unreconciled code paths, and no evolved design has "
        "yet been fabricated.",
        indent=True,
    )
    p(
        doc,
        "**Keywords:** generative design · topology and assembly representation · design "
        "for additive manufacturing · STEP AP242 · manufacturability · open-source software",
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
        "transform-and-control policies [3, 4], self-assembling modular morphologies [5], "
        "and quality-diversity search over collections of designs [6, 7].",
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
        "actually evaluated, so that topology, part geometry and assembly configuration "
        "form one design object rather than three unrelated artefacts, (ii) an export chain "
        "that converts such a body into the deliverables a fabricator can consume, and "
        "(iii) a manufacturing score that is wired into the objective — together with a "
        "measurement, in Section 7, showing that "
        "at the shipped weight this term does not in fact change selection, and identifying "
        "the export layer rather than the objective as the component that does useful work. "
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
        "Evolved bodies are compiled to MuJoCo MJCF models [15] for rigid-body simulation, "
        "with an optional Genesis backend for GPU-accelerated evaluation. To keep search "
        "tractable, the evaluation path uses deliberately simplified collision geometry "
        "(convex proxies), which decouples the cost of physics evaluation from the cost of "
        "geometric detail. Environments follow the Gymnasium API [16], and task "
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
        "[17, 18] into fixed-length embeddings, and controllers are PPO [19] actors "
        "conditioned on those embeddings. Conditioning the policy on a morphology "
        "embedding allows a controller to be trained for an arbitrary topology, but "
        "ForgeCraft trains a *separate* controller per evaluated individual within that "
        "individual's evaluation budget, optionally warm-started from the previous "
        "generation: it does not learn a single policy that generalises zero-shot across "
        "body topologies, and we make no such claim. Selection combines elitism, "
        "tournament and Pareto-based multi-objective selection (NSGA-II/III [20, 21]) "
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
        "filtered. We tested that claim against the release rather than asserting it, and "
        "the measurement does not support it at the shipped weight — which changes which "
        "component of the framework we claim to be useful.",
    )
    p(
        doc,
        "**Method.** Comparing two independently evolved arms is unreliable in this "
        "codebase: evaluation runs in worker processes, and on Windows those workers are "
        "spawned rather than forked, so they do not inherit the parent seed and two "
        "same-seed arms are not strictly paired. We therefore measured *within* a single "
        "run. Each individual carries a fitness that includes the manufacturability term "
        "and a directly measurable value of that term, so the counterfactual fitness "
        "obtained by removing it is `fitness_without = fitness_with − w_mfg · mfg` with no "
        "re-running required. The effective weight is `w_mfg = 0.05`: the task file lists "
        "`manufacturability` among its fitness components but declares no matching reward "
        "component, so `get_weight` returns zero and the aggregator's fallback applies.",
    )
    p(
        doc,
        "**The term does not change selection.** Across 20 generations and a population of "
        "16, removing the term perturbs mid-table ranks but never the set of individuals "
        "that selection actually acts on (Table 3):",
    )
    table(
        doc,
        ["Catalogue", "Spearman ρ", "Kendall τ", "Ranks changed",
         "top-5 overlap", "top-10 overlap", "mfg range", "w·range/σ(fitness)"],
        [
            ["`primitives`", "0.9992", "0.9948", "2 / 16", "5 / 5", "10 / 10", "0.0186", "0.0009"],
            ["`parameterized_parts_v7`", "0.9736", "0.9319", "8 / 16", "5 / 5", "10 / 10", "0.0692", "0.0057"],
        ],
        font_size=8.5,
    )
    caption(
        doc,
        "**Table 3** Counterfactual removal of the manufacturability term from the objective, "
        "measured within a single 20-generation run (population 16, seed 42, non-quick). "
        "Rank correlation falls measurably below 1, but the top-5 and top-10 sets are "
        "identical in both catalogues and the best individual never changes; the entire "
        "span of the term is under 0.6% of the fitness standard deviation.",
    )
    p(
        doc,
        "**Why the term is inert.** The fitness-side score is close to constant. Over 21 "
        "catalogues and 100 randomly generated bodies each, the five-check function that "
        "feeds the objective ranges from 0.9708 to 1.0 in cross-catalogue mean and never "
        "falls below 0.9129 for any body observed; in the default `primitives` catalogue it "
        "is identically 1.0, so every candidate passes every check and the term contributes "
        "a constant offset that cannot reorder anything. Its most favourable dynamic range, "
        "0.071, is still an order of magnitude short of the fitness spread it is added to.",
    )
    p(
        doc,
        "**Two different quantities share one name.** `compute_manufacturability` applies "
        "five geometric checks — build volume, wall thickness, assembly complexity, torque "
        "budget, cantilever — and is what the search layer consumes. "
        "`score_manufacturability` instead scores structure, materials, tolerances and cost, "
        "with the structure term driven by the number of colliding part pairs, and is what "
        "the exported report quotes. This is the origin of an inconsistency recorded in the "
        "README, where the reference body reports `overall` 0.9925 beside "
        "`manufacturability_score` 0.70 in the same file: the two numbers answer different "
        "questions. The export-side score is itself close to constant, returning exactly "
        "0.70 for 20 of the 21 catalogues.",
    )
    p(
        doc,
        "**What does work is the export layer.** The defects the objective cannot see are "
        "detected downstream, and they are the norm rather than the exception. In every run "
        "reported here, all 16 individuals in the final population contain colliding part "
        "pairs, with up to 151 colliding pairs in a single body, while the fitness-side "
        "score of those same bodies remains above 0.97. The reference body of Section 5 "
        "carries 120 such pairs and 11 out-of-range joint limits. Self-collision is "
        "therefore not a tail case that a filter would catch at the end; it is the typical "
        "state of the population, and the interference report is the only component of the "
        "framework that surfaces it.",
    )
    p(
        doc,
        "**Consequence for the contribution.** We therefore do not claim, in this release, "
        "that manufacturability functions as a first-class optimisation objective. It "
        "appears in the objective, and the plumbing that would let it drive search is in "
        "place, but at the shipped weight and with the shipped scoring function the term is "
        "selection-inert, and the framework's actual manufacturing contribution is the "
        "export and verification chain described in Section 4: a design is carried to "
        "STEP, DXF, bill of materials and an interference report, and the defects that "
        "search ignores are reported there. Making the objective term consequential — by "
        "recalibrating the checks so that they discriminate within the shipped catalogues, "
        "and by raising the weight until it is comparable to the fitness spread — is "
        "straightforward engineering that we have not yet done, and doing it would "
        "invalidate the reference snapshot reported here.",
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
        "reframes this question rather than closing it. Buildability is currently reported "
        "but not optimised, and the reason is specific and fixable: the five checks "
        "saturate, so the term carries no information for selection to act on. Once the "
        "checks discriminate within the shipped design spaces, the interesting question "
        "becomes available — how does the structure of evolutionary search change when "
        "designs must be buildable? Treating part standardisation, minimum feature size and "
        "assembly complexity as objectives rather than filters would then turn a post-hoc "
        "constraint into an experimental factor. The framework makes this study *possible*; "
        "we have not yet carried it out, and Section 7 is an account of why the naive "
        "version of it does not work.",
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
        "defects the search objective does not see. A measurement reported here shows that "
        "the manufacturing term is present in the objective but does not currently change "
        "selection, so the contribution we claim is the deliverable chain and its "
        "diagnostics, not an improvement in search. ForgeCraft is distributed under the MIT "
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
    if not anonymised:
        h1(doc, "CRediT authorship contribution statement")
        p(
            doc,
            "**Zhenxin Liu:** Conceptualization, Methodology, Software, Validation, Formal "
            "analysis, Investigation, Data curation, Writing – original draft, Writing – review "
            "& editing, Visualization.",
        )

    h1(doc, "Declaration of competing interest")
    p(
        doc,
        "The author declares that he has no known competing financial interests or "
        "personal relationships that could have appeared to influence the work reported in "
        "this paper.",
    )

    h1(doc, "Funding")
    p(
        doc,
        "This research did not receive any specific grant from funding agencies in the "
        "public, commercial, or not-for-profit sectors.",
    )

    h1(doc, "Data availability")
    if anonymised:
        p(
            doc,
            "The framework is released under the MIT licence. The repository URL and the "
            "exact release tag used for this paper are given on the title page. The "
            "reference run is committed in full under `examples/demo_robot/`; STL, STEP, "
            "DXF, URDF and the production archive regenerate from the committed "
            "`best_body.json`.",
        )
    else:
        p(
            doc,
            "The framework is released under the MIT licence at "
            "https://github.com/jige0611/forgecraft. The reference run is committed in full "
            "under `examples/demo_robot/`; STL, STEP, DXF, URDF and the production archive "
            "regenerate from the committed `best_body.json`. The exact release used for this "
            "paper is https://github.com/jige0611/forgecraft/releases/tag/v0.4.0.",
        )

    h1(
        doc,
        "Declaration of Generative AI and AI-assisted technologies in the writing process",
    )
    p(
        doc,
        "During the preparation of this work the author used TRAE in order to draft and "
        "revise the manuscript and to assist with implementing and documenting the "
        "associated software. After using this tool, the author reviewed and edited the "
        "content as needed and takes full responsibility for the content of the "
        "publication.",
    )

    # ---- references ------------------------------------------------------- #
    h1(doc, "References")
    references = [
        "Sims K. Evolving virtual creatures. In: Proceedings of the 21st Annual "
        "Conference on Computer Graphics and Interactive Techniques (SIGGRAPH '94); 1994. "
        "p. 15–22. https://doi.org/10.1145/192161.192167",
        "Zhao A, Xu J, Konaković-Luković M, Hughes J, Spielberg A, Rus D, Matusik W. "
        "RoboGrammar: graph grammar for terrain-optimized robot design. ACM Trans Graph "
        "2020;39:188:1–188:16. https://doi.org/10.1145/3414685.3417831",
        "Gupta A, Savarese S, Ganguli S, Fei-Fei L. Embodied intelligence via learning and "
        "evolution. Nat Commun 2021;12:5721. https://doi.org/10.1038/s41467-021-25874-z",
        "Yuan Y, Song Y, Luo Z, Sun W, Kitani KM. Transform2Act: learning a "
        "transform-and-control policy for efficient agent design. In: International "
        "Conference on Learning Representations (ICLR); 2022. arXiv:2110.03659",
        "Pathak D, Lu C, Darrell T, Isola P, Efros AA. Learning to control self-assembling "
        "morphologies: a study of generalization via modularity. In: Advances in Neural "
        "Information Processing Systems 32 (NeurIPS); 2019.",
        "Mouret J-B, Clune J. Illuminating search spaces by mapping elites. arXiv "
        "preprint arXiv:1504.04909; 2015.",
        "Pugh JK, Soros LB, Stanley KO. Quality diversity: a new frontier for evolutionary "
        "computation. Front Robot AI 2016;3:40. https://doi.org/10.3389/frobt.2016.00040",
        "Bhatia JS, Jackson H, Tian Y, Xu J, Matusik W. Evolution Gym: a large-scale "
        "benchmark for evolving soft robots. In: Advances in Neural Information Processing "
        "Systems 34 (NeurIPS); 2021. arXiv:2201.09863",
        "Hupkes E, Jelisavcic M, Eiben AE. Revolve: a versatile simulator for online robot "
        "evolution. In: Applications of Evolutionary Computation (EvoApplications 2018), "
        "LNCS 10784. Cham: Springer; 2018. p. 687–702. "
        "https://doi.org/10.1007/978-3-319-77538-8_46",
        "Duarte M, Silva F, Rodrigues T, Oliveira SM, Christensen AL. JBotEvolver: a "
        "versatile simulation platform for evolutionary robotics. In: Proceedings of the "
        "14th International Conference on the Synthesis and Simulation of Living Systems "
        "(ALIFE 14). Cambridge, MA: MIT Press; 2014. p. 210–211. "
        "https://doi.org/10.7551/978-0-262-32621-6-CH035",
        "Gibson I, Rosen D, Stucker B, Khorasani M. Additive manufacturing technologies. "
        "3rd ed. Cham: Springer; 2021. https://doi.org/10.1007/978-3-030-56127-7",
        "Boothroyd G, Dewhurst P, Knight WA. Product design for manufacture and assembly. "
        "3rd ed. Boca Raton: CRC Press; 2010. https://doi.org/10.1201/9781420089288",
        "ISO. ISO 10303-242:2022 industrial automation systems and integration — product "
        "data representation and exchange — part 242: application protocol: managed "
        "model-based 3D engineering. 3rd ed. Geneva: International Organization for "
        "Standardization; 2022.",
        "ISO. ISO 2768-1:1989 general tolerances — part 1: tolerances for linear and "
        "angular dimensions without individual tolerance indications. 1st ed. Geneva: "
        "International Organization for Standardization; 1989.",
        "Todorov E, Erez T, Tassa Y. MuJoCo: a physics engine for model-based control. In: "
        "IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS); 2012. "
        "p. 5026–5033. https://doi.org/10.1109/IROS.2012.6386109",
        "Brockman G, Cheung V, Pettersson L, Schneider J, Schulman J, Tang J, Zaremba W. "
        "OpenAI Gym. arXiv preprint arXiv:1606.01540; 2016.",
        "Gilmer J, Schoenholz SS, Riley PF, Vinyals O, Dahl GE. Neural message passing for "
        "quantum chemistry. In: Proceedings of the 34th International Conference on "
        "Machine Learning (ICML), PMLR 70; 2017. p. 1263–1272.",
        "Veličković P, Cucurull G, Casanova A, Romero A, Liò P, Bengio Y. Graph attention "
        "networks. In: International Conference on Learning Representations (ICLR); 2018. "
        "arXiv:1710.10903",
        "Schulman J, Wolski F, Dhariwal P, Radford A, Klimov O. Proximal policy "
        "optimization algorithms. arXiv preprint arXiv:1707.06347; 2017.",
        "Deb K, Pratap A, Agarwal S, Meyarivan T. A fast and elitist multiobjective genetic "
        "algorithm: NSGA-II. IEEE Trans Evol Comput 2002;6:182–197. "
        "https://doi.org/10.1109/4235.996017",
        "Deb K, Jain H. An evolutionary many-objective optimization algorithm using "
        "reference-point-based nondominated sorting approach, part I: solving problems with "
        "box constraints. IEEE Trans Evol Comput 2014;18:577–601. "
        "https://doi.org/10.1109/TEVC.2013.2281535",
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

    out.parent.mkdir(parents=True, exist_ok=True)
    doc.save(out)
    print(f"wrote {out}")


def build_highlights() -> None:
    """Elsevier requires Highlights in a separate editable file, 3-5 bullets of <=85 chars."""
    if not 3 <= len(HIGHLIGHTS) <= 5:
        raise SystemExit(f"Elsevier requires 3 to 5 highlights, got {len(HIGHLIGHTS)}")
    for text in HIGHLIGHTS:
        if len(text) > 85:
            raise SystemExit(f"highlight is {len(text)} chars, over the 85 limit: {text}")

    doc = Document()
    normal = doc.styles["Normal"]
    normal.font.name = BODY_FONT
    normal.font.size = Pt(11)
    normal.element.rPr.rFonts.set(qn("w:eastAsia"), "SimSun")
    for text in HIGHLIGHTS:
        par = doc.add_paragraph(style="List Bullet")
        par.paragraph_format.space_after = Pt(6)
        par.add_run(text)

    HIGHLIGHTS_OUT.parent.mkdir(parents=True, exist_ok=True)
    doc.save(HIGHLIGHTS_OUT)
    print(f"wrote {HIGHLIGHTS_OUT} ({len(HIGHLIGHTS)} bullets, longest {max(map(len, HIGHLIGHTS))} chars)")


def verify_citations() -> None:
    """CAD numbers citations by order of appearance, so check the rendered manuscript."""
    doc = Document(OUT)
    body: list[str] = []
    for par in doc.paragraphs:
        if par.text.strip() == "References":
            break
        body.append(par.text)

    order: list[int] = []
    for match in re.finditer(r"\[(\d+(?:\s*,\s*\d+)*)\]", " ".join(body)):
        for token in re.split(r"\s*,\s*", match.group(1)):
            number = int(token)
            if number >= 1 and number not in order:
                order.append(number)

    if order != list(range(1, len(order) + 1)):
        raise SystemExit(f"citations are not numbered in order of appearance: {order}")
    print(f"citations 1-{len(order)} first appear in ascending order")


def build_title_page() -> None:
    """Editorial Manager uploads this as "Title page with author details"."""
    doc = new_document()
    title(doc, TITLE_TEXT)
    centred(doc, "Zhenxin Liu¹*, ORCID: 0009-0005-0773-647X")
    centred(doc, "¹ Beijing Jiaotong University, Beijing, China", size=9.5)
    centred(doc, "* Corresponding author. E-mail: liuzhenxin0611@qq.com", size=9.5, italic=True)

    h1(doc, "Acknowledgements")
    p(doc, "None.")

    h1(doc, "CRediT authorship contribution statement")
    p(
        doc,
        "**Zhenxin Liu:** Conceptualization, Methodology, Software, Validation, Formal "
        "analysis, Investigation, Data curation, Writing – original draft, Writing – review "
        "& editing, Visualization.",
    )

    h1(doc, "Funding")
    p(
        doc,
        "This research did not receive any specific grant from funding agencies in the "
        "public, commercial, or not-for-profit sectors.",
    )

    h1(doc, "Declaration of competing interest")
    p(
        doc,
        "The author declares that he has no known competing financial interests or "
        "personal relationships that could have appeared to influence the work reported in "
        "this paper.",
    )

    h1(doc, "Data availability")
    p(
        doc,
        "The framework is released under the MIT licence at "
        "https://github.com/jige0611/forgecraft. The reference run is committed in full "
        "under `examples/demo_robot/`; STL, STEP, DXF, URDF and the production archive "
        "regenerate from the committed `best_body.json`. The exact release used for this "
        "paper is https://github.com/jige0611/forgecraft/releases/tag/v0.4.0.",
    )

    TITLE_OUT.parent.mkdir(parents=True, exist_ok=True)
    doc.save(TITLE_OUT)
    print(f"wrote {TITLE_OUT}")


def verify_anonymity() -> None:
    """A leaked name defeats the point of the split, so fail loudly instead of silently."""
    doc = Document(MANUSCRIPT_OUT)
    text = "\n".join(par.text for par in doc.paragraphs)
    leaked = [token for token in IDENTIFYING if token in text]
    if leaked:
        raise SystemExit(f"identifying strings survived into the anonymised manuscript: {leaked}")
    print(f"anonymised manuscript free of {len(IDENTIFYING)} identifying tokens")


if __name__ == "__main__":
    build(anonymised=False, out=OUT)
    build(anonymised=True, out=MANUSCRIPT_OUT)
    build_title_page()
    build_highlights()
    verify_citations()
    verify_anonymity()