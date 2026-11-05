# Cover letter — Computer-Aided Design

**To:** The Editors-in-Chief, *Computer-Aided Design* (Elsevier, ISSN 0010-4485)
**From:** Zhenxin Liu, Beijing Jiaotong University, Beijing, China
**E-mail:** liuzhenxin0611@qq.com
**Manuscript type:** Full length research paper

---

Dear Editors,

I am submitting the manuscript **"ForgeCraft: closing the gap between evolved robot
morphology and machine-readable manufacturing deliverables"** for consideration as a
full length research paper in *Computer-Aided Design*.

## Why this manuscript belongs in Computer-Aided Design

The journal asks for contributions with significant geometric, topological, spatial or
configuration design content. This manuscript makes that content its central object
rather than a detail of implementation:

- **An explicit design representation.** An evolved robot body is represented as a
  typed graph of parametric mechanical parts drawn from declarative catalogues. The
  same parameter set defines the topology of the assembly, the geometry of each part,
  and the configuration of the joints, so the artefact that was evaluated and the
  artefact that is exported are one design object rather than three unrelated
  descriptions.
- **A design-to-manufacture chain.** From that representation the framework derives
  bolt holes, bearing seats and snap fits from the joint types; applies tolerance and
  shrinkage compensation for a print profile; plans support and build orientation;
  classifies each mating fit as clearance, transition or interference; and emits a
  tessellated **STEP AP242** assembly, **tolerance-annotated DXF** drawings, a bill of
  materials and a production archive.
- **A configuration-level result.** The interference report is the component that
  surfaces the geometric defects the search objective cannot see: in every run
  reported, *all* individuals in the final population contain colliding part pairs.

The paper therefore sits on the boundary the journal already publishes in — generative
and configuration design, assembly and representation, programmable manufacturing, and
product data exchange under ISO 10303.

## What is new

The contribution is deliberately not another co-evolution algorithm. Genetic
operators, quality-diversity archives and graph-encoded policies are taken from the
existing literature. What is new is (i) evolved morphologies represented as parametric
mechanical parts sharing parameters with the bodies actually evaluated, (ii) an export
and verification chain that converts such a body into deliverables a fabricator can
consume, and (iii) a **measurement**, not an assertion: a counterfactual analysis shows
that the manufacturability term in the objective does not change selection at the
shipped weight, because the score it consumes is nearly constant across the shipped
design spaces. I report this negative result rather than hiding it, and identify the
export layer — not the objective — as the component doing useful work.

The manuscript also states its own limitations explicitly, including that the STEP
output carries triangle meshes rather than analytic B-rep geometry, that the mass and
cost of one body are reported inconsistently by three unreconciled code paths, and
that no evolved design has yet been fabricated.

## Reproducibility

ForgeCraft is released under the MIT licence with 424 automated tests and continuous
integration. The reference run is committed in full; STL, STEP, DXF, URDF and the
production archive regenerate from the committed body description. I would be glad to
provide the code and data for the journal's **Replicability Badge** review through
GRSI if the editors consider it appropriate.

## Statements

- The manuscript is original, has not been published previously, and is not under
  consideration elsewhere.
- The author has no competing interests to declare.
- Funding: This research received no specific grant from any funding agency in the
  public, commercial, or not-for-profit sectors.
- The manuscript has been prepared by a single author, who takes full responsibility
  for its content.

Thank you for considering this submission. I would be happy to provide any further
information the editorial office requires.

Yours sincerely,

Zhenxin Liu
Beijing Jiaotong University, Beijing, China
liuzhenxin0611@qq.com
ORCID: 0009-0005-0773-647X