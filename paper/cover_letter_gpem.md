# Cover letter — Genetic Programming and Evolvable Machines

**To:** The Editor-in-Chief, *Genetic Programming and Evolvable Machines* (Springer, ISSN 1389-2576)
**From:** Zhenxin Liu, Beijing Jiaotong University, Beijing, China
**E-mail:** liuzhenxin0611@qq.com
**Manuscript type:** Original research

---

Dear Editor,

I am submitting the manuscript **"When does a manufacturability term change what
evolution selects? Measurement, repair and a computable screening bound, with the
ForgeCraft platform"** for consideration as an original research article in *Genetic
Programming and Evolvable Machines*.

## Why this manuscript belongs in this journal

The journal's remit is the automatic evolution of software and hardware, including
variable-size genomes, morphogenesis-inspired representations and the infrastructure
that makes such evolution reproducible. The manuscript is a contribution to that remit
on both counts.

- **A methodological result for the evolutionary-computation community.** Adding a
  manufacturing term to a fitness function is a routine move, and its effectiveness is
  usually assumed. The manuscript measures it. A counterfactual ablation within single
  runs — removing the term without re-evaluating anything, so that unpaired worker seeds
  cannot contaminate the comparison — shows that the term changed nothing selection
  acted on. Investigating why exposed two instrumentation defects that are common in
  evolutionary systems and hard to see: a scoring function built from capped threshold
  tests whose entire range was [0.80, 1.00], and a weight lookup that read the wrong
  configuration namespace and silently fell back to a default.
- **A result that replaces a broken rule with a provable one.** After repair the score
  discriminates — its mean within-catalogue range rises from 0.024 to 0.225 — yet
  selection still does not respond. The manuscript proves why and turns the proof into a
  usable statistic: a pair of individuals can reverse order only if its fitness gap is
  smaller than the term's contribution, so the fraction of such "at-risk" pairs bounds
  the reordering any weight can produce, and it is computable from a single generation
  before a long run is spent. That fraction is 6–8% for one catalogue and 12–48% for
  another over the same twentyfold weight sweep, which explains why one nominal weight
  behaves differently in each. An earlier draft of this work proposed a shared threshold
  near `r ≈ 0.4–0.5`; the exact count refutes it, and the manuscript says so.
- **Evolved machines, not simulated sketches.** An evolved body is represented as a
  typed graph of parametric mechanical parts drawn from declarative YAML catalogues.
  The same parameter set defines the topology that is searched, the geometry that is
  simulated, and the geometry that is exported, so the artefact evaluated and the
  artefact delivered are one design object rather than three unrelated descriptions.
  The export chain converts such a body into a tessellated STEP AP242 assembly,
  tolerance-annotated DXF drawings, an interference report, a bill of materials and a
  production archive.

## What is new

The contribution is deliberately not another co-evolution algorithm. Genetic operators,
quality-diversity archives (MAP-Elites) and graph-encoded PPO policies are taken from
the existing literature. What is new is (i) the measurement method above — a within-run
counterfactual with a tie-aware statistic — together with the two instrumentation
defects it found; (ii) the bound and its single-generation statistic, which tell a
practitioner whether a weight is worth running at all; (iii) a replication budget of 13
to 143 seeds, obtained by power analysis, stating what evidence a population-level claim
would require; and (iv) evolved morphologies represented as parametric mechanical parts
sharing parameters with the bodies actually evaluated, with an export and verification
chain that turns such a body into deliverables a fabricator can consume.

The manuscript also states its own limitations explicitly: the STEP output carries
triangle meshes rather than analytic B-rep geometry; the mass and cost of one body are
reported inconsistently by three unreconciled code paths; the verification experiment
is under-powered at the reported budget, with a power analysis giving the replication
actually required; and no evolved design has yet been fabricated.

## Reproducibility

ForgeCraft is released under the MIT licence with 424 automated tests and continuous
integration. Runs are seeded, design spaces and tasks are declarative configuration
files, and checkpoints capture both evolutionary and training state. The reference run
is committed in full; STL, STEP, DXF, URDF and the production archive regenerate from
the committed body description.

## Statements

- The manuscript is original, has not been published previously, and is not under
  consideration elsewhere.
- The author has no competing interests to declare.
- Funding: This research received no specific grant from any funding agency in the
  public, commercial, or not-for-profit sectors.
- Use of generative AI is declared in the manuscript.
- The manuscript has been prepared by a single author, who takes full responsibility
  for its content.

Thank you for considering this submission. I would be happy to provide any further
information the editorial office requires.

Yours sincerely,

Zhenxin Liu
Beijing Jiaotong University, Beijing, China
liuzhenxin0611@qq.com
ORCID: 0009-0005-0773-647X
