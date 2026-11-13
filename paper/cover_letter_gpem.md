# Cover letter — Genetic Programming and Evolvable Machines (Springer)

**Manuscript:** *When does a manufacturability term change what evolution selects?
Measurement, repair and a computable screening bound, with the ForgeCraft platform*

**Article type:** Original Research Article
**Thematic area:** Robotics and Artificial Life

**Corresponding author:** Zhenxin Liu, Beijing Jiaotong University, Beijing, China
liuzhenxin0611@qq.com — ORCID 0009-0005-0773-647X

Dear Editors,

I am submitting the enclosed manuscript as an Original Research Article, and would be
grateful if it could be handled within the *Robotics and Artificial Life* thematic area.

The paper asks a question that arises whenever an evolutionary search is asked to satisfy a
constraint from outside the simulation: **when does a manufacturability term actually change
what selection keeps, and when is it merely present in the objective?** In this journal's
terms, the genotype is a machine — a graph of parts with shared fabrication parameters whose
phenotype is evaluated in MuJoCo and whose terminal product is a set of manufacturing
deliverables. The paper turns that loop around and audits the selection machinery itself.
Our answer is a measurement, a repair and a bound.

1. **Measurement.** We instrument the pipeline so that the marginal contribution of the
   manufacturability term can be counted rather than asserted. Counterfactual ablation is
   performed *within* single runs, so no re-evaluation is needed and unpaired worker seeds
   cannot contaminate the comparison. On the shipped release the term reverses the ordering
   of exactly zero of the 120 individual pairs in either test catalogue, and we show why: the
   score's within-population range is below the fitness gap it would have to overcome.
2. **Repair.** The score was defective rather than merely weak, and we document the defect
   and the repair: 21 of 22 catalogue entries had collapsed onto one constant, three of five
   penalised features were never exercised, and the term's weight was read from the wrong
   configuration namespace, so every run silently used a hard-coded 0.05. After repair the
   spread grows by a factor of ten and the export layer is rewritten from a fixed template to
   a parametric emitter.
3. **A computable bound and a screening protocol.** We derive an exact, weight-and-spread
   condition under which no pair of individuals can reverse: reordering is impossible whenever
   the fitness gap exceeds the weighted score gap. This makes the question decidable **before**
   a run is spent, and we turn it into a five-case screening protocol, with a reference
   implementation and a decision table, that tells a reader whether a given weight is worth a
   seed budget. We apply the protocol retrospectively to our own experiment and report that it
   would have stopped us twice.
4. **A negative result stated as a budget.** Raising the weight past the bound does produce
   per-generation reordering, but the population-level benefit is not detectable with 16
   individuals, 20 generations and the thirteen seeds a power calculation demands for a
   difference of 0.10; resolving a difference of 0.03 would take 57 seeds per arm. We state
   that as a cost rather than a verdict, and we correct an earlier claim of our own that the
   paper's data contradicts.

**Fit with the journal.** *Genetic Programming and Evolvable Machines* is where the community
that builds these machines publishes: body representation, variation operators, surrogate
scoring, and the accounting of what an evolved design actually is. The paper addresses the
part of that practice that is usually left implicit — whether a term in the objective is doing
anything — and it does so on a system whose evolved bodies are mechanically realised rather
than abstract link-and-joint diagrams. The result is negative by construction, which is the
kind of result the field needs to be able to publish: a way to decide, cheaply, whether a term
in an objective is worth a seed budget at all.

**Reproducibility.** ForgeCraft is released under the MIT licence with 424 automated tests and
continuous integration. Every number in Tables 3, 5 and 6 is regenerated from the stored run
files by two scripts in the repository (`recompute_tables.py`, `aggregate_seeds.py`); the
manuscript is built by a single script that reads the same JSON, so the text cannot drift from
the data. All seeds, weights and catalogue versions are named in the captions.

**Declarations.** The manuscript is original, has not been published elsewhere and is not under
consideration by another journal. It is single-authored, with no competing interests and no
external funding. No evolved design has been fabricated; the paper states this as its principal
limitation and confines its physical claims accordingly. Use of a generative AI coding assistant
during preparation is disclosed in the Statements and Declarations section. I am happy to
suggest reviewers or to provide any additional material.

Thank you for your time and for considering the submission.

Yours sincerely,

Zhenxin Liu
Beijing Jiaotong University
Beijing, China
liuzhenxin0611@qq.com
