# Cover letter — Evolutionary Intelligence (Springer)

**Manuscript:** *When does a manufacturability term change what evolution selects? Measurement, repair and a computable screening bound, with the ForgeCraft platform*

**Corresponding author:** Zhenxin Liu, Beijing Jiaotong University, Beijing, China
liuzhenxin0611@qq.com — ORCID 0009-0005-0773-647X

Dear Editor-in-Chief,

I am submitting the enclosed original research article for consideration in
*Evolutionary Intelligence*.

The paper asks a question that arises whenever a population-based search is asked to satisfy a
constraint from outside the simulation: **when does a manufacturability term actually change what
the search selects, and when is it merely present in the objective?** Our answer is a
measurement, a repair and a bound.

1. **Measurement.** We instrument a morphology–control co-design pipeline so that the marginal
   contribution of a manufacturability term can be counted rather than asserted. On the shipped
   release the term reverses the ordering of exactly zero of the 120 individual pairs in either
   test catalogue, and we show why: the score's within-population range is below the fitness gap
   it would have to overcome.
2. **Repair.** The score was defective rather than merely weak, and we document the defect and the
   repair — 21 of 22 catalogue entries had collapsed onto one constant, and three of five
   penalised features were never exercised, because a constraint no catalogue could violate was
   never activated. After repair the spread grows by a factor of ten and the export layer is
   rewritten from a fixed template to a parametric emitter.
3. **A computable bound and a screening protocol.** We derive an exact, weight-and-spread
   condition under which no pair of individuals can reverse: reordering is impossible whenever the
   fitness gap exceeds the weighted score gap. This makes the question decidable **before** a run
   is spent, and we turn it into a five-case screening protocol, with a reference implementation
   and a decision table, that tells a reader whether a given weight is worth a seed budget. We
   apply the protocol retrospectively to our own experiment and report that it would have stopped
   us twice.
4. **A negative result stated as a budget.** Raising the weight past the bound does produce
   per-generation reordering, but the population-level benefit is not detectable with 16
   individuals, 20 generations and the thirteen seeds a power calculation demands for a difference
   of 0.10; resolving a difference of 0.03 would take 143 seeds per arm. We state that as a cost,
   not as a verdict, and we correct an earlier claim of our own that the paper's own data
   contradicts.

**Fit with the journal.** The work sits where *Evolutionary Intelligence* places its emphasis:
population-based search hybridised with other artificial intelligence techniques — here
graph-neural surrogate scoring, reinforcement-learned control and genetic programming sharing one
fitness function, with the coupling between them measured instead of assumed. The contribution is
methodological and negative-by-construction, which is the kind of result the field rarely gets to
publish and routinely needs: a way to decide, cheaply, whether a term in an objective is doing
anything at all.

**Reproducibility.** ForgeCraft is released under the MIT licence. Every number in Tables 3, 5 and
6 is regenerated from the stored run files by two scripts in the repository
(`recompute_tables.py`, `aggregate_seeds.py`); the manuscript is built by a single script that
reads the same JSON, so the text cannot drift from the data. All seeds, weights and catalogue
versions are named in the captions.

**Declarations.** The manuscript is original, has not been published elsewhere and is not under
consideration by another journal. It is single-authored, with no competing interests and no
external funding. No evolved design has been fabricated; the paper states this as its principal
limitation and confines its physical claims accordingly. I am happy to suggest reviewers or to
provide any additional material.

Thank you for your time and for considering the submission.

Yours sincerely,

Zhenxin Liu
Beijing Jiaotong University
Beijing, China
liuzhenxin0611@qq.com
