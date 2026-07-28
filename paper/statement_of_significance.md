# Statement of Significance — Medical Image Analysis

> **MANDATORY** separate upload for MIA (Editorial Manager). Failure to provide,
> or irrelevant answers, may result in rejection. Answers pre-drafted from the
> actual experiments; **verify and complete every `[BRACKETED]` item** and upload
> as a `.doc/.docx`. Keep each answer within the stated word limit.

---

**Q1. Peer-Review: Has this paper been previously submitted to this journal or to another journal?** (≤300 words)
> [ANSWER — e.g. "No, this is the first submission." If yes, name the journal and
> detail the improvements. Note: re-submission to MedIA of papers previously
> rejected by MedIA is not permitted.]

**Q2. Has a version of the paper previously been published as a peer-reviewed paper (e.g. conference)?** (≤300 words)
> [ANSWER — e.g. "No." If yes, upload that article and outline the improvements,
> and confirm they are stated in the Introduction.]

**Q3. Provide 2–3 Associate Editors (in a different country to the authors, no conflict of interest).**
> [ANSWER — pick from the MIA editorial board; avoid prior collaborators.]

**Q4. Novelty: methodological innovation (1–2 sentences).** (≤300 words)
> We introduce *PoseGated*, a channel-wise gated fusion that injects
> doctor-annotated region-of-interest attention maps into a 3D-CNN gait
> classifier with per-joint side-head supervision, and we provide a controlled
> study of *where* and *how much* of the clinical prior to inject, showing that a
> shallow, lightly-regularised injection is optimal while all-stage fusion and the
> auxiliary losses are counter-productive.

**Q5. How is this work different from prior publications of any of the listed authors?** (≤300 words)
> Relative to our earlier gait work [Chen et al., 2023, two-stage gait CNN;
> Chen et al., 2024, PhaseMix], this paper (i) adds explicit, supervised clinical
> priors rather than unsupervised temporal fusion, and (ii) contributes the
> fusion-location/depth analysis and interpretability alignment, which are absent
> from the prior work. [Complete with full references and confirm.]

**Q6. List 3–5 published papers closest to your work (title; DOI; first/last author; journal; year), noting differences.** (≤300 words)
> [ANSWER — list the closest fusion / clinical-prior / gait-classification papers
> and, for each, one sentence on how this work differs.]

**Q7. Is the proposed method significantly better than the state of the art? How were gains quantified and verified?**
> On a three-way (ASD/DHS/LCS-HipOA) gait dataset with **patient-grouped
> 3-fold cross-validation**, PoseGated (shallow multi-[0,1]) reaches
> **94.8 ± 1.9%** video accuracy vs **90.7 ± 1.9%** for an RGB-only 3D-CNN
> baseline, and exceeds early fusion (best 93.7%), squeeze-and-excitation (92.0%)
> and QKV cross-attention (89.9%). Metrics: video accuracy and macro-F1, mean±std
> over the 3 folds. **Limitation (stated in the manuscript):** with only 3 folds
> we report mean±std and do **not** claim formal statistical significance; only the
> larger gaps (baseline vs. best; all-stage vs. shallow) should be over-interpreted.

**Q8. Does your method operate on full-resolution, full–field-of-view images?**
> The model consumes full–field-of-view monocular side-view RGB gait frames
> (uniformly temporally subsampled within a gait cycle, spatially resized to
> $224\times224$). It does not operate on cropped patches. [Confirm resize details.]

**Q9. Confirm the manuscript includes figures showing findings across the full spatial extent of the input.**
> The input is 2D video over time (not a 3D volume), so orthogonal-plane views do
> not apply. The attention-alignment case-study figure shows the model's attention
> over the full spatial frame. [Confirm once figures are added.]

**Q10. Confirm novelty (Q4–Q7) is explicitly described in the manuscript.**
> Confirmed — stated in the Introduction (contributions) and Section 5–6
> (Results/Discussion). [Confirm after final edits.]

**Q11. Reproducibility.**
> - *Multiple datasets / at least one public?* No — a single institutional dataset
>   is used; it is **not** public due to ethical/privacy constraints on patient
>   video (justified in the Data availability statement).
> - *Stratified cross-validation / external test?* Yes — patient-grouped
>   **StratifiedGroupKFold** (3 folds), stratified by class and grouped by patient.
> - *Data leakage?* No — grouping guarantees that **all data of a given subject is
>   confined to a single fold**; no patient appears in both training and test.

**Q12. For each dataset: public?, sample size, all samples used?, train/test counts.**
> One dataset (not public). Three diagnostic classes (ASD/DHS/LCS-HipOA);
> [~81] patients; videos segmented into gait-cycle chunks (~3,100 chunks per fold).
> Evaluation is 3-fold cross-validation (no held-out separate test set); per fold
> approximately [2,080] training and [1,050] test chunks. Training uses class-
> balancing over-sampling; test folds are not over-sampled. [Verify exact counts.]

**Q13. Potential biases in the data and how they are accounted for.**
> Class imbalance (addressed by training-split over-sampling; test folds unaltered);
> single-institution acquisition and a single side-view camera (limits
> generalisability — stated as a limitation); modest patient count (mitigated by
> patient-grouped CV to avoid identity leakage). [Add demographic/sex-gender notes
> per the SAGER guidance if available.]

**Q14. Is the data and code publicly available? If not, justify.**
> Code: available at [REPOSITORY URL]. Data: not public — patient gait video is
> sensitive/confidential; available from the corresponding author on reasonable
> request subject to institutional approval.

**Q15. Confirm reproducibility info (Q11–Q12) is explicitly described in the manuscript.**
> Confirmed — cross-validation protocol, patient grouping, and dataset description
> are in Section 4 (Experiments). [Confirm after final edits.]

**Q16. Was the manuscript written with the help of a Large Language Model? If yes, specify which, how, and how integrity was ensured (incl. citation correctness). Confirm this is in the manuscript.**
> Yes. A large language model ([NAME/VERSION]) was used for assistance with code
> development, experiment orchestration, results aggregation, and drafting/language
> editing. All reported numbers were verified against the experiment logs, all
> references were checked by the authors, and the authors take full responsibility
> for the content. This is disclosed in the manuscript's *Declaration of generative
> AI and AI-assisted technologies* section. [Confirm tool name/version.]

**Q17. Confirm the submission does not include any patient-identifying information.**
> Confirmed — the manuscript reports aggregate metrics and attention maps only;
> no names, identifiers, or identifiable patient imagery are included. [Re-confirm
> once figures are finalised — ensure any frame overlays are de-identified.]

**Q18. Additional comments to the publication office (not part of the submission).**
> [Optional.]
