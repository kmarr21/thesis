# Undergraduate Thesis Project: A Comparative Analysis of Two-Step Task Modeling Algorithms and Assumptions
[Completed April 26, 2021]

## Background & Motivation

A long line of literature has established the coexistence of two paradigms during the learning of choice preferences: model-free learning and model-based learning. The two-step task, a layered decision task, has often been used as a way to dissociate model-free and model-based influences as well as to find parameter predictors for compulsivity in study participants. Recent literature has established a tension between the long-validated hybrid model (Daw et al., 2011) and newer EA model (Toyama et al., 2017; 2019). This project seeks to exhaustively explore two-step task model families and assumptions in order to examine how well these models and their respective parameters capture participant compulsivity. 

## The Two-Step Task

- The two-step task is a Markov decision task with two layered decision stages followed by an outcome stage
- Each of the two first-stage actions leads to one of the second-stage states with a higher probability of 70% (a common transition) and the other state with a lower probability of 30% (a rare transition)
- Purely model-free agents will stay more often following a reward, while purely model-based agents will stay more often when a transition value matches a reward value (i.e. rewarded & common = 1:1)

## Participants and Gillan Data

- 1409 participants were used after exclusion criteria was applied
- Task consisted of 20 practice trials and 200 actual trials
- Participant data (from Gillan et al., 2016): gathered via Amazon mTurk
- Psychiatric data: participants completed 9 self-report symptom questionnaires; Obsessive-compulsive tendency was assessed using the Obsessive-Compulsive Inventory — Revised (OCI-R) (Foa et al., 2002)

## Model Types & Adjusters

During this study, we evaluated expanded types of models alongside the standard model-free and model-based algorithms for the two-step task. These models were:
1) The Hybrid model (Daw et al., 2011): contains parallel model-free and model-based learning where the influence of each learner on the first stage choice is controlled by a weighting parameter w.
2) The Eligibility Assignment (EA) model (Toyama et al., 2019): calculates and updates only a single hybrid value for each action, using model-based learning as a critic to the update function rather than a fully parallel learning process.
3) The Transition Dependent Learning Rates (TDLR) model (DaSilva & Hare 2020): uses a solely model-based structure with modified first-stage learning rates dependent on transition type. 

To each of these, we also added various combinations of the following assumptions:
1) Perseveration: assumes that actions have autocorrelations with preceding actions
2) Forgetting: assumes that the unchosen action-values are updated

## Code Structure:

Main components of the code: 
- 'notebooks' contains inner scratch-work and should be disregarded
- 'theseus' contains the main body of the work, including:
  - 'PPC Code': python fitting and testing of models
  - 'rof': additional necessary files (rewards and drifts for two-step task, helper functions, etc.)
  - 'sim_models': simulation models for each of the base types of models in python
  - 'stan_models': collection of all hierarchical bayesian models coded in Stan. Labelled in shorthand according to abbreviation of base model + any version changes and added assumptions
  - '__init__.py': initialization file
  - 'cleaned_Gillandata.csv.zip': participant data used to calibrate and compare model performance
  - 'requirements.txt': requirements file
  - setup files: 'setup.cfg' and 'setup.py'

## Relevant Citations

[1] Daw, N. D., Gershman, S. J., Seymour, B., Dayan, P., & Dolan, R. J. (2011). Model-based influences on humans' choices and striatal prediction errors. Neuron 69(6), 1204-1215.
[2] Da Silva, C. F., and Hare, T. A. (2020). Humans are primarily model-based and not model-free learners in the two-stage task. BioRxiv [preprint].
[3] Toyama, A., Katahira, K. & Ohira, H. (2019). Biases in estimating the balance between model- free and model-based learning systems due to model misspecification. Journal of Mathematical Psychology 91, 88–102.
[4] Toyama, A., Katahira, K., & Ohira, H. (2017). A simple computational algorithm of model-based choice preference. Cognition Affect Behavior Neuroscience http://dx.doi.org/10.3758/s13415- 017- 0511- 2.
[5] Zorowitz, S., Niv, Y., & Bennett, D. (2021). Inattentive responding can induce spurious associations between task behavior and symptom measures. https://doi.org/10.31234/osf.io/rynhk.
[6] Gillan, C. M., Kosinski, M., Whelan, R., Phelps, E. A., & Daw, N. D. (2016). Characterizing a psychiatric symptom dimension related to deficits in goal-directed control. Elife 5, e11305.
