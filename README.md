# Deep Q-Network with Knowledge Graph Constraints for Adaptive Tutoring in Early Childhood Teacher Education

This repository contains the dataset and the source code accompanying
the study of a knowledge-graph-constrained double dueling deep Q-network
(ID3QN) for adaptive tutoring in early childhood teacher education
(ECTE). The tutoring problem is modelled as a Markov decision process
over 68 curriculum knowledge points, 153 prerequisite edges and an
816-dimensional action space combining knowledge point, difficulty
level and resource type; ID3QN extends the double dueling deep Q-network
with a validity mask applied inside both the double target and the
dueling aggregation and with a breakthrough-aware priority in the
replay buffer.

## Repository layout

```
.
├── README.md                       This file
├── ECTE_Dataset/                   Deidentified interaction dataset
│   ├── README.md
│   ├── interactions.csv            127,458 learner activity records
│   ├── learners.csv                386 learners
│   ├── knowledge_points.csv        68 curriculum knowledge points
│   ├── prerequisites.csv           153 prerequisite edges with strength
│   ├── items.csv                   735 assessment items
│   ├── courses.csv                 4 core courses
│   ├── annotator_ratings.csv       6 expert annotators over 412 candidates
│   └── statistics.json             Aggregate statistics
│
└── ECTE_Code/                      Python and PyTorch implementation
    ├── README.md
    ├── requirements.txt
    ├── config.py
    ├── dataset/                    Copy of ECTE_Dataset for convenience
    ├── src/                        Data loader, MDP, ID3QN, baselines
    ├── scripts/                    Training and rendering entry points
    ├── outputs/                    Tables (CSV + Markdown) and figures (PNG)
    └── run_all.sh                  End-to-end pipeline
```

## Dataset

The `ECTE_Dataset/` directory holds the deidentified learner interaction
logs from the online learning platform of the early childhood teacher
education program at the Faculté des Sciences de l'Éducation, Mohammed V
University in Rabat, Morocco, spanning academic years 2022 to 2024. The
release contains 127,458 activity records from 386 learners across four
core courses, together with the curriculum knowledge graph (68 nodes
and 153 directed prerequisite edges with strength weights), the 735
assessment items, the six-annotator ratings of 412 candidate
prerequisite pairs and the aggregate statistics of the release.

Every file is UTF-8 encoded English. Learner, record and session
identifiers are salted SHA-256 hashes truncated to 16 hexadecimal
characters. Every timestamp carries a per-learner random offset of
±45 days. Deidentification took place under approval by the Ethics
Committee for Research at Mohammed V University in Rabat. See
`ECTE_Dataset/README.md` for the full data card and field definitions.

## Code

The `ECTE_Code/` directory contains a Python 3.9 and PyTorch 1.13
implementation of ID3QN together with the twelve baselines evaluated in
the accompanying paper: DQN, Double DQN, Dueling DQN, PER-DQN, D3QN+PER,
Rainbow, tabular Q-learning with state aggregation, CQL, BCQ, doubly
constrained offline Q-learning, CSEAL, RL-DKT, KT-greedy, SASRec,
syllabus order and a uniform random policy. The learner simulator uses
a deep knowledge-tracing LSTM for training and a context-aware attentive
knowledge-tracing model for evaluation, both fitted on the training
learners of each cross-validation fold.

Every module is directly executable. The full pipeline is invoked by

```bash
cd ECTE_Code
bash run_all.sh
```

which trains the seven value-based agents across the five folds and
five seeds, evaluates them together with the heuristic baselines,
writes the tables to `ECTE_Code/outputs/tables/` and renders the
figures to `ECTE_Code/outputs/figures/`. See `ECTE_Code/README.md` for
detailed usage.

## Method summary

The MDP has an 80-dimensional state combining a 68-dimensional
knowledge mastery vector updated by Bayesian knowledge tracing, an
11-dimensional learning preference vector updated by exponential
smoothing and a scalar cognitive-load index derived from a rolling
window of response times and error rates. The action space combines 68
knowledge points, 3 difficulty levels and 4 resource types, yielding
816 candidate actions filtered through a validity mask that requires
support in the training logs, unmastered target knowledge points and
availability of the (knowledge point, difficulty, resource) combination
in the item pool. The composite reward covers signed knowledge growth
weighted by out-degree, prerequisite satisfaction rate and cognitive
load deviation from the moderate range.

ID3QN differs from D3QN+PER by three domain adaptations:

* the double Q-learning target restricts both action selection and
  action evaluation to the valid action set of the next state,
* the dueling aggregation averages advantages over valid actions only,
* the prioritised replay priority receives a multiplicative boost for
  breakthrough transitions where mastery crosses the mastery threshold
  after at least two consecutive errors.

## Evaluation metrics

Six metrics are reported: recommendation accuracy (RA), normalised
learning gain (NLG), prerequisite violation rate (PVR), average
cumulative reward per episode (ACR), cognitive-load moderate range rate
(CLMR) and convergence episodes (CE). Details in
`ECTE_Code/src/metrics.py` and in the paper's methodology section.

## Requirements

* Python 3.9 or newer
* PyTorch 1.13 or newer
* pandas, numpy, scipy, scikit-learn, matplotlib, tqdm

Install with

```bash
cd ECTE_Code
pip install -r requirements.txt
```

Training the full protocol uses an NVIDIA GPU with at least 8 GB of
memory. Evaluation and figure rendering run on CPU.

## Citation

If you use the dataset or the code, please cite the accompanying paper
on the deep Q-network with knowledge graph constraints for adaptive
tutoring in early childhood teacher education.

## License

The dataset is released for non-commercial academic research under the
Ethics Committee approval referenced above. The source code is released
under the MIT License.
