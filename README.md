# HTYLLM-PG: How To Train Your LLM

This repository is dedicated to developing and investigating language modeling techniques for massively multilingual language models with 200 or more languages.

The work was carried out by a project group from the DICE group at the University of Paderborn over two semesters: SS25 and WS25/26.

## Team

- Tutors: Nikit Srivastava, Rene Speck
- Supervisor: Prof. Dr. Axel Ngonga
- Participants: Yven de Buhr, Jamil Mounzer, Joel Dag, Sashreek Nayak Dhaimodkar, Luke Friedrichs, Martin Schröder

## Project Scope

During the project group, we investigated different approaches and explored their feasibility for training scalable and extensible massively multilingual LLMs.

### Approaches explored in SS25

- Mixture-of-Experts (MoE) training: sparse expert routing to scale model capacity efficiently.
- Cross-lingual transfer fine-tuning: training related languages together to improve transfer between them.
- Joint multilingual pretraining: training a dense model across all target languages.
- Adapter techniques (LoRA, QLoRA, XLoRA): parameter-efficient adaptation for multilingual settings.

### Focus in WS25/26

- Asymmetric hierarchical LoRA adapters: hierarchical adapter-based methods with shared and language-specific components.
- Dynamic MoE: adaptive expert routing variants and large-scale multilingual models.

## Where To Find The Approaches

- `approaches/CoLA`: hierarchical multilingual adapter line (CoLA/HydraLoRA and language-aware routing).
- `approaches/adapter`: adapter-centered experiments and setup notes.
- `approaches/cross_lingual_transfer`: cross-lingual transfer fine-tuning and evaluation scripts.
- `approaches/joint_multilingual_pretraining`: dense multilingual pretraining pipeline.
- `approaches/moe`: MoE pretraining pipeline and scripts.
- `approaches/dynamic_moe`: dynamic MoE with DeepSpeed, conversion, and evaluation tooling.

## How To Get Started With The Different Approaches

For the hierarchical asymmetric LoRA approach (in `approaches/CoLA`), there is extensive documentation of the intermediate steps across data sampling, preparation, training, and evaluation in `approaches/CoLA/docs/`.

Additionally, you can use the presentations in `presentations/` to understand the overall approaches and project progress.

Most explored approaches, especially the ones we focused on in WS25/26, have their own README files to support onboarding.

For CoLA:
- `approaches/CoLA/README.md`
- `approaches/CoLA/docs/01_project_documentation.md`

For dynamic MoE:
- `approaches/dynamic_moe/README.md`

For the other approaches:
- `approaches/adapter/README.md`
- `approaches/cross_lingual_transfer/readme.md`
- `approaches/joint_multilingual_pretraining/readme.md`
- `approaches/moe/readme.md`


### Project Management And Shared Resources

We organized and tracked work mainly via GitHub issues/milestones and a Kanban board:

- GitHub issues: https://github.com/dice-group/HTYLLM-PG/issues
- Kanban board: https://kanboard.cs.uni-paderborn.de/?controller=BoardViewController&action=show&project_id=850&search=status%3Aopen

We also maintained a shared Sciebo folder for plots, documentation, literature review material, and approach-specific resources such as important training/evaluation datasets:

- Sciebo folder: https://uni-paderborn.sciebo.de/apps/files/files/850613857?dir=/HTYLLM-PG