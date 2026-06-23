## About ClawGym-SynData
**ClawGym-SynData** contains **13.5K executable Claw-style tasks**. It combines two synthesis routes:

- **Persona-driven synthesis**: samples user profiles, scenario categories, and atomic operations to generate realistic workspace-grounded requests.
- **Skill-grounded synthesis**: builds tasks from OpenClaw skills, using one primary skill with optional supporting skills to encourage multi-step workflows.

The task generation process covers **9 scenario categories**, **43 subcategories**, **7 operation categories**, and **26 atomic operations**. For skill-grounded synthesis, we annotate **16,837** collected skills across categories such as Data & APIs, Dev Tools, Workflows, Automation, Security, Prompts, MCP Tools, and others.

We provide ClawGym-SynData as follows:

| Data | Link |
| --- | --- |
| 13.5K Tasks | [🤗 HuggingFace](https://huggingface.co/datasets/RUC-AIBOX/ClawGym-Task ) |
| 24.5K Trajectories | [🤗 HuggingFace](https://huggingface.co/datasets/RUC-AIBOX/ClawGym-Trajectory ) |
|  |  |


## Synthesis Pipeline
We have released our task generation pipeline.