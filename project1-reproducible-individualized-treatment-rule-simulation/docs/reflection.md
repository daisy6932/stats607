# Project Reflection

The original ITR analysis was part of a larger research workspace, making the relevant scripts, dependencies, and settings difficult to identify. The main challenge was selecting a manageable experiment while preserving the original methods and clearly distinguishing the small demonstration from the full research study.

Separating code from generated data and outputs, fixing random seeds, recording dependencies, and providing one reproduction command made the project easier to maintain and reproduce. Tests check data validity, policy calculations, pipeline execution, and source integrity. A committed reference table and figure also make it easier to inspect results and compare a new run with the expected output.

In future projects, I would establish version control and the directory structure at the beginning, record dependencies as they are added, and develop tests alongside the analysis.

I spent approximately 16 hours in total: 8 hours organizing the project and configuring the environment, 5 hours automating and testing the workflow, and 3 hours on documentation and review.

AI tools assisted with code organization, automation, tests, and documentation. Verification included successful pipeline execution, passing tests, and consistent numerical results from a clean clone. The limited simulation demonstrates computational reproducibility rather than establishing which method performs best.
