# SSI-on-2D-Fundamental-Period
# OpenSees SSI Frame Analysis & ML Dataset Generator 🏢🌍

This repository contains a robust Python framework for conducting large-scale parametric studies on 2D steel frames subject to Soil-Structure Interaction (SSI). Built on top of **OpenSeesPy**, this tool is explicitly designed to generate high-quality, padded datasets for Machine Learning applications while automatically producing a suite of academic-quality statistical visualizations.

## Key Features

* **Massive Parametric Engine**: Generates up to 300,000 unique frame/soil combinations, efficiently saving data in chunks to prevent memory overflow.
* **SSI Modeling**: Automatically calculates and assigns soil spring stiffnesses (Vertical, Horizontal, Rotational) using Newmark & Rosenblueth formulas.
* **ML-Ready Fixed Schema**: Pads mode shape outputs to a maximum of 10 stories, ensuring a consistent tensor shape for machine learning ingestion.
* **Automated Eigenvalue Analysis**: Extracts periods, frequencies, and mode shapes for both Fixed-Base and Flexible-Base (SSI) conditions to calculate period elongation ratios.
* **Publication-Quality Visualizations**: Automatically generates 12 distinct analytical plots (e.g., Correlation Heatmaps, Veletsos-Meek Trend Verification, 3D Interaction Maps) using an embedded academic styling configuration.
* **Robust Error Handling**: Skips non-converging or mathematically unstable parameter sets and logs them with timestamps for post-analysis review.

## Prerequisites

Ensure you have Python 3.8+ installed. The script relies on the following core libraries:

```bash
pip install openseespy numpy pandas matplotlib
