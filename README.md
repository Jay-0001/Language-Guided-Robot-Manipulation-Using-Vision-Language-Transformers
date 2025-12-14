# Language-Aware-Robot-Manipulation-Using-Vision-Language-Transformers

**Author:** Jayakaran Saravanan Indira  
**Course:** CS581B – Robot Perception  
**Semester:** Fall 2025

---

## 1. Project Description

This project implements a complete perception-to-action pipeline for **Language-Guided Robot Manipulation using Vision–Language Transformers**.

Given a natural-language command referring to a colored sphere (e.g., “pick the red sphere”), the system:

- Uses a Vision–Language Transformer grounding model (**GroundingDINO**) to detect the specified object in a PyBullet simulation
- Maps the predicted bounding box to a corresponding simulated object ID using camera projection
- Executes a pick-and-place action via inverse kinematics using a **Franka Panda** arm in PyBullet

The repository contains:

- Synthetic dataset generation for grounding analysis
- Grounding experiments and evaluation scripts
- Full integration pipeline combining parsing, grounding, projection, and manipulation
- Scripts to generate the qualitative and quantitative results used in the final report

---

## 2. Hardware / Software Requirements

### Hardware

- Ubuntu 20.04 / 22.04
- NVIDIA RTX 3070 (any CUDA-capable NVIDIA GPU should work)
- 8 GB RAM minimum
- SSD storage (≥ 10 GB)

### Software and Packages

- PyTorch ≥ 1.13 (with CUDA support)
- torchvision
- transformers
- timm
- numpy
- Pillow (PIL)
- opencv-python
- pandas
- seaborn
- matplotlib
- tqdm

---

## 3. Repository Structure

- dataset/  
  Dataset generation and quantitative metrics

- Grounding/  
  Grounding experiments and utilities

- Images/  
  Sample qualitative frames

- Initial_Simulation_Attempt_Failed/  
  Archived early simulation attempts

- Integration_Pipeline/  
  End-to-end perception and manipulation pipeline

- Pybullet_Exploration/  
  Initial PyBullet environment trials

- Visual_Linguistic_Reasoning_Attempts/  
  Vision–Language reasoning experiments

- README.md

---

## 4. Running Instructions and References

This repository contains all components required to reproduce the perception-to-action pipeline.

Before running any scripts, ensure that **GroundingDINO** is correctly installed along with its pretrained weights.  
The grounding model is responsible for detecting the referenced sphere in each PyBullet scene and forms the core perception module of the system.

Installation instructions and pretrained weights for GroundingDINO are available at Reference [1].

All project-level imports are contained within the relevant subdirectories (`dataset/` and `Integration_Pipeline/`).  
The scripts should execute correctly as long as GroundingDINO and all software dependencies are installed, and the configuration paths are set correctly inside the executable file, as shown below:

if name == "main":  
CONFIG = "/home/jay/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"  
CHECKPOINT = "/home/jay/GroundingDINO/weights/groundingdino_swint_ogc.pth"

### (A) End-to-End Manipulation Pipeline

To execute the full perception-to-action pipeline:

cd Integration_Pipeline  
python end_to_end_pipeline.py

### (B) Quantitative Evaluation

To generate sphere-count–wise grounding metrics:

cd dataset  
python grounding_eval.py

To aggregate and summarize global performance metrics:

python dataset_metrics.py

---

## 5. References

[1] GroundingDINO  
IDEA Research.  
_Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection._  
https://github.com/IDEA-Research/GroundingDINO

[2] PyBullet  
Coumans, E.  
_PyBullet Physics Simulation for Robotics._  
https://pybullet.org
