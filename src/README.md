## Workflow for KeyPoint-Moseq

### Standard Workflow that comes with the package but tailored to our lab's data/needs

- 1 Standard scripts for running the pipeline
   - `submit_project_setup.sh` --> `submit_egocentric_alignment.sh` --> `submit_create_trainset.sh` --> `submit_train.sh` --> `submit_infer_cpu.sh` --> `submit_segmentation.sh` --> `submit_motif.sh`

 - 
   1 Custom scripts for downstream analysis
     - `community_wasserstein.py` # For building a hierarchical clustering tree using the community Wasserstein distance on latent space 
     -
  

2. 
