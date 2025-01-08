
[KeyPoint-Moseq](https://keypoint-moseq.readthedocs.io/en/latest/index.html)

- It uses key-points data to build a AR-HMM model where each state is modeled using a auto-regressive model 

- Syllables are predicted as states of the HMM model. 

- The model discover the number of states by giving an upper bound on number of states and setting kappa parameter which dictates the syllable duration which indirectly influences the number of states. Furthermore, we only keep syllabes showing > 0.5% frequency/usage. 

- In general, we approximately matched the number of states/clusters in VAME to the number identified by keypoint-MoSeq, ….”we first ran kp-MoSeq, noted the number of syllables, and then specified this number for each of the other algorithms.
- 

## Computational Resources required

- Grooming dataset with 2M frames can fit into a GPU (A100 non MIG config) using around 42G of VRAM and around 90G of CPU RAM. Computation takes around 3-4 hours.
- For JABS250 with around 32M frames, we used a GPU (A100 non MIG config) which consumes around 78G of VRAM and around 90G of CPU RAM. Computation takes around 12-14 hours.