# specXplore prototype code

# Current Tool Outline 0.1

# To Do Stack:


**Pairwise Similarity Matrices**

- [ ] All pairs of similarities may be very computationally expensive to compute. ms2deepscore as well as cosine scores will not scale well.
    - [ ] Check the memory and computational requirements of having 3 pairwise similarity matrices of 10,000 by 10,000 stored in memory alongside node and edge lists. 
    - [ ] Implement a pipeline that limits similarity computations to a max mass defect range. Also consider block-wise construction using numpy add to file approach. If out of range immediate 0 score for all similarity measures.
    - [ ] For multiple lare pairwise similarity matrices make use of numpy mmap_mode for slicing from file.
    - [ ] for very large dictionaries consider the shove module. on disk dictionaries with key access.

- [ ] Add numeric hover data for additional scores to augmented heatmap.

- [ ] For consistency, consider coloring the nodes in the heatmap as well via x-tick coloring. Hacky solution: An axis for each color to be included. i.e. a  See https://stackoverflow.com/questions/57549944/setting-a-different-font-color-for-specific-x-axis-ticks


- [ ] Organize code into "tab-specific" modules for better code organization.
