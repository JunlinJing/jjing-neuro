---
layout: post
title: "Brain Connectivity Toolbox: A Practical Guide"
description: "A comprehensive tutorial on using the Brain Connectivity Toolbox (BCT) for brain network analysis, including installation, theoretical foundations, and practical applications with academic references"
date: 2025-05-20
author: Jim Jing
categories: [tutorials]
tags: [neuroimaging, connectivity, graph theory, network analysis, tutorial]
---

# Brain Connectivity Toolbox: A Practical Guide

The Brain Connectivity Toolbox (BCT) is a MATLAB toolbox for complex brain network analysis, widely used for analyzing structural and functional brain connectivity datasets. This tutorial provides a comprehensive introduction to BCT, including its theoretical foundations, installation procedures, and practical applications in neuroscience research. Designed specifically for neuroscience beginners, this guide breaks down complex concepts with detailed explanations, mathematical formulas, and practical examples.

## 1. Introduction

### 1.1 What is the Brain Connectivity Toolbox?

The Brain Connectivity Toolbox is an open-source MATLAB toolbox developed by Rubinov and Sporns (2010) for the network analysis of structural and functional brain connectivity data. It provides a comprehensive collection of functions to calculate various network measures that characterize the topology of brain networks.

#### 1.1.1 Understanding Brain Connectivity

Before diving into the toolbox itself, let's define what we mean by "brain connectivity":

- **Structural connectivity**: Physical connections between brain regions via white matter tracts (axons). These connections are typically measured using diffusion MRI and tractography techniques.
  - *How it's measured*: Diffusion-weighted MRI (dMRI) tracks the movement of water molecules along axon bundles. Tractography algorithms then reconstruct the pathways these bundles form, creating a structural connectivity map.
  - *Interpretation*: These connections represent the physical "wiring" of the brain, similar to the highways connecting cities.
  - *Limitations*: Cannot detect directionality (which neuron connects to which) and has difficulty resolving crossing fiber pathways.

- **Functional connectivity**: Statistical dependencies between the activity of different brain regions, often measured as temporal correlations using fMRI, EEG, or MEG.
  - *How it's measured*: By calculating correlations between time series of neural activity from different brain regions. For example, if two regions consistently activate and deactivate together, they have high functional connectivity.
  - *Interpretation*: Represents which regions "work together" during rest or tasks, like cities that have high business interactions regardless of the physical distance between them.
  - *Limitations*: Does not imply causality and can be affected by indirect connections (regions A and C may appear connected because both connect to region B).

- **Effective connectivity**: Causal influences that one brain region exerts over another, often assessed using methods like dynamic causal modeling or Granger causality.
  - *How it's measured*: Through statistical models that test whether activity in one region predicts subsequent activity in another region, beyond what can be predicted from the second region's past activity alone.
  - *Interpretation*: Represents directed influence, revealing which regions drive or control the activity of other regions.
  - *Limitations*: Models are often complex and require assumptions about the underlying neural mechanisms.

**Visual representation of connectivity types:**
```
Structural Connectivity:        Functional Connectivity:       Effective Connectivity:
    A-----B                         A - - - B                       A---->B
    |     |                         |       |                       |     ^
    |     |                         |       |                       |     |
    C-----D                         C - - - D                       C---->D
(Physical connections)         (Statistical correlations)      (Causal influences)
```

#### 1.1.2 Historical Context

The study of brain connectivity has evolved significantly:
- **1990s**: Focus on activation studies and individual regions
  - *Dominant approach*: Brain mapping identified specialized brain regions for specific functions (the "where" of brain function).
  - *Limitations*: Treated brain regions as isolated entities, ignoring their interactions.

- **2000s**: Shift towards studying interactions between regions
  - *Key development*: Recognition that brain functions emerge from coordinated activity across distributed networks.
  - *New methods*: Functional connectivity analysis became widespread with resting-state fMRI.

- **2010s**: Application of graph theory to understand complex network properties
  - *Paradigm shift*: The brain began to be conceptualized as a complex network with specific topological properties.
  - *Key discovery*: Identification of the brain's small-world organization, rich club hubs, and modular structure.

- **2020s**: Integration of multi-modal connectivity data and advanced computational approaches
  - *Current trends*: Machine learning approaches to connectivity, dynamic connectivity analyses, and individualized network mapping.
  - *Future directions*: Personalized connectivity fingerprints for clinical applications and integration with genetic data.

### 1.2 Why Use Graph Theory for Brain Connectivity Analysis?

Graph theory offers a powerful mathematical framework for quantitatively describing the patterns of connections between brain regions. By representing brain regions as nodes and their connections as edges, we can:

- Quantify whole-brain topological properties
- Identify key regions (hubs) and modules in brain networks
- Compare brain networks across different populations or conditions
- Correlate network properties with behavioral or clinical measures

#### 1.2.1 The Power of Graphs for Neuroscience Beginners

For newcomers to brain science, graph theory provides several advantages:

1. **Intuitive visualization**: Networks can be visually represented, making complex connectivity patterns more accessible.
   - *Example*: A connectivity matrix can be visualized as a network diagram with nodes (brain regions) and lines (connections), allowing researchers to literally "see" the network architecture.
   - *Benefit for beginners*: Provides a concrete representation of abstract connectivity data.

2. **Standardized metrics**: Well-established mathematical measures allow for objective comparisons.
   - *Example*: Two different brain states (e.g., rest vs. task) can be quantitatively compared using metrics like clustering coefficient or path length.
   - *Benefit for beginners*: Offers clear numerical values that can be compared across studies and conditions.

3. **Multi-scale analysis**: Can study properties at local, intermediate, and global levels simultaneously.
   - *Local level*: Properties of individual nodes (e.g., degree, clustering)
   - *Intermediate level*: Properties of subnetworks or modules (e.g., modularity, within/between module connectivity)
   - *Global level*: Properties of the entire network (e.g., efficiency, small-worldness)
   - *Benefit for beginners*: Provides a framework to understand brain organization at multiple scales without changing analytical approaches.

4. **Cross-disciplinary applications**: Same principles apply to many complex systems (social networks, gene interactions, etc.).
   - *Example*: The concept of "hubs" applies to both brain networks (e.g., precuneus as a central hub) and social networks (e.g., influential individuals).
   - *Benefit for beginners*: Allows leveraging intuition from everyday network examples (social networks, transportation systems) to understand brain networks.

**Real-world analogy:**
Think of graph theory as analyzing flight routes between airports:
- Airports = Nodes (brain regions)
- Flight routes = Edges (connections)
- Hub airports with many connections = Hub brain regions
- Direct flights = Short path lengths
- Regional airport clusters = Brain modules
- Network resilience = How flight networks handle airport closures

## 2. Installation and Setup

### 2.1 Download and Installation

To install the Brain Connectivity Toolbox:

1. Visit the official BCT website: [https://sites.google.com/site/bctnet/](https://sites.google.com/site/bctnet/)
2. Download the latest version of the toolbox (currently 2019-03-03)
3. Extract the downloaded file to a directory of your choice
4. Add the BCT directory to your MATLAB path:

```matlab
% Add BCT to MATLAB path
addpath('/path/to/BCT/');
% Save path for future MATLAB sessions (optional)
savepath;
```

**Explanation of each step:**

1. **Downloading**: The BCT is regularly updated with bug fixes and new functions. The latest version includes optimizations and additional functions for network analysis.

2. **Extraction**: The download comes as a compressed ZIP file. Extract it to create a folder structure like:
   ```
   /path/to/BCT/
   ├── LICENSE.txt
   ├── README.txt
   ├── categorical/
   │   └── (various .m files)
   ├── clustering_coef_bd.m
   ├── clustering_coef_bu.m
   ... (and many other .m files)
   ```

3. **Adding to path**: In MATLAB, functions are only accessible if they're in the current directory or in the MATLAB path. The `addpath` command adds the BCT directory to MATLAB's search path.

4. **Saving the path**: The `savepath` command makes the path addition permanent across MATLAB sessions. Without this, you'll need to run `addpath` every time you restart MATLAB.

#### 2.1.1 Prerequisites for Beginners

Before installing BCT, ensure you have:

- **MATLAB**: Version R2014b or later recommended (some functions may work on earlier versions)
  - *Why*: BCT is written in MATLAB and requires its computational environment to run.
  - *Alternative options*: There is a Python implementation called 'bctpy' if you prefer Python.

- **Basic MATLAB knowledge**: Understanding of matrix operations and basic programming
  - *Essential concepts*: Matrices, indexing, loops, functions
  - *Learning resources*: MATLAB's documentation, online tutorials, or courses like Coursera's "Introduction to Programming with MATLAB"

- **Statistical knowledge**: Basic understanding of statistics for interpreting results
  - *Essential concepts*: Correlation, significance testing, distributions
  - *Application*: Understanding what network measures mean and how to compare them statistically

- **RAM**: Minimum 8GB RAM for analyzing moderate-sized networks (more for large networks)
  - *Why this matters*: Network analyses can be memory-intensive, especially for large networks (e.g., voxel-level networks).
  - *Rule of thumb*: For a network with n nodes, operations like path length calculations may require memory proportional to n².

**Practical tip for beginners:**
Set up a dedicated project folder structure before starting:
```
MyBrainNetworkProject/
├── code/
│   └── (your analysis scripts)
├── data/
│   ├── raw/
│   └── processed/
├── results/
│   ├── figures/
│   └── tables/
└── BCT/
    └── (BCT files)
```

#### 2.1.2 Common Installation Issues

- **Path conflicts**: BCT function names might conflict with other toolboxes. In case of conflicts, call BCT functions using their full path or ensure BCT is higher in the MATLAB path hierarchy.
  - *Symptom*: Error messages like "Warning: Function 'clustering_coef_bu' defined multiple times..."
  - *Solution*: Use `which -all function_name` to identify conflicting paths, then adjust the path order or use the full path in your calls.

- **Missing dependencies**: Some visualizations require additional toolboxes like BrainNet Viewer.
  - *Symptom*: Errors when trying to visualize results
  - *Solution*: Install additional visualization tools like BrainNet Viewer (for 3D brain visualizations) or the MATLAB Statistics and Machine Learning Toolbox (for statistical analyses).

- **Version compatibility**: Some newer functions may not work on older MATLAB versions.
  - *Symptom*: Errors mentioning functions or syntax not recognized
  - *Solution*: Check the minimum MATLAB version requirements for specific functions and update MATLAB if needed.

### 2.2 Verifying Installation

To verify that the toolbox is properly installed, you can try running a simple function:

```matlab
% Create a random adjacency matrix
A = rand(10);
% Make the adjacency matrix symmetric
A = (A + A')/2;
% Set diagonal to zero (no self-connections)
A(1:size(A,1)+1:end) = 0;
% Calculate the clustering coefficient
C = clustering_coef_wu(A);
% Display result
disp('Clustering coefficients:');
disp(C);
```

If the code runs without errors and displays the clustering coefficients, the BCT is installed correctly.

#### 2.2.1 Step-by-Step Explanation of Verification Code

For beginners, let's break down the verification code:

1. `A = rand(10)`: Creates a 10×10 matrix with random values between 0 and 1.
   - *What this represents*: Random connection strengths between 10 brain regions
   - *Matrix visualization*:
     ```
     A = [0.81  0.91  0.13  ...;
          0.24  0.96  0.92  ...;
          0.35  0.13  0.61  ...;
          ...                  ]
     ```

2. `A = (A + A')/2`: Makes the matrix symmetric by averaging it with its transpose.
   - *Why*: Most brain networks are undirected (bidirectional connections)
   - *What changes*: The connection from region i to j equals the connection from j to i
   - *Mathematical operation*: For any position (i,j), the new value is the average of the original A(i,j) and A(j,i)

3. `A(1:size(A,1)+1:end) = 0`: Sets the diagonal elements to zero.
   - *Why*: Diagonal elements represent self-connections, which are typically excluded
   - *How it works*: The expression `1:size(A,1)+1:end` creates an index vector [1, 12, 23, ..., 100] that selects diagonal elements in a 10×10 matrix

4. `C = clustering_coef_wu(A)`: Calculates the weighted clustering coefficient for each node.
   - *What this measures*: How interconnected each node's neighbors are
   - *Output*: A vector of 10 values (one per node) between 0 and 1

5. `disp('Clustering coefficients:'); disp(C);`: Displays the results in the MATLAB command window.

#### 2.2.2 Expected Output Interpretation

The output will be a vector of 10 values, each representing the clustering coefficient for one node. These values should be between 0 and 1, where:
- Values closer to 0 indicate low clustering (neighbors not connected)
- Values closer to 1 indicate high clustering (neighbors highly interconnected)

**Example output:**
```
Clustering coefficients:
    0.3241
    0.5127
    0.2895
    0.4678
    0.3902
    0.4159
    0.3847
    0.4231
    0.3756
    0.4012
```

**How to interpret this output:**
- *Node 2 (value 0.5127)*: Has the highest clustering, meaning its neighboring nodes are well-connected among themselves, forming a tight-knit community.
- *Node 3 (value 0.2895)*: Has the lowest clustering, meaning its neighboring nodes have fewer connections among themselves.

**What this tells us about the random network:**
- The random adjacency matrix naturally creates some variability in clustering
- Even in random networks, some nodes will have higher clustering by chance
- These values serve as a baseline to compare with real brain networks, which typically show higher clustering than random networks

## 3. Theoretical Background

### 3.1 Graph Theory Fundamentals for Brain Networks

In the context of brain networks, graph theory provides a mathematical framework for analyzing complex patterns of interactions. The fundamental components include:

- **Nodes (Vertices)**: Represent brain regions or voxels
  - *In neuroscience terms*: Anatomical or functional brain regions like "amygdala," "dorsolateral prefrontal cortex," or "primary visual cortex"
  - *How nodes are defined*: Through brain parcellation (dividing the brain into regions), which can be anatomical (based on structure) or functional (based on activity patterns)
  - *Node properties*: Can include size, spatial location, tissue type, cytoarchitecture, and neurochemical composition
  
- **Edges**: Represent connections between brain regions, which can be:
  - **Structural (anatomical connections)**
    - *Physical basis*: White matter tracts containing axon bundles
    - *Edge properties*: Fiber count, tract volume, fractional anisotropy, myelin content
  - **Functional (statistical dependencies between neural activity)**
    - *Temporal basis*: Correlated time series of neural activity
    - *Edge properties*: Correlation coefficient, coherence, phase synchronization
  - **Effective (causal influences)**
    - *Directional basis*: Temporal precedence and causal influence
    - *Edge properties*: Granger causality values, transfer entropy, dynamic causal modeling parameters

- **Adjacency Matrix**: A mathematical representation of the network where each element A(i,j) represents the connection between nodes i and j

**Extended example for beginners:**

Let's consider a simple 4-node brain network with these regions:
1. Left Amygdala (LA)
2. Right Amygdala (RA)
3. Left Prefrontal Cortex (LPFC)
4. Right Prefrontal Cortex (RPFC)

The connections might be represented in an adjacency matrix:

$$
A = \begin{pmatrix}
  0 & 0.8 & 0.6 & 0.3 \\
  0.8 & 0 & 0.2 & 0.7 \\
  0.6 & 0.2 & 0 & 0.5 \\
  0.3 & 0.7 & 0.5 & 0
\end{pmatrix}
$$

This matrix can be interpreted as:
- Strong connection (0.8) between left and right amygdala
- Moderate connections (0.6, 0.7) between amygdala and ipsilateral prefrontal cortex
- Weaker connections (0.2, 0.3) across hemispheres between non-homologous regions
- Moderate connection (0.5) between left and right prefrontal cortex
- No self-connections (zeros on diagonal)

The corresponding network visualization would show thicker lines for stronger connections:

```
    (LA)---0.8---(RA)
     |  \        /  |
     |   \      /   |
    0.6   0.3  0.7  0.2
     |      \  /    |
     |       \/     |
    (LPFC)--0.5---(RPFC)
```

#### 3.1.1 Types of Networks with Neuroscientific Examples

Networks can be classified as:

- **Binary vs. Weighted**:
  - **Binary networks**: Only indicate presence/absence of connections
    - *Example*: Tractography-derived structural networks thresholded to show only whether tracts exist between regions
    - *Mathematical representation*: $a_{ij} \in \{0,1\}$
    - *Neuroscience application*: Used to study the basic topology of connections, especially when the strength measurements are unreliable
    - *Example research question*: "Does a direct anatomical pathway exist between the amygdala and ventromedial prefrontal cortex?"

  - **Weighted networks**: Quantify connection strengths
    - *Example*: fMRI functional connectivity with correlation coefficients ranging from -1 to 1
    - *Mathematical representation*: $a_{ij} \in \mathbb{R}$
    - *Neuroscience application*: Captures the variability in connection strengths, providing richer information about brain organization
    - *Example research question*: "How does the strength of functional connectivity between default mode network regions change during cognitive tasks?"

  **Practical implications for beginners:**
  - Binary networks are simpler to analyze but lose information about connection strength
  - Weighted networks preserve more information but require more complex analyses
  - Some network measures have different formulations for binary vs. weighted networks

- **Directed vs. Undirected**:
  - **Directed networks**: Specify the direction of influence between nodes
    - *Example*: Effective connectivity from Granger causality analysis showing that activity in the prefrontal cortex precedes and influences activity in the parietal cortex
    - *Mathematical property*: A is not symmetric ($a_{ij} \neq a_{ji}$)
    - *Neuroscience application*: Models information flow and causal relationships in brain activity
    - *Example research question*: "Does the hippocampus drive activity in the prefrontal cortex during memory retrieval?"
    
    **Visualization of a directed connection:**
    ```
    PFC --------→ Parietal
     ↑             |
     |             |
     |             ↓
    Visual ←------ Temporal
    ```

  - **Undirected networks**: Only indicate connections without directionality
    - *Example*: Most functional connectivity matrices from correlation analyses
    - *Mathematical property*: A is symmetric ($a_{ij} = a_{ji}$)
    - *Neuroscience application*: Models functional relationships when causal direction is unknown or bidirectional
    - *Example research question*: "Which brain regions show synchronized activity during resting state?"
    
    **Visualization of an undirected connection:**
    ```
    PFC --------  Parietal
     |             |
     |             |
     |             |
    Visual -------- Temporal
    ```

  **Key consideration for beginners:**
  - The brain likely contains bidirectional influences, but most methods only measure undirected connectivity
  - Directed measures provide more informative models but require more complex analyses and assumptions

- **Signed vs. Unsigned**:
  - **Signed networks**: Differentiate between positive and negative associations
    - *Example*: Functional connectivity with both positive correlations (synchronized activity) and negative correlations (anti-synchronized activity)
    - *Mathematical property*: $a_{ij} \in [-1,1]$
    - *Neuroscience application*: Captures both cooperative and competitive relationships between brain regions
    - *Example research question*: "Is the anticorrelation between task-positive and default mode networks altered in schizophrenia?"
    
    **Interpretation of signs in brain networks:**
    - *Positive edge (+0.7)*: Regions activate/deactivate together
    - *Negative edge (-0.6)*: When one region activates, the other tends to deactivate
    - *Zero or near-zero edge (0.1)*: No consistent relationship between regions

  - **Unsigned networks**: Consider only the magnitude of associations
    - *Example*: Structural connectivity measured as fiber density
    - *Mathematical property*: $a_{ij} \geq 0$
    - *Neuroscience application*: Used when only connection strength matters, not direction of correlation
    - *Example research question*: "How dense are the fiber connections between cortical and subcortical regions?"

  **Methodological consideration for beginners:**
  - The interpretation of negative functional connections is debated in neuroscience
  - Global signal regression (a preprocessing step) can introduce artificial negative correlations
  - Some network measures can handle signed networks, while others require conversion to unsigned networks

#### 3.1.2 Network Topologies in the Brain

Different network architectures serve different functional purposes:

- **Random Networks**: Connections distributed randomly
  - *Properties*: Low clustering, short path lengths
  - *Example*: Some pathological brain states may exhibit more random organization
  - *Mathematical model*: Erdős–Rényi random graphs where each edge has equal probability of existing
  - *Clinical relevance*: Brain networks in certain neurological conditions (e.g., advanced Alzheimer's disease) show randomization of connectivity patterns
  
  **Characteristics in brain terms:**
  - Fewer specialized processing modules
  - Inefficient local processing
  - Relatively efficient global integration
  - Low resilience to targeted attacks

- **Regular/Lattice Networks**: Structured, local connections
  - *Properties*: High clustering, long path lengths
  - *Example*: Primary sensory cortices show more regular, neighbor-to-neighbor connectivity patterns
  - *Mathematical model*: K-nearest neighbor graphs where each node connects only to its spatial neighbors
  - *Developmental relevance*: Early brain development shows more regular connectivity before pruning and specialization
  
  **Characteristics in brain terms:**
  - Strong local processing capabilities
  - Inefficient global integration (information must travel through many nodes)
  - Highly segregated processing
  - Resilient to random damage but vulnerable to targeted attacks

- **Small-World Networks**: Balance of integration and segregation
  - *Properties*: High clustering (like regular networks) but short path lengths (like random networks)
  - *Example*: Healthy human brain networks typically exhibit small-world properties
  - *Mathematical definition*: $\sigma = \frac{C/C_{\text{random}}}{L/L_{\text{random}}} > 1$, where C is clustering and L is path length
  - *Functional significance*: Enables both specialized processing in local circuits and efficient integration across distant regions
  
  **Why small-world organization is important for the brain:**
  - Balances the metabolic cost of wiring with the need for efficient processing
  - Allows for specialized modules while maintaining efficient global communication
  - Provides resilience to both random failures and targeted attacks
  - Supports both segregated and integrated information processing
  
  **Clinical applications:**
  - Reduced small-worldness is observed in Alzheimer's disease, schizophrenia, and epilepsy
  - Developmental disorders often show altered small-world organization
  - Recovery from brain injury may correlate with restoration of small-world properties

- **Scale-Free Networks**: Few highly connected hub nodes, many sparsely connected nodes
  - *Properties*: Degree distribution follows a power law: $P(k) \sim k^{-\gamma}$
  - *Example*: Some evidence suggests the human connectome has scale-free properties, with hub regions in association cortices
  - *Defining characteristic*: The probability that a node has k connections decreases as k^-γ, where γ is typically between 2 and 3
  - *Relevance to brain organization*: Creates a hierarchy of connectivity where a few hubs play crucial roles in network integration
  
  **Hub nodes in the human brain:**
  - *Structural hubs*: Precuneus, posterior cingulate cortex, superior frontal gyrus, insula
  - *Functional hubs*: Posterior cingulate, precuneus, medial prefrontal cortex
  - *Clinical significance*: Hub regions are particularly vulnerable in neurodegenerative diseases ("rich club vulnerability")
  
  **Evolutionary perspective:**
  - Scale-free organization may have evolved to maximize efficiency while minimizing wiring costs
  - Hub regions tend to be evolutionarily newer cortical areas in humans compared to other species

**Visual comparison of network topologies:**

```
Regular Network      Small-World Network    Random Network
   o---o---o            o---o---o             o---o---o
   |   |   |            |   | \ |             | \ | / |
   o---o---o            o---o---o             o---o---o
   |   |   |            | / |   |             | / | \ |
   o---o---o            o---o---o             o---o---o
(High clustering,     (High clustering,      (Low clustering,
 long paths)           short paths)           short paths)
```

> **Citation**: Fornito, A., Zalesky, A., & Bullmore, E. (2016). Fundamentals of Brain Network Analysis. Academic Press. https://doi.org/10.1016/C2012-0-06036-X

### 3.2 Network Construction from Neuroimaging Data

Before applying BCT functions, researchers must construct brain networks from neuroimaging data. The process typically involves:

1. **Node Definition**: Defining brain regions using:
   - Anatomical parcellations (e.g., AAL, Desikan-Killiany)
   - Functional parcellations (e.g., Schaefer parcellation)
   - Voxel-based approaches

2. **Edge Definition**:
   - For structural networks: Diffusion MRI tractography measures (e.g., fiber count, fractional anisotropy)
   - For functional networks: Time series correlations (Pearson's r, partial correlations, wavelet coherence)

3. **Thresholding** (optional):
   - Absolute thresholding: Retaining connections above a certain weight
   - Proportional thresholding: Retaining a fixed percentage of strongest connections
   - Statistical thresholding: Retaining connections with statistical significance

**Decision flowchart for network construction:**

```
1. RESEARCH QUESTION
   ↓
2. SELECT MODALITY
   ├── Structural (DTI/DWI) → WHITE MATTER CONNECTIVITY
   ├── Functional (fMRI) → TEMPORAL CORRELATIONS
   └── Effective (Granger causality, DCM) → CAUSAL INFLUENCES
   ↓
3. DEFINE NODES
   ├── Anatomical atlas (e.g., AAL90, Desikan-Killiany)
   ├── Functional atlas (e.g., Schaefer400, Gordon)
   └── Voxel-wise (high-resolution, thousands of nodes)
   ↓
4. DEFINE EDGES
   ├── Binary (0/1) connections
   └── Weighted connections
       ├── Keep negative weights?
       └── Apply transformation (log, absolute value)?
   ↓
5. THRESHOLD?
   ├── No threshold (full networks)
   ├── Absolute threshold (fixed value)
   ├── Proportional threshold (fixed density)
   ├── Statistical threshold (p-value)
   └── Explore multiple thresholds
   ↓
6. NETWORK ANALYSIS
```

#### 3.2.1 Detailed Workflow: From Raw Data to Connectivity Matrix

For complete beginners, here's a step-by-step workflow for creating connectivity matrices:

**For Functional Connectivity (using fMRI data):**

1. **Preprocessing fMRI data**:
   - *Motion correction*: Aligns all volumes to correct for head movement
     - *Why it matters*: Even small head movements can create spurious correlations
     - *Common methods*: Rigid-body transformation with 6 parameters (3 translations, 3 rotations)
   
   - *Slice timing correction*: Adjusts for different acquisition times of slices
     - *Why it matters*: Brain slices are acquired sequentially, not simultaneously
     - *Common methods*: Temporal interpolation (often sinc interpolation)
   
   - *Spatial normalization*: Transforms individual brains to a standard template
     - *Why it matters*: Allows for comparison across subjects
     - *Common templates*: MNI152, Talairach
   
   - *Smoothing*: Applies a spatial filter (typically Gaussian)
     - *Why it matters*: Increases signal-to-noise ratio and accommodates anatomical variability
     - *Typical parameters*: 5-8mm FWHM (full-width at half-maximum) kernel
   
   - *Nuisance regression*: Removes confounding signals
     - *Common nuisance signals*: Motion parameters, CSF signal, white matter signal, global signal
     - *Controversial step*: Global signal regression can introduce negative correlations
   
   - *Bandpass filtering*: Retains frequencies of interest
     - *For resting-state*: Typically 0.01-0.1 Hz to remove physiological noise and scanner drift
     - *Rationale*: BOLD fluctuations of interest primarily occur at low frequencies

2. **Extracting regional time series**:
   - *Define ROIs using an atlas*: e.g., AAL90, Schaefer400
     - *Anatomical atlases*: Based on structural landmarks, consistent across individuals
     - *Functional atlases*: Based on functional homogeneity, better aligned with functional organization
   
   - *Calculate mean time series for each ROI*
     - *Process*: Average the time series of all voxels within each ROI
     - *Alternative approaches*: First eigenvariate extraction, weighted averages

3. **Computing correlations**:
   ```matlab
   % Assume time_series is an n×t matrix (n regions, t time points)
   FC_matrix = corrcoef(time_series');
   ```
   
   *What happens in this step*:
   - Calculates Pearson correlation between every pair of regional time series
   - Creates an n×n matrix where each element represents the temporal similarity between two regions
   - Values range from -1 (perfect anticorrelation) to 1 (perfect correlation)
   - The diagonal is always 1 (each region correlates perfectly with itself)

   *Alternative correlation measures*:
   - *Partial correlation*: Measures direct relationships controlling for all other regions
   - *Wavelet coherence*: Measures correlation in frequency-specific components
   - *Dynamic correlations*: Measures correlations in sliding time windows

4. **Creating an adjacency matrix**:
   ```matlab
   % Option 1: Binary thresholding
   threshold = 0.3;
   A_binary = FC_matrix > threshold;
   
   % Option 2: Proportional thresholding (keep top 10% of connections)
   density = 0.1;
   A_proportional = threshold_proportional(FC_matrix, density);
   
   % Option 3: Absolute value (for methods that cannot handle negative weights)
   A_absolute = abs(FC_matrix);
   
   % Remember to set diagonal to zero
   A_binary(eye(size(A_binary)) == 1) = 0;
   A_proportional(eye(size(A_proportional)) == 1) = 0;
   A_absolute(eye(size(A_absolute)) == 1) = 0;
   ```
   
   *Explanation of thresholding approaches*:
   - *Binary thresholding*: Creates a simple 0/1 network, losing information about connection strength
   - *Proportional thresholding*: Ensures all subjects/conditions have the same number of connections, making comparison easier
   - *Absolute value approach*: Treats negative correlations as strong positive ones, useful for measures that can't handle negative values

   *Key considerations*:
   - Setting diagonal to zero removes self-connections
   - The choice of threshold impacts all subsequent analyses
   - Analyzing across multiple thresholds can assess the robustness of findings

**For Structural Connectivity (using DTI/DWI data):**

1. **Preprocessing diffusion data**:
   - *Eddy current correction*: Corrects distortions from gradient switching
     - *Why it matters*: Eddy currents cause spatial distortions that vary across gradient directions
     - *Common methods*: Registration-based approaches like FSL's eddy
   
   - *Motion correction*: Aligns volumes to account for head movement
     - *Why it matters*: Movement between diffusion directions causes misalignment
     - *Common approach*: Register each volume to the b=0 image
   
   - *Tensor fitting or fiber orientation modeling*
     - *Diffusion Tensor Imaging (DTI)*: Fits a tensor model to each voxel
     - *Advanced models*: HARDI, Q-ball imaging, CSD for resolving crossing fibers

2. **Tractography**:
   - *Deterministic or probabilistic tracking*
     - *Deterministic*: Follows the principal diffusion direction, simpler but can't handle uncertainty
     - *Probabilistic*: Samples from a distribution of possible directions, handles uncertainty but is computationally intensive
   
   - *Whole-brain or region-to-region tracking*
     - *Whole-brain*: Seeds streamlines throughout the white matter
     - *Region-to-region*: Only tracks between specific ROIs, more computationally efficient

3. **Constructing connectivity matrix**:
   - *Count fiber streamlines between regions*
     - *Process*: Count how many streamlines connect each pair of regions
     - *Limitation*: Affected by distance bias (fewer streamlines between distant regions)
   
   - *Calculate mean fractional anisotropy (FA) along fibers*
     - *Alternative metric*: Represents the "quality" of the connection
     - *Biological interpretation*: Related to fiber organization, myelination, and axon density
   
   - *Normalize by ROI sizes or fiber lengths if needed*
     - *Why normalize*: Larger ROIs are likely to have more streamlines by chance
     - *Common approaches*: Divide by ROI surface area or volume, or adjust for fiber length

   ```matlab
   % Structural connectivity preprocessing example:
   % SC_counts: raw fiber count matrix
   % ROI_sizes: vector of ROI volumes/surface areas
   
   % Normalize by ROI sizes
   SC_normalized = SC_counts ./ (ROI_sizes * ROI_sizes');
   
   % Apply log transform (common for count data with skewed distribution)
   SC_log = log(SC_normalized + 1);  % Add 1 to avoid log(0)
   ```
   
   *What happens in normalization*:
   - Dividing by the product of ROI sizes corrects for the probability of connection by chance
   - Log transformation makes the distribution more normal and reduces the effect of outliers
   - Adding 1 before log transformation avoids the mathematical error of log(0)

#### 3.2.2 Parcellation Schemes: Choosing the Right Nodes

The choice of parcellation significantly impacts network properties:

- **Anatomical parcellations**:
  - Based on structural landmarks (gyri, sulci)
  - Examples: AAL (90/116 regions), Desikan-Killiany (68 regions)
  - Pros: Stable, anatomically meaningful
  - Cons: May not align with functional units
  
  *Impact on network analysis*:
  - Provides consistent regions across subjects and studies
  - May combine functionally distinct areas into single nodes
  - Often used in structural connectivity studies
  
  *Sample regions in AAL atlas*:
  - Frontal_Sup_L (left superior frontal gyrus)
  - Temporal_Mid_R (right middle temporal gyrus)
  - Precuneus_L (left precuneus)

- **Functional parcellations**:
  - Based on functional homogeneity
  - Examples: Schaefer parcellations (100-1000 regions), Gordon atlas
  - Pros: Better functional homogeneity
  - Cons: May vary between individuals or tasks
  
  *Impact on network analysis*:
  - Better reflects functional organization of the brain
  - Can identify network communities (e.g., default mode network, visual network)
  - Often used in functional connectivity studies
  
  *Sample networks in Schaefer atlas*:
  - DefaultA (regions in the default mode network)
  - VisCent (central visual network regions)
  - SomMotA (somatomotor network regions)

- **Connectivity-based parcellations**:
  - Derived from connectivity patterns themselves
  - Examples: Data-driven parcellations using clustering or ICA
  - Pros: Maximally captures connectivity structure
  - Cons: Less generalizable, difficult to compare across studies
  
  *Impact on network analysis*:
  - Can reveal natural divisions in connectivity patterns
  - Highly sensitive to data quality and preprocessing choices
  - May require validation across datasets

- **Resolution considerations**:
  - Coarse (tens of nodes): Easier computation, more stable, less specific
  - Fine (hundreds/thousands of nodes): Better spatial specificity, computationally intensive, noisier connections
  
  *Resolution trade-offs*:
  - *Coarse resolution (e.g., 90 regions)*
    - Advantages: Computationally efficient, statistically stable
    - Disadvantages: May combine distinct functional areas, limited spatial precision
  
  - *Fine resolution (e.g., 1000 regions)*
    - Advantages: Better spatial specificity, can detect subtle patterns
    - Disadvantages: Computationally demanding, noisier individual connections, more susceptible to motion artifacts

  *Resolution effects on network measures*:
  - Clustering coefficient tends to decrease with increasing resolution
  - Path length tends to increase with increasing resolution
  - Modularity detection becomes more detailed at higher resolutions
  
  *Practical recommendation for beginners*:
  - Start with established, moderate-resolution parcellations (e.g., AAL90 or Schaefer200)
  - Consider analyzing at multiple resolutions to ensure robustness of findings

#### 3.2.3 Critical Considerations in Network Construction

- **Handling negative correlations**: 
  - Option 1: Set negative values to zero (loses information)
    - *Rationale*: Some network measures are only defined for positive weights
    - *Consequence*: Potential loss of important information about antagonistic relationships
  
  - Option 2: Take absolute values (treats negative correlations as strong positive ones)
    - *Rationale*: Preserves the strength regardless of direction
    - *Consequence*: Fundamentally changes the interpretation of the network
  
  - Option 3: Analyze positive and negative networks separately
    - *Rationale*: Preserves the distinct nature of positive and negative relationships
    - *Consequence*: Doubles the number of analyses but maintains separate interpretations
  
  - Option 4: Use specialized metrics for signed networks
    - *Examples*: Signed clustering coefficient, signed modularity
    - *Advantage*: Directly incorporates both positive and negative weights in a theoretically sound way

  *Neurobiological interpretation*:
  - Positive correlations: Functional cooperation, shared inputs, or direct connections
  - Negative correlations: Functional opposition, competitive interactions, or reciprocal inhibition
  - *Controversial point*: Global signal regression can introduce artificial negative correlations

- **Thresholding decisions**:
  - No threshold: Retain all connections (noisy but complete)
    - *Advantage*: No arbitrary cutoff, preserves all information
    - *Disadvantage*: Includes noisy, potentially spurious connections
  
  - Fixed threshold: Same threshold for all subjects (different network densities)
    - *Advantage*: Consistent meaning of connection strength across subjects
    - *Disadvantage*: Different subjects will have different numbers of connections
  
  - Fixed density: Same percentage of connections (different threshold values)
    - *Advantage*: Controls for overall connectivity differences between subjects/groups
    - *Disadvantage*: Threshold value differs between subjects, may include noise in some subjects
  
  - Multiple thresholds: Analyze across a range of thresholds/densities
    - *Advantage*: Tests robustness of findings across different network densities
    - *Disadvantage*: Multiple comparisons issue, increases analysis complexity

  *Practical example*:
  - *Subject A* might have generally higher correlation values than *Subject B* due to:
    - Data quality differences (less noise = higher correlations)
    - Physiological differences
    - Behavioral differences during scan
  
  *Recommended approach for beginners*:
  - Use proportional thresholding to maintain equal density across subjects
  - Analyze across multiple densities (e.g., 5%, 10%, 15%, 20%)
  - Report results that are consistent across thresholds

- **Normalization considerations**:
  - Global signal regression: Controversial, can introduce artificial negative correlations
    - *Benefit*: Removes global confounds that affect all regions
    - *Risk*: May remove real global neural signals and introduce artificial anticorrelations
  
  - Region size effects: Larger regions may have more connections by chance
    - *Issue*: In structural connectivity, larger ROIs statistically intercept more streamlines
    - *Solution*: Normalize by ROI size (volume, surface area)
  
  - Connection distance: Long-distance connections are harder to detect with tractography
    - *Issue*: Tractography has difficulty following long-distance pathways due to noise and crossing fibers
    - *Solutions*: Distance-based normalization, distance-aware statistical testing

  *Importance for interpreting results*:
  - Preprocessing and normalization choices can dramatically affect network properties
  - Important to report all processing steps in detail
  - Consider testing the sensitivity of results to different preprocessing choices

> **Citation**: Zalesky, A., Fornito, A., & Bullmore, E. T. (2010). Network-based statistic: Identifying differences in brain networks. NeuroImage, 53(4), 1197-1207. https://doi.org/10.1016/j.neuroimage.2010.06.041

## 4. Core Network Measures in BCT

### 4.1 Measures of Segregation

Segregation refers to the presence of densely interconnected groups of brain regions, which enables specialized processing.

#### 4.1.1 Clustering Coefficient

The clustering coefficient quantifies the tendency of nodes to form triangles (closed triplets), reflecting local interconnectivity.

##### Mathematical Definition

For binary networks, the clustering coefficient of node i is defined as:

$$
C_i = \frac{2t_i}{k_i(k_i-1)}
$$

Where:
- $t_i$ is the number of triangles around node i
- $k_i$ is the degree of node i
- $k_i(k_i-1)/2$ is the maximum possible number of triangles

For weighted networks, the formula is extended to incorporate edge weights:

$$
C_i^w = \frac{\sum_{j,h} (w_{ij}w_{ih}w_{jh})^{1/3}}{k_i(k_i-1)}
$$

Where $w_{ij}$ represents the weight of the connection between nodes i and j.

##### Biological Interpretation

High clustering coefficients in the brain indicate regions with specialized processing capabilities:
- Primary sensory regions often show high clustering
- Reflects efficient local information processing
- Creates robust local circuits resistant to perturbation
- Changes in clustering have been observed in conditions like Alzheimer's disease and schizophrenia

##### Code Example with Visualization

```matlab
% Generate a toy network with community structure
N = 30;  % 30 nodes
A = zeros(N);
% Create three modules with dense within-module connections
module_size = N/3;
for i = 1:N
    for j = 1:N
        % If nodes belong to same module, high probability of connection
        if floor((i-1)/module_size) == floor((j-1)/module_size)
            A(i,j) = rand() > 0.2;  % 80% chance of connection
        else
            A(i,j) = rand() > 0.9;  % 10% chance of connection
        end
    end
end
% Make symmetric and remove self-connections
A = A - diag(diag(A));
A = (A + A')/2;
A(A > 0) = 1;  % Binarize

% Calculate clustering coefficients
C_bin = clustering_coef_bu(A);

% Visualize network with nodes colored by clustering coefficient
figure;
% Using imagesc to plot adjacency matrix
subplot(1,2,1);
imagesc(A);
colormap('gray');
title('Adjacency Matrix with Modular Structure');
xlabel('Node Index'); ylabel('Node Index');
% Plot clustering coefficients
subplot(1,2,2);
bar(C_bin);
title('Clustering Coefficients');
xlabel('Node Index'); ylabel('Clustering Coefficient');

% Average clustering coefficient
mean_C = mean(C_bin);
fprintf('Mean clustering coefficient: %.4f\n', mean_C);
```

**Expected output**: Nodes within the same module will typically have higher clustering coefficients (values closer to 1) because their neighbors are also connected to each other.

```matlab
% For binary networks
C_bin = clustering_coef_bu(A);

% For weighted networks
C_wei = clustering_coef_wu(A);
```

#### 4.1.2 Transitivity

Transitivity is a global version of the clustering coefficient, defined as the ratio of triangles to triplets in the network.

##### Mathematical Definition

$$
T = \frac{3 \times \text{number of triangles}}{\text{number of connected triplets}} = \frac{3 \times \text{number of triangles}}{\sum_i k_i(k_i-1)}
$$

This provides a single network-wide measure of clustering that is less influenced by nodes with low degree than the average clustering coefficient.

##### Differences from Clustering Coefficient

- Clustering coefficient: Average of the local clustering values (more influenced by low-degree nodes)
- Transitivity: Ratio of triangles to triplets across the whole network (more accurate global measure)

```matlab
T = transitivity_bu(A);  % Binary undirected
T = transitivity_wu(A);  % Weighted undirected
```

#### 4.1.3 Modularity

Modularity quantifies the degree to which a network can be divided into non-overlapping communities or modules with dense within-module connections and sparse between-module connections.

##### Mathematical Definition

$$
Q = \frac{1}{2m}\sum_{ij} \left[ A_{ij} - \frac{k_i k_j}{2m} \right] \delta(c_i, c_j)
$$

Where:
- $m$ is the total number of edges
- $A_{ij}$ is the adjacency matrix
- $k_i$ and $k_j$ are the degrees of nodes i and j
- $\delta(c_i, c_j)$ equals 1 if nodes i and j are in the same community, 0 otherwise

##### Detecting Communities: A Step-by-Step Example

The BCT implements several algorithms for community detection, with the Louvain algorithm being one of the most popular:

```matlab
% Detect communities using Louvain algorithm
[M, Q] = community_louvain(A);  % M: module assignments, Q: modularity score

% Visualize the community structure
figure;
% Reorder the adjacency matrix by community
[~, idx] = sort(M);
A_reordered = A(idx, idx);
% Plot the reordered adjacency matrix
imagesc(A_reordered);
colormap('gray');
title(sprintf('Adjacency Matrix Ordered by Communities (Q=%.3f)', Q));
xlabel('Node Index (Reordered)'); ylabel('Node Index (Reordered)');

% Draw horizontal and vertical lines separating communities
hold on;
% Find indices where community assignment changes
change_points = find(diff(M(idx)) ~= 0);
for i = 1:length(change_points)
    line([0.5, N+0.5], [change_points(i)+0.5, change_points(i)+0.5], 'Color', 'r');
    line([change_points(i)+0.5, change_points(i)+0.5], [0.5, N+0.5], 'Color', 'r');
end
```

##### Interpretations in Neuroscience

- Modules in brain networks often correspond to functional systems (visual, motor, default mode, etc.)
- Strong modularity balances integration and segregation
- Altered modularity appears in many neurological and psychiatric conditions
- Developmental studies show that modularity changes throughout brain maturation
- Highly modular structures might be more resilient to focal damage

> **Citation**: Sporns, O., & Betzel, R. F. (2016). Modular Brain Networks. Annual Review of Psychology, 67, 613-640. https://doi.org/10.1146/annurev-psych-122414-033634

#### 4.1.4 Local Efficiency

Local efficiency is a node-wise measure that quantifies how efficiently a node's neighbors can communicate when the node is removed. It is related to clustering.

##### Mathematical Definition

$$
E_{loc,i} = \frac{1}{k_i(k_i-1)}\sum_{j,h \in N_i, j \neq h} \frac{1}{d_{jh}(N_i)}
$$

Where:
- $N_i$ is the set of neighbors of node i
- $d_{jh}(N_i)$ is the length of the shortest path between j and h that contains only neighbors of i

```matlab
% Calculate local efficiency
E_loc = efficiency_bin(A, 1);  % Second parameter 1 means "local"
```

### 4.2 Measures of Integration

Integration measures capture the ease with which brain regions communicate with each other.

#### 4.2.1 Path Length

The characteristic path length is the average shortest path length between all pairs of nodes in the network, reflecting global network efficiency.

##### Mathematical Definition

For binary networks, the shortest path length $d_{ij}$ is the minimum number of edges that must be traversed to go from node i to node j. The characteristic path length is:

$$
L = \frac{1}{n(n-1)} \sum_{i \neq j} d_{ij}
$$

For weighted networks, the edge weights typically represent connection strengths. Since stronger connections represent shorter "distances," weights are often inverted or transformed:

$$
d_{ij}^w = \sum_{\text{edge} \in \text{shortest path}(i,j)} \frac{1}{w_{\text{edge}}}
$$

Or alternatively:

$$
d_{ij}^w = \sum_{\text{edge} \in \text{shortest path}(i,j)} \frac{1}{\ln(w_{\text{edge}})}
$$

##### How to Calculate Distance Matrices

Before calculating path lengths, we need to convert the connectivity/adjacency matrix to a distance matrix:

```matlab
% For binary networks
D_bin = distance_bin(A);     

% For weighted networks (weights represent connection strengths)
% First, invert weights to get distances (stronger connection = shorter distance)
W_dist = 1 ./ A;
% Replace infinite values (from zeros in A) with 0 to indicate no direct connection
W_dist(isinf(W_dist)) = 0;
% Calculate distance matrix
D_wei = distance_wei(W_dist);

% Calculate characteristic path length
L_bin = charpath(D_bin);
L_wei = charpath(D_wei);

% Display the characteristic path length
fprintf('Binary characteristic path length: %.4f\n', L_bin);
fprintf('Weighted characteristic path length: %.4f\n', L_wei);
```

##### Biological Interpretation

- Shorter path lengths indicate more integrated information processing
- Path length increases have been observed in Alzheimer's disease and schizophrenia
- Efficient long-distance connections are metabolically expensive but are critical for integration
- Developmental studies show decreasing path lengths during brain maturation

#### 4.2.2 Global Efficiency

Global efficiency is the average inverse shortest path length, which is more robust to disconnected components.

##### Mathematical Definition

$$
E_{glob} = \frac{1}{n(n-1)} \sum_{i \neq j} \frac{1}{d_{ij}}
$$

Note that when nodes i and j are not connected, $d_{ij} = \infty$ and thus $1/d_{ij} = 0$, avoiding mathematical problems with disconnected components.

##### Comparing Path Length and Global Efficiency

Path length and global efficiency are inversely related but handle disconnected components differently:
- Path length: Disconnected pairs contribute infinity (often excluded or approximated)
- Global efficiency: Disconnected pairs contribute zero (naturally handled)

```matlab
% Calculate global efficiency directly
E_glob_bin = efficiency_bin(A);  % Binary
E_glob_wei = efficiency_wei(A);  % Weighted

fprintf('Binary global efficiency: %.4f\n', E_glob_bin);
fprintf('Weighted global efficiency: %.4f\n', E_glob_wei);

% Relationship between path length and efficiency
% For a fully connected network:
% E_glob ≈ 1/L
```

##### Visualizing Integration in Network

```matlab
% Calculate distance matrix
D = distance_bin(A);

% Find diameter (maximum shortest path length)
diameter = max(D(~isinf(D)));

% Pick two distant nodes (with shortest path = diameter)
[row, col] = find(D == diameter, 1);

% Find shortest path between these nodes
[shortest_path, distance] = breadth(A, row, col);

% Highlight the shortest path in a visualization
figure;
% Create coordinates for nodes (for simple visualization)
theta = linspace(0, 2*pi, N+1);
theta = theta(1:end-1);
x = cos(theta);
y = sin(theta);

% Plot all connections first
gplot(A, [x' y'], 'k-');
hold on;

% Highlight the shortest path
for i = 1:length(shortest_path)-1
    n1 = shortest_path(i);
    n2 = shortest_path(i+1);
    line([x(n1) x(n2)], [y(n1) y(n2)], 'Color', 'r', 'LineWidth', 3);
end

% Plot nodes
scatter(x, y, 100, 'filled');
% Highlight source and target nodes
scatter(x(row), y(row), 150, 'r', 'filled');
scatter(x(col), y(col), 150, 'g', 'filled');

title(sprintf('Shortest Path (Distance = %d)', distance));
axis equal off;
```

> **Citation**: Latora, V., & Marchiori, M. (2001). Efficient behavior of small-world networks. Physical Review Letters, 87(19), 198701.

### 4.3 Measures of Centrality

Centrality measures identify the most important nodes (hubs) in a network.

#### 4.3.1 Degree/Strength Centrality

The simplest measure of node importance is its degree (number of connections) or strength (sum of connection weights).

##### Mathematical Definition

For binary networks, the degree of node i is:

$$
k_i = \sum_{j=1}^{n} a_{ij}
$$

For weighted networks, the strength of node i is:

$$
s_i = \sum_{j=1}^{n} w_{ij}
$$

##### Implementation and Visualization

```matlab
% Degree for binary networks
k = degrees_und(A);

% Strength for weighted networks
s = strengths_und(A);

% Visualize the degree distribution
figure;
subplot(1,2,1);
histogram(k, 'BinMethod', 'integers');
xlabel('Degree (k)'); ylabel('Frequency');
title('Degree Distribution');

% Visualize the network with node size proportional to degree
subplot(1,2,2);
% Create coordinates for nodes (for simple visualization)
theta = linspace(0, 2*pi, N+1);
theta = theta(1:end-1);
x = cos(theta);
y = sin(theta);

% Plot all connections first
gplot(A, [x' y'], 'k-');
hold on;

% Plot nodes with size proportional to degree
scatter(x, y, 20+k*5, 'filled');

title('Network with Node Size by Degree');
axis equal off;
```

##### Biological Interpretation of Hub Regions

Brain hubs (high-degree nodes) have specific properties:
- Higher metabolic activity
- Longer-distance connections
- Critical for network resilience
- Often affected early in neurodegenerative diseases
- Evolutionarily more conserved

Common hub regions in the human brain include:
- Precuneus/posterior cingulate cortex
- Superior parietal cortex
- Medial prefrontal cortex
- Parts of the lateral prefrontal cortex

#### 4.3.2 Betweenness Centrality

Betweenness centrality quantifies the number of shortest paths passing through a node, indicating its importance for network communication.

##### Mathematical Definition

$$
b_i = \sum_{j \neq i \neq k} \frac{\sigma_{jk}(i)}{\sigma_{jk}}
$$

Where:
- $\sigma_{jk}$ is the total number of shortest paths between nodes j and k
- $\sigma_{jk}(i)$ is the number of those paths that pass through node i

##### Intuitive Explanation

Betweenness identifies "bridge" nodes that connect different parts of the network:
- High betweenness nodes may not necessarily have high degree
- Critical for information flow between otherwise distant regions
- Removal can significantly disrupt network communication

```matlab
% Calculate betweenness centrality
BC_bin = betweenness_bin(A);  % Binary
BC_wei = betweenness_wei(A);  % Weighted

% Normalize by the maximum possible betweenness
BC_bin_norm = BC_bin / ((N-1)*(N-2)/2);
```

#### 4.3.3 Eigenvector Centrality

Eigenvector centrality assigns relative scores to nodes based on the concept that connections to high-scoring nodes contribute more to the score of the node than equal connections to low-scoring nodes.

##### Mathematical Definition

The eigenvector centrality of node i is the i-th element of the eigenvector corresponding to the largest eigenvalue λ of the adjacency matrix A:

$$
\lambda e_i = \sum_{j=1}^{n} a_{ij} e_j
$$

Where:
- $e_i$ is the eigenvector centrality of node i
- $\lambda$ is the largest eigenvalue of the adjacency matrix

##### Implementation in MATLAB

```matlab
% Calculate eigenvector centrality
[V, D] = eig(A);
[~, ind] = max(diag(D));
EC = abs(V(:, ind));

% Compare different centrality measures
figure;
subplot(3,1,1);
bar(k);
title('Degree Centrality');
xlabel('Node'); ylabel('Degree');

subplot(3,1,2);
bar(BC_bin);
title('Betweenness Centrality');
xlabel('Node'); ylabel('Betweenness');

subplot(3,1,3);
bar(EC);
title('Eigenvector Centrality');
xlabel('Node'); ylabel('Eigenvector Centrality');
```

##### Biological Interpretation

Different centrality measures identify different aspects of node importance:
- Degree: Direct connections and immediate influence
- Betweenness: Control over information flow between regions
- Eigenvector: Captures influence within connected components and identifies "prestige"

In the brain:
- Association cortices often have high eigenvector centrality
- Regions connecting different modules have high betweenness
- Primary sensory/motor regions may have high degree but lower betweenness/eigenvector centrality

> **Citation**: van den Heuvel, M. P., & Sporns, O. (2013). Network hubs in the human brain. Trends in Cognitive Sciences, 17(12), 683-696. https://doi.org/10.1016/j.tics.2013.09.012

### 4.4 Small-World Properties

Small-world networks have high clustering (like regular networks) and short path lengths (like random networks), enabling both specialized and integrated processing.

#### 4.4.1 Small-World Coefficient

##### Mathematical Definition

The small-world coefficient σ is defined as:

$$
\sigma = \frac{C/C_{\text{rand}}}{L/L_{\text{rand}}}
$$

Where:
- C is the clustering coefficient of the network
- C_rand is the clustering coefficient of an equivalent random network
- L is the characteristic path length of the network
- L_rand is the characteristic path length of an equivalent random network

A network is considered "small-world" if σ > 1.

#### 4.4.2 Generating Random Networks for Comparison

To determine if a network has small-world properties, we need to compare it to random networks with similar basic properties:

```matlab
% Generate random networks with same degree distribution
nrand = 100;  % Number of random networks
C_rand_values = zeros(nrand, 1);
L_rand_values = zeros(nrand, 1);

for i = 1:nrand
    % Create randomized network preserving degree distribution
    R = randmio_und(A, 10);  % 10 rewiring iterations per edge
    
    % Calculate metrics
    C_rand_values(i) = mean(clustering_coef_bu(R));
    D_rand = distance_bin(R);
    L_rand_values(i) = charpath(D_rand);
end

% Calculate small-world measures
C = mean(clustering_coef_bu(A));
D = distance_bin(A);
L = charpath(D);

% Take average of random networks
C_rand = mean(C_rand_values);
L_rand = mean(L_rand_values);

% Small-world coefficient
sigma = (C/C_rand)/(L/L_rand);

fprintf('Clustering Coefficient:\n  Network: %.4f\n  Random: %.4f\n  Ratio: %.4f\n', C, C_rand, C/C_rand);
fprintf('Path Length:\n  Network: %.4f\n  Random: %.4f\n  Ratio: %.4f\n', L, L_rand, L/L_rand);
fprintf('Small-world coefficient (sigma): %.4f\n', sigma);
```

#### 4.4.3 Interpreting Small-World Results

- σ ≈ 1: The network has random-like properties
- σ >> 1: The network has small-world properties
- Typically, healthy brain networks have σ in the range of 1.5 to 3

##### Visualizing the Small-World Spectrum

```matlab
% Create a regular lattice network
N = 30;
k = 4;  % Each node connects to k neighbors
A_lattice = zeros(N);
for i = 1:N
    for j = 1:k/2
        A_lattice(i, mod(i+j-1, N)+1) = 1;
        A_lattice(i, mod(i-j-1+N, N)+1) = 1;
    end
end

% Create completely random network with same density
p = density_und(A_lattice);
A_random = double(rand(N) < p);
A_random = triu(A_random, 1) + triu(A_random, 1)';

% Create a range of small-world networks by rewiring
rewire_prob = [0, 0.01, 0.05, 0.1, 0.5, 1];
C_values = zeros(size(rewire_prob));
L_values = zeros(size(rewire_prob));

for i = 1:length(rewire_prob)
    % Create small-world network with Watts-Strogatz model
    A_sw = A_lattice;
    
    % Rewire edges with probability rewire_prob(i)
    for j = 1:N
        neighbors = find(A_sw(j,:));
        for n = 1:length(neighbors)
            if rand() < rewire_prob(i)
                % Remove current edge
                k = neighbors(n);
                A_sw(j,k) = 0;
                A_sw(k,j) = 0;
                
                % Add new edge to random node
                possible_targets = setdiff(1:N, [j find(A_sw(j,:))]);
                if ~isempty(possible_targets)
                    new_k = possible_targets(randi(length(possible_targets)));
                    A_sw(j,new_k) = 1;
                    A_sw(new_k,j) = 1;
                end
            end
        end
    end
    
    % Calculate metrics
    C_values(i) = mean(clustering_coef_bu(A_sw));
    D = distance_bin(A_sw);
    L_values(i) = charpath(D);
end

% Normalize values
C_norm = C_values / C_values(1);
L_norm = L_values / L_values(1);

% Plot the small-world spectrum
figure;
semilogx(rewire_prob, C_norm, 'o-', 'LineWidth', 2);
hold on;
semilogx(rewire_prob, L_norm, 's-', 'LineWidth', 2);
xlabel('Rewiring Probability');
ylabel('Normalized Value');
legend('Clustering Coefficient', 'Path Length');
title('Small-World Spectrum');
grid on;
```

The small-world property is thought to enable brain networks to efficiently balance segregated (specialized) and integrated processing:
- High clustering: Local processing and specialization
- Short path lengths: Global integration and rapid information transfer

> **Citation**: Watts, D. J., & Strogatz, S. H. (1998). Collective dynamics of 'small-world' networks. Nature, 393(6684), 440-442. https://doi.org/10.1038/30918

## 5. Advanced Analyses

### 5.1 Rich Club Organization

Rich club organization refers to the tendency of highly connected nodes to be more densely connected among themselves than with nodes of lower degree.

#### 5.1.1 Mathematical Foundation

The rich club coefficient Φ(k) is defined as:

$$
\Phi(k) = \frac{E_{>k}}{N_{>k}(N_{>k}-1)/2}
$$

Where:
- $E_{>k}$ is the number of edges between nodes with degree > k
- $N_{>k}$ is the number of nodes with degree > k
- $N_{>k}(N_{>k}-1)/2$ is the maximum possible number of edges between these nodes

To determine if rich club organization is significant, we normalize Φ(k) by the rich club coefficient of randomized networks with the same degree distribution:

$$
\Phi_{norm}(k) = \frac{\Phi(k)}{\Phi_{random}(k)}
$$

Values of $\Phi_{norm}(k) > 1$ indicate rich club organization.

#### 5.1.2 Implementation and Visualization

```matlab
% Calculate rich club coefficient for a range of degrees
k_range = 1:max(degrees_und(A));
RC = rich_club_bu(A, k_range);

% Generate randomized networks for normalization
nrand = 100;
RC_rand = zeros(length(k_range), nrand);
for i = 1:nrand
    R = randmio_und(A, 10);
    RC_rand(:,i) = rich_club_bu(R, k_range);
end

% Calculate normalized rich club coefficient
RC_norm = RC ./ mean(RC_rand, 2);

% Plot rich club coefficient
figure;
subplot(2,1,1);
plot(k_range, RC, 'b-', 'LineWidth', 2);
hold on;
plot(k_range, mean(RC_rand, 2), 'r--', 'LineWidth', 2);
xlabel('Degree (k)'); ylabel('Rich Club Coefficient \Phi(k)');
legend('Network', 'Random Networks');
title('Rich Club Coefficient');
grid on;

% Plot normalized rich club coefficient
subplot(2,1,2);
plot(k_range, RC_norm, 'k-', 'LineWidth', 2);
hold on;
plot(k_range, ones(size(k_range)), 'r--');
xlabel('Degree (k)'); ylabel('Normalized Rich Club Coefficient');
title('Normalized Rich Club Coefficient');
grid on;

% Find significant rich club levels
sig_k = k_range(RC_norm > 1.1);  % Threshold of 1.1 for significance
if ~isempty(sig_k)
    fprintf('Significant rich club organization at degree k >= %d\n', min(sig_k));
    
    % Identify rich club nodes
    rich_nodes = find(degrees_und(A) >= min(sig_k));
    fprintf('Rich club nodes: ');
    fprintf('%d ', rich_nodes);
    fprintf('\n');
    
    % Compute rich club, feeder, and local connections
    n = size(A, 1);
    is_rich = zeros(n, 1);
    is_rich(rich_nodes) = 1;
    
    % Rich club edges (rich-to-rich)
    rich_edges = zeros(size(A));
    for i = 1:length(rich_nodes)
        for j = i+1:length(rich_nodes)
            if A(rich_nodes(i), rich_nodes(j)) > 0
                rich_edges(rich_nodes(i), rich_nodes(j)) = 1;
                rich_edges(rich_nodes(j), rich_nodes(i)) = 1;
            end
        end
    end
    
    % Feeder edges (rich-to-peripheral)
    feeder_edges = zeros(size(A));
    for i = 1:n
        for j = 1:n
            if A(i,j) > 0 && ((is_rich(i) && ~is_rich(j)) || (~is_rich(i) && is_rich(j)))
                feeder_edges(i,j) = 1;
            end
        end
    end
    
    % Local edges (peripheral-to-peripheral)
    local_edges = A - rich_edges - feeder_edges;
    
    fprintf('Number of rich club connections: %d\n', sum(sum(rich_edges))/2);
    fprintf('Number of feeder connections: %d\n', sum(sum(feeder_edges))/2);
    fprintf('Number of local connections: %d\n', sum(sum(local_edges))/2);
end
```

#### 5.1.3 Biological Significance

Rich club organization in the brain:
- Creates a high-capacity backbone for global communication
- Typically includes regions in the default mode network, salience network, and executive control network
- Is associated with higher cognitive functions
- Shows alterations in neurological and psychiatric disorders
- May make the brain more efficient but also more vulnerable to targeted attacks

> **Citation**: van den Heuvel, M. P., & Sporns, O. (2011). Rich-club organization of the human connectome. Journal of Neuroscience, 31(44), 15775-15786. https://doi.org/10.1523/JNEUROSCI.3539-11.2011

### 5.2 Network Resilience and Attack Vulnerability

Network resilience measures how the network's integrity is maintained when nodes are removed, simulating lesions or brain damage.

#### 5.2.1 Types of Attack Strategies

1. **Random Failure**: Randomly removing nodes to simulate non-targeted damage
2. **Targeted Attack**: Removing nodes based on centrality measures to simulate targeted attacks or focal lesions

#### 5.2.2 Measuring Network Deterioration

Key metrics to quantify network damage include:
- Size of the largest connected component
- Global efficiency
- Number of isolated components
- Characteristic path length

#### 5.2.3 Implementation Example

```matlab
% Set up attack simulations
n_removals = size(A, 1) - 1;  % Remove all but one node
attack_types = {'random', 'degree', 'betweenness'};
n_attacks = length(attack_types);
n_repeats = 10;  % For random attacks

% Initialize arrays to store results
size_gcc = zeros(n_removals, n_attacks);
global_eff = zeros(n_removals, n_attacks);

% Original network metrics for reference
orig_size = size(A, 1);
orig_eff = efficiency_bin(A);

% Simulate attacks
for attack = 1:n_attacks
    attack_type = attack_types{attack};
    
    if strcmp(attack_type, 'random')
        % Average over multiple random attack sequences
        size_gcc_tmp = zeros(n_removals, n_repeats);
        global_eff_tmp = zeros(n_removals, n_repeats);
        
        for repeat = 1:n_repeats
            % Create copy of original network
            A_tmp = A;
            nodes = 1:size(A, 1);
            % Random permutation of nodes
            remove_order = randperm(length(nodes));
            
            for i = 1:n_removals
                % Remove node
                node_to_remove = remove_order(i);
                A_tmp(node_to_remove, :) = 0;
                A_tmp(:, node_to_remove) = 0;
                
                % Measure network properties
                [comp_sizes, comp] = components(A_tmp);
                size_gcc_tmp(i, repeat) = max(comp_sizes) / orig_size;
                global_eff_tmp(i, repeat) = efficiency_bin(A_tmp) / orig_eff;
            end
        end
        % Average over repeats
        size_gcc(:, attack) = mean(size_gcc_tmp, 2);
        global_eff(:, attack) = mean(global_eff_tmp, 2);
    else
        % Create copy of original network
        A_tmp = A;
        
        % Calculate node centrality for targeted attack
        if strcmp(attack_type, 'degree')
            centrality = degrees_und(A);
        elseif strcmp(attack_type, 'betweenness')
            centrality = betweenness_bin(A);
        end
        
        % Sort nodes by centrality (descending)
        [~, remove_order] = sort(centrality, 'descend');
        
        for i = 1:n_removals
            % Remove node
            node_to_remove = remove_order(i);
            A_tmp(node_to_remove, :) = 0;
            A_tmp(:, node_to_remove) = 0;
            
            % Measure network properties
            [comp_sizes, comp] = components(A_tmp);
            size_gcc(i, attack) = max(comp_sizes) / orig_size;
            global_eff(i, attack) = efficiency_bin(A_tmp) / orig_eff;
        end
    end
end

% Plot results
figure;
subplot(2,1,1);
plot(1:n_removals, size_gcc, 'LineWidth', 2);
xlabel('Number of Nodes Removed');
ylabel('Relative Size of Giant Component');
legend(attack_types);
title('Network Fragmentation Under Attack');
grid on;

subplot(2,1,2);
plot(1:n_removals, global_eff, 'LineWidth', 2);
xlabel('Number of Nodes Removed');
ylabel('Relative Global Efficiency');
legend(attack_types);
title('Efficiency Degradation Under Attack');
grid on;
```

#### 5.2.4 Clinical Applications

Attack simulations help understand:
- Brain resilience to injury or disease
- Potential targets for therapeutic interventions
- Individual differences in vulnerability to specific damage patterns
- Effects of surgical resections in epilepsy treatment
- Progressive deterioration in neurodegenerative diseases

> **Citation**: Achard, S., Salvador, R., Whitcher, B., Suckling, J., & Bullmore, E. (2006). A resilient, low-frequency, small-world human brain functional network with highly connected association cortical hubs. Journal of Neuroscience, 26(1), 63-72. https://doi.org/10.1523/JNEUROSCI.3874-05.2006

### 5.3 Motif Analysis

Network motifs are recurring, significant patterns of interconnections that appear more frequently than in randomized networks.

#### 5.3.1 Understanding Motifs

Motifs are subgraphs (typically 3-4 nodes) that represent fundamental building blocks of networks. Different types of networks (e.g., social, biological, technological) often have characteristic motif signatures.

In brain networks, motifs are thought to represent elementary processing units with specific computational functions:
- Divergent motifs: One node influencing many (broadcasting)
- Convergent motifs: Many nodes influencing one (integration)
- Chain motifs: Sequential processing
- Cyclic motifs: Recurrent processing or feedback
- Fully connected motifs: Densely interacting processing units

#### 5.3.2 Motif Counting in Directed Networks

For 3-node motifs in directed networks, there are 13 possible unique patterns:

```matlab
% For directed networks - count 3-node motifs
M = motif3struct_dir(A_dir);

% Plot motif counts
figure;
bar(M, 'FaceColor', [0.4 0.6 0.8]);
xlabel('Motif ID'); ylabel('Frequency');
title('Distribution of 3-node Motifs');
```

#### 5.3.3 Motif Significance through Randomization

To determine if motifs are significantly overrepresented, we compare their frequency to randomized networks:

```matlab
% Generate randomized networks
nrand = 100;
M_rand = zeros(13, nrand);

for i = 1:nrand
    % Preserve in/out degree distribution
    R = randmio_dir(A_dir, 10);
    M_rand(:,i) = motif3struct_dir(R);
end

% Calculate z-score for each motif
M_mean = mean(M_rand, 2);
M_std = std(M_rand, 0, 2);
Z = (M - M_mean) ./ M_std;

% Plot z-scores
figure;
bar(Z);
xlabel('Motif ID'); ylabel('Z-score');
title('Motif Significance (Z-scores)');
grid on;

% Highlight significant motifs
hold on;
significant = abs(Z) > 2;
bar(find(significant), Z(significant), 'r');
```

#### 5.3.4 Biological Interpretation of Motifs

Different motifs can have specific functional implications:
- **Feedback loops** (motif with reciprocal connections) may stabilize neural activity
- **Feed-forward loops** may filter signaling noise or generate transient responses
- **Fan-in motifs** (multiple inputs to one node) represent integration of information
- **Fan-out motifs** (one node projecting to many) represent broadcasting or amplification

> **Citation**: Sporns, O., & Kötter, R. (2004). Motifs in brain networks. PLoS Biology, 2(11), e369. https://doi.org/10.1371/journal.pbio.0020369

### 5.4 Communicability and Alternative Path Measures

Beyond shortest paths, there are other measures that quantify how nodes communicate through multiple paths simultaneously.

#### 5.4.1 Communicability

Communicability accounts for all possible paths between nodes, with longer paths contributing less:

$$
G_{ij} = \sum_{k=0}^{\infty} \frac{A^k_{ij}}{k!} = (e^A)_{ij}
$$

Where:
- $A$ is the adjacency matrix
- $A^k$ represents paths of length k
- $e^A$ is the matrix exponential

This captures parallel information transfer via multiple routes, which may be more realistic for brain communication.

```matlab
% Calculate communicability
G = expm(A);  % Matrix exponential

% Remove diagonal (self-communicability)
G = G - diag(diag(G));

% Visualize communicability matrix
figure;
imagesc(G);
colormap('jet');
colorbar;
title('Communicability Matrix');
xlabel('Node Index'); ylabel('Node Index');
```

#### 5.4.2 Navigation Efficiency

Network navigation considers greedy routing strategies where nodes only have information about their local neighborhood, which may be more biologically plausible than global shortest paths:

```matlab
% A very simple navigation simulation (requires node coordinates)
% Assume coords is an n×3 matrix of node coordinates

% Initialize navigation success matrix
n = size(A, 1);
nav_success = zeros(n);
hop_count = zeros(n);

% Simulate navigation from each source to each target
for source = 1:n
    for target = 1:n
        if source == target
            continue;
        end
        
        % Start navigation
        current = source;
        path = [current];
        
        % Maximum steps to prevent infinite loops
        max_steps = n * 2;
        
        for step = 1:max_steps
            if current == target
                % Successfully reached target
                nav_success(source, target) = 1;
                hop_count(source, target) = length(path) - 1;
                break;
            end
            
            % Get neighbors of current node
            neighbors = find(A(current, :));
            
            if isempty(neighbors)
                % Dead end
                break;
            end
            
            % Calculate Euclidean distance from each neighbor to target
            distances = zeros(length(neighbors), 1);
            for i = 1:length(neighbors)
                distances(i) = norm(coords(neighbors(i),:) - coords(target,:));
            end
            
            % Move to neighbor closest to target
            [~, idx] = min(distances);
            next = neighbors(idx);
            
            % Check for loops
            if ismember(next, path)
                % We're in a loop
                break;
            end
            
            current = next;
            path = [path current];
        end
    end
end

% Calculate navigation efficiency
nav_efficiency = sum(sum(nav_success)) / (n * (n-1));
fprintf('Navigation success rate: %.4f\n', nav_efficiency);
```

#### 5.4.3 Search Information and Path Transitivity

Search information quantifies how difficult it is to navigate along the shortest path, considering the probability of making the correct choice at each node:

```matlab
% Calculate distance matrix
D = distance_bin(A);
% Compute search information (this is a simplified version)
n = size(A, 1);
SI = zeros(n);

for i = 1:n
    for j = 1:n
        if i == j || isinf(D(i,j))
            SI(i,j) = Inf;
            continue;
        end
        
        % Find shortest path from i to j
        path = retrieve_shortest_path(D, i, j);  % Custom function
        
        % Calculate search information
        si = 0;
        for s = 1:length(path)-1
            curr = path(s);
            next = path(s+1);
            k_curr = sum(A(curr,:));  % Degree of current node
            si = si + log2(k_curr);  % Add log of degree (information needed to select correct neighbor)
        end
        
        SI(i,j) = si;
    end
end
```

#### 5.4.4 Path Ensembles and Diffusion Distance

Diffusion distance considers how a random walker would navigate from one node to another, capturing all possible pathways weighted by their probability:

```matlab
% Calculate normalized Laplacian
D_inv_sqrt = diag(1./sqrt(sum(A, 2)));
L_norm = eye(n) - D_inv_sqrt * A * D_inv_sqrt;

% Define diffusion time
t = 3;  % Typical diffusion time

% Calculate diffusion kernel
K = expm(-t * L_norm);

% Calculate diffusion distance
diffusion_dist = zeros(n);
for i = 1:n
    for j = 1:n
        diffusion_dist(i,j) = sqrt(sum((K(i,:) - K(j,:)).^2));
    end
end
```

### 5.5 Temporal Network Analysis

Most brain network analyses consider static networks, but brain connectivity is dynamic. Temporal network analysis captures how networks change over time.

#### 5.5.1 Sliding Window Analysis

The simplest approach is to compute connectivity in sliding time windows:

```matlab
% Assume time_series is a regions × time points matrix
n_regions = size(time_series, 1);
n_timepoints = size(time_series, 2);

% Sliding window parameters
window_size = 60;  % 60 time points
step_size = 10;    % 10 time points

% Calculate number of windows
n_windows = floor((n_timepoints - window_size) / step_size) + 1;

% Initialize dynamic connectivity matrix
dyn_FC = zeros(n_regions, n_regions, n_windows);

% Calculate connectivity for each window
for w = 1:n_windows
    start_idx = (w-1) * step_size + 1;
    end_idx = start_idx + window_size - 1;
    
    % Extract time series for current window
    window_ts = time_series(:, start_idx:end_idx);
    
    % Calculate correlation
    dyn_FC(:,:,w) = corrcoef(window_ts');
end

% Analyze network metrics over time
temporal_metrics = struct();
temporal_metrics.clustering = zeros(n_windows, 1);
temporal_metrics.path_length = zeros(n_windows, 1);
temporal_metrics.modularity = zeros(n_windows, 1);

for w = 1:n_windows
    % Threshold the correlation matrix
    A_w = threshold_proportional(abs(dyn_FC(:,:,w)), 0.2);  % Keep top 20% connections
    
    % Calculate network metrics
    temporal_metrics.clustering(w) = mean(clustering_coef_wu(A_w));
    D_w = distance_wei(1./A_w);
    temporal_metrics.path_length(w) = charpath(D_w);
    [~, Q] = community_louvain(A_w);
    temporal_metrics.modularity(w) = Q;
end

% Plot temporal evolution of metrics
figure;
subplot(3,1,1);
plot(temporal_metrics.clustering, 'LineWidth', 2);
title('Clustering Coefficient Over Time');
xlabel('Window Number'); ylabel('Mean Clustering');
grid on;

subplot(3,1,2);
plot(temporal_metrics.path_length, 'LineWidth', 2);
title('Path Length Over Time');
xlabel('Window Number'); ylabel('Path Length');
grid on;

subplot(3,1,3);
plot(temporal_metrics.modularity, 'LineWidth', 2);
title('Modularity Over Time');
xlabel('Window Number'); ylabel('Modularity Q');
grid on;
```

#### 5.5.2 Dynamic Community Structure

Analyzing how communities evolve over time provides insights into brain flexibility:

```matlab
% Calculate community structure for each time window
modules = zeros(n_regions, n_windows);
for w = 1:n_windows
    A_w = threshold_proportional(abs(dyn_FC(:,:,w)), 0.2);
    [M, ~] = community_louvain(A_w);
    modules(:,w) = M;
end

% Calculate flexibility (frequency of community changes)
flexibility = zeros(n_regions, 1);
for r = 1:n_regions
    changes = sum(diff(modules(r,:)) ~= 0);
    flexibility(r) = changes / (n_windows - 1);
end

% Visualize node flexibility
figure;
bar(flexibility);
xlabel('Node Index'); ylabel('Flexibility');
title('Node Flexibility (Community Switching)');
```

#### 5.5.3 Temporal Motifs

Temporal motifs extend the concept of network motifs to include the timing and ordering of interactions:

```matlab
% Create an event list for temporal network analysis
% [node1, node2, time, duration]
% This requires specialized algorithms beyond BCT
```

### 5.6 Multimodal Network Integration

Combining different types of brain networks (structural, functional, effective) provides a more comprehensive understanding of brain organization.

#### 5.6.1 Structure-Function Relationships

A key question in neuroscience is how structural connectivity constrains and enables functional connectivity:

```matlab
% Assume we have structural connectivity (SC) and functional connectivity (FC) matrices
% SC: structural connectivity (e.g., from diffusion MRI)
% FC: functional connectivity (e.g., from fMRI)

% Calculate correlation between structural and functional connectivity
% First, extract upper triangular elements (excluding diagonal)
n = size(SC, 1);
mask = triu(ones(n), 1) > 0;  % Upper triangular mask

SC_vec = SC(mask);
FC_vec = FC(mask);

% Calculate correlation
[r, p] = corr(SC_vec, FC_vec, 'type', 'Spearman');
fprintf('Structure-function correlation: r = %.4f (p = %.4g)\n', r, p);

% Visualize relationship
figure;
scatter(SC_vec, FC_vec, 50, 'filled', 'MarkerFaceAlpha', 0.3);
xlabel('Structural Connectivity');
ylabel('Functional Connectivity');
title(sprintf('Structure-Function Relationship (r = %.4f)', r));
grid on;

% Fit line
hold on;
b = polyfit(SC_vec, FC_vec, 1);
x_range = linspace(min(SC_vec), max(SC_vec), 100);
y_fit = polyval(b, x_range);
plot(x_range, y_fit, 'r-', 'LineWidth', 2);
```

#### 5.6.2 Multi-Layer Network Analysis

Multi-layer networks represent different modalities or frequency bands as separate layers with inter-layer connections:

```matlab
% Create multi-layer adjacency matrix
% Assume we have connectivity matrices for different frequency bands
% alpha_FC, beta_FC, gamma_FC, etc.

% Create inter-layer connections (identity matrix for connecting the same node across layers)
n_nodes = size(alpha_FC, 1);
n_layers = 3;  % Three frequency bands
inter_layer = eye(n_nodes);

% Build multi-layer adjacency matrix
A_multi = zeros(n_nodes * n_layers);

% Add intra-layer connections
A_multi(1:n_nodes, 1:n_nodes) = alpha_FC;
A_multi(n_nodes+1:2*n_nodes, n_nodes+1:2*n_nodes) = beta_FC;
A_multi(2*n_nodes+1:3*n_nodes, 2*n_nodes+1:3*n_nodes) = gamma_FC;

% Add inter-layer connections
A_multi(1:n_nodes, n_nodes+1:2*n_nodes) = inter_layer;
A_multi(n_nodes+1:2*n_nodes, 1:n_nodes) = inter_layer;
A_multi(n_nodes+1:2*n_nodes, 2*n_nodes+1:3*n_nodes) = inter_layer;
A_multi(2*n_nodes+1:3*n_nodes, n_nodes+1:2*n_nodes) = inter_layer;

% Visualize multi-layer network
figure;
imagesc(A_multi);
colormap('jet');
title('Multi-layer Network');
```

## 6. Practical Examples

### 6.1 Comparing Network Metrics Between Groups

This example shows how to compare network metrics between two groups (e.g., patients vs. controls).

```matlab
% Assume we have adjacency matrices for two groups
% A_group1: array of adjacency matrices for group 1
% A_group2: array of adjacency matrices for group 2

% Initialize arrays to store metrics
CC1 = zeros(1, size(A_group1, 3));
CC2 = zeros(1, size(A_group2, 3));
PL1 = zeros(1, size(A_group1, 3));
PL2 = zeros(1, size(A_group2, 3));

% Calculate metrics for each subject in group 1
for i = 1:size(A_group1, 3)
    CC1(i) = mean(clustering_coef_wu(A_group1(:,:,i)));
    D = distance_wei(A_group1(:,:,i));
    PL1(i) = charpath(D);
end

% Calculate metrics for each subject in group 2
for i = 1:size(A_group2, 3)
    CC2(i) = mean(clustering_coef_wu(A_group2(:,:,i)));
    D = distance_wei(A_group2(:,:,i));
    PL2(i) = charpath(D);
end

% Statistical comparison (e.g., t-test)
[h_cc, p_cc] = ttest2(CC1, CC2);
[h_pl, p_pl] = ttest2(PL1, PL2);

fprintf('Clustering coefficient comparison: p = %.4f\n', p_cc);
fprintf('Path length comparison: p = %.4f\n', p_pl);
```

### 6.2 Visualizing Brain Networks

BCT itself doesn't provide visualization functions, but you can use other MATLAB toolboxes for visualization:

```matlab
% Example using the MATLAB Brain Connectivity Toolbox visualization
% Requires the 'BrainNet Viewer' or similar tools
% Assume 'coords' contains 3D coordinates of nodes

% Plot network
figure;
hold on;

% Plot nodes
scatter3(coords(:,1), coords(:,2), coords(:,3), 50, 'r', 'filled');

% Plot edges (connections)
for i = 1:size(A,1)
    for j = i+1:size(A,2)
        if A(i,j) > 0
            line([coords(i,1) coords(j,1)], ...
                 [coords(i,2) coords(j,2)], ...
                 [coords(i,3) coords(j,3)], ...
                 'Color', [0.5 0.5 0.5 A(i,j)], ...
                 'LineWidth', A(i,j)*3);
        end
    end
end

axis equal;
axis off;
view(3);
```

> **Citation**: Xia, M., Wang, J., & He, Y. (2013). BrainNet Viewer: A network visualization tool for human brain connectomics. PloS One, 8(7), e68910. https://doi.org/10.1371/journal.pone.0068910

## 7. Common Issues and Solutions

### 7.1 Memory Issues with Large Networks

When dealing with large networks, you may encounter memory limitations. Possible solutions include:

- **Using sparse matrices**: `A = sparse(A)`
  - *How it works*: Stores only non-zero elements and their indices, drastically reducing memory footprint
  - *Memory savings*: For a network with 10% density, sparse matrices can reduce memory usage by approximately 90%
  - *Implementation*:
    ```matlab
    % Convert dense matrix to sparse
    A_sparse = sparse(A);
    
    % Check memory usage difference
    whos A A_sparse
    ```
  - *BCT compatibility*: Most BCT functions automatically handle sparse matrices
  - *Performance note*: Some operations are faster on sparse matrices, others are slower

- **Processing networks in smaller chunks**
  - *Approach 1*: Analyze subnetworks separately
    ```matlab
    % Example: Process network in two halves
    n = size(A, 1);
    half = ceil(n/2);
    
    % First half
    A1 = A(1:half, 1:half);
    results1 = your_analysis_function(A1);
    
    % Second half
    A2 = A(half+1:end, half+1:end);
    results2 = your_analysis_function(A2);
    ```
  - *Approach 2*: Use memory-efficient algorithms
    ```matlab
    % Example: Calculate betweenness centrality in batches
    n = size(A, 1);
    batch_size = 100;
    BC = zeros(n, 1);
    
    for i = 1:batch_size:n
        end_idx = min(i+batch_size-1, n);
        BC(i:end_idx) = betweenness_bin_batch(A, i:end_idx);
    end
    ```
  - *Caveat*: Not all network measures can be calculated on partial networks

- **Increasing MATLAB memory allocation**: `memory -java 4g` (4GB example)
  - *System requirements*: Your computer must have sufficient physical RAM
  - *Alternative approach*: Edit MATLAB's preferences file directly
    ```
    MATLAB → Preferences → General → Java Heap Memory
    ```
  - *Common memory errors*:
    - "Out of memory."
    - "Maximum variable size allowed by the program is exceeded."
  - *Typical memory requirements*:
    - 1000-node network: ~8GB RAM
    - 10,000-node network: ~64GB RAM or more

**Real-world example:**
For a voxel-level network (e.g., 50,000 nodes), calculating all shortest paths would require storing a 50,000 × 50,000 matrix, which needs approximately 20GB of RAM just for one matrix. Strategies for handling such networks:

1. Use sparse matrix representations
2. Apply more aggressive parcellation to reduce network size
3. Consider dimensionality reduction techniques
4. Use specialized "big data" network analysis tools
5. Analyze only specific subnetworks of interest

### 7.2 Handling Negative Weights

Many BCT functions are designed for positive weights only. For functional connectivity with negative correlations:

- **Option 1: Use absolute values** `A_abs = abs(A)`
  - *Pros*: Simple, preserves connection strength
  - *Cons*: Loses information about correlation direction, alters network topology
  - *When to use*: When you're primarily interested in connection strength regardless of direction
  - *Implementation*:
    ```matlab
    % Convert to absolute values
    A_abs = abs(A);
    
    % Calculate measures
    C_abs = clustering_coef_wu(A_abs);
    E_abs = efficiency_wei(A_abs);
    ```
  - *Interpretation challenge*: Anticorrelations are treated as strong positive correlations

- **Option 2: Separate positive and negative networks**
  ```matlab
  A_pos = A .* (A > 0);
  A_neg = abs(A .* (A < 0));
  ```
  - *Pros*: Preserves distinction between positive and negative correlations
  - *Cons*: Doubles the number of analyses, may reduce statistical power
  - *When to use*: When the sign of the correlation carries important information
  - *Implementation*:
    ```matlab
    % Separate positive and negative components
    A_pos = A .* (A > 0);  % Keep only positive values
    A_neg = abs(A .* (A < 0));  % Convert negative values to positive
    
    % Analyze separately
    C_pos = clustering_coef_wu(A_pos);
    C_neg = clustering_coef_wu(A_neg);
    ```
  - *Interpretation approach*: Compare results between positive and negative networks

- **Option 3: Use functions specifically designed for signed networks** (e.g., `clustering_coef_wu_sign`)
  - *Pros*: Theoretically sound, preserves network topology
  - *Cons*: Limited number of available measures
  - *When to use*: When both positive and negative correlations are important and theoretical foundation exists
  - *Implementation*:
    ```matlab
    % Calculate signed clustering coefficient
    C_sign = clustering_coef_wu_sign(A);
    
    % Calculate positive-weighted/negative-weighted components
    [C_pos, C_neg] = clustering_coef_wu_sign(A);
    ```
  - *Theoretical basis*: Positive weights indicate similarity/cooperation, negative weights indicate dissimilarity/competition

**Neurobiological context:**
Negative correlations in functional connectivity often represent:
- Opponent neural systems (e.g., task-positive vs. default mode networks)
- Inhibitory connections
- Complementary processing (when one system is active, another is suppressed)

> **Citation**: Rubinov, M., & Sporns, O. (2011). Weight-conserving characterization of complex functional brain networks. NeuroImage, 56(4), 2068-2079. https://doi.org/10.1016/j.neuroimage.2011.03.069

### 7.3 Choosing Appropriate Network Thresholds

Thresholding is a critical step that can influence results:

- **Density-based thresholding**: Retain a fixed percentage of strongest connections
  ```matlab
  density = 0.1;  % 10% density
  A_thresh = threshold_proportional(A, density);
  ```
  - *How it works*: Sorts all connection weights and keeps the top X% strongest ones
  - *Advantages*: 
    - Ensures equal edge count across subjects/conditions
    - Controls for overall differences in connectivity strength
    - Makes group comparisons more straightforward
  - *Disadvantages*:
    - Different absolute threshold values across subjects
    - May include noise in subjects with overall weaker connectivity
  - *Recommended densities*: Try multiple values (e.g., 5%, 10%, 15%, 20%, 25%)

- **Absolute thresholding**: Retain connections above a specific value
  ```matlab
  threshold = 0.3;
  A_thresh = threshold_absolute(A, threshold);
  ```
  - *How it works*: Only keeps connections with weights greater than the specified value
  - *Advantages*:
    - Consistent meaning of connection strength across subjects
    - Theoretically justifiable (e.g., only correlations with p < 0.05)
  - *Disadvantages*:
    - Results in different network densities across subjects/conditions
    - May bias group comparisons if overall connectivity differs
  - *Selecting absolute thresholds*:
    - For correlation matrices: Common values are 0.2, 0.3, 0.4
    - For significance-based thresholds: Use p-values (e.g., p < 0.05 with FDR correction)

- **Analyze across multiple thresholds to ensure robustness of findings**
  ```matlab
  % Example: Analyze across multiple densities
  densities = [0.05, 0.10, 0.15, 0.20, 0.25];
  modularity_values = zeros(size(densities));
  
  for i = 1:length(densities)
      A_thresh = threshold_proportional(A, densities(i));
      [~, modularity_values(i)] = community_louvain(A_thresh);
  end
  
  % Plot results across thresholds
  figure;
  plot(densities, modularity_values, 'o-', 'LineWidth', 2);
  xlabel('Network Density');
  ylabel('Modularity');
  title('Modularity Across Network Densities');
  ```
  - *Strategy*: Report results that are consistent across a range of thresholds
  - *Statistical approaches*: 
    - Area under the curve (AUC) across thresholds
    - Peak values and their corresponding thresholds
    - Statistical testing at each threshold with correction for multiple comparisons

**Threshold selection framework for beginners:**

1. **Determine your primary research question**
   - Group comparison? → Consider density-based thresholding
   - Absolute connectivity strength matters? → Consider absolute thresholding
   - Exploratory analysis? → Use multiple thresholding approaches

2. **Consider network type**
   - Structural connectivity → Often sparser (5-15% density)
   - Functional connectivity → Often denser (10-30% density)
   - Effective connectivity → Typically uses statistical thresholds

3. **Apply multiple thresholds**
   - Test a range of values rather than a single threshold
   - Verify that key findings are robust across thresholds

4. **Document threshold selection**
   - Clearly report thresholding approach in methods
   - Explain rationale for chosen thresholds
   - Show results across thresholds when possible

**Common pitfalls to avoid:**
- Using a single arbitrary threshold without justification
- Selecting thresholds post-hoc to maximize desired effects
- Failing to account for different densities when comparing groups
- Not considering the biological meaning of the chosen threshold

> **Citation**: van Wijk, B. C., Stam, C. J., & Daffertshofer, A. (2010). Comparing brain networks of different size and connectivity density using graph theory. PloS One, 5(10), e13701. https://doi.org/10.1371/journal.pone.0013701

### 7.4 Dealing with Missing or Noisy Data

Neuroimaging data often contains artifacts, noise, or missing regions. Here are strategies to handle these issues:

- **Handling missing nodes**
  - *Problem*: Some subjects may have regions with no data due to signal dropout or lesions
  - *Solution 1*: Restrict analysis to nodes present in all subjects
    ```matlab
    % Identify nodes with missing data across subjects
    missing_mask = any(isnan(A), [1, 2]);  % For single matrix
    common_nodes = find(~missing_mask);
    
    % Restrict analysis to common nodes
    A_common = A(common_nodes, common_nodes);
    ```
  - *Solution 2*: Impute missing values
    ```matlab
    % Simple imputation using mean of non-missing connections
    A_imputed = A;
    [i, j] = find(isnan(A));
    for idx = 1:length(i)
        row = i(idx);
        col = j(idx);
        % Mean of non-missing values in same row and column
        row_vals = A(row, ~isnan(A(row,:)));
        col_vals = A(~isnan(A(:,col)), col);
        A_imputed(row, col) = mean([row_vals, col_vals']);
    end
    ```

- **Managing motion artifacts in functional connectivity**
  - *Problem*: Head motion creates spurious correlations, especially in short-range connections
  - *Solution 1*: Strict motion regression
    - Include 24 motion parameters (6 direct + their derivatives + their squares)
    - Include spike regressors for high-motion volumes
  - *Solution 2*: Motion scrubbing/censoring
    - Remove high-motion volumes before computing correlations
  - *Solution 3*: Group matching and covariate analysis
    - Ensure groups have similar motion profiles
    - Include motion metrics as covariates in statistical analyses

- **Handling poor quality scans**
  - *Assessment metrics*:
    - Temporal SNR for fMRI
    - Mean framewise displacement for motion
    - FA baseline for diffusion imaging
  - *Decision framework*:
    - Set objective quality thresholds for inclusion
    - Document excluded subjects and reasons for exclusion
    - Consider sensitivity analyses with and without borderline cases

### 7.5 Statistical Analysis of Network Measures

Network measures often violate assumptions of common statistical tests. Here are robust approaches:

- **Comparing network metrics between groups**
  - *Challenge*: Network metrics are often non-normally distributed
  - *Solution 1*: Non-parametric tests
    ```matlab
    % Example: Mann-Whitney U-test for between-group comparison
    [p, ~, stats] = ranksum(group1_clustering, group2_clustering);
    fprintf('Group difference in clustering: z = %.2f, p = %.4f\n', stats.zval, p);
    ```
  - *Solution 2*: Permutation testing
    ```matlab
    % Permutation test for group differences
    n_perm = 10000;
    actual_diff = mean(group1_metric) - mean(group2_metric);
    perm_diffs = zeros(n_perm, 1);
    
    combined = [group1_metric; group2_metric];
    n1 = length(group1_metric);
    n2 = length(group2_metric);
    
    for i = 1:n_perm
        perm_idx = randperm(n1 + n2);
        perm_group1 = combined(perm_idx(1:n1));
        perm_group2 = combined(perm_idx(n1+1:end));
        perm_diffs(i) = mean(perm_group1) - mean(perm_group2);
    end
    
    p_value = sum(abs(perm_diffs) >= abs(actual_diff)) / n_perm;
    ```

- **Controlling for multiple comparisons**
  - *Problem*: Testing multiple brain regions or network measures inflates Type I error
  - *Solution 1*: False Discovery Rate (FDR) correction
    ```matlab
    % FDR correction for multiple comparisons
    [~, ~, ~, p_fdr] = fdr_bh(p_values);
    ```
  - *Solution 2*: Network-based statistic (NBS)
    - Tests for connected components of differences rather than individual connections
    - Available at: [https://www.nitrc.org/projects/nbs/](https://www.nitrc.org/projects/nbs/)

- **Accounting for dependencies between network measures**
  - *Problem*: Network measures are often highly correlated with each other
  - *Solution 1*: Multivariate approaches
    ```matlab
    % Example: MANCOVA with network metrics
    X = [clustering_coef, path_length, modularity];  % Network measures
    group = [ones(n1,1); 2*ones(n2,1)];  % Group labels
    covariates = [age, sex, motion];  % Potential confounds
    
    [d, p] = manova1([X, covariates], group);
    ```
  - *Solution 2*: Principal Component Analysis
    ```matlab
    % Reduce dimensionality of network measures
    [coeff, score, ~, ~, explained] = pca(X);
    
    % Use first few components that explain most variance
    cum_var = cumsum(explained);
    n_components = find(cum_var > 80, 1);  % Components explaining 80% variance
    X_reduced = score(:, 1:n_components);
    ```

## 8. Further Resources

### 8.1 BCT Documentation

- Official BCT website: [https://sites.google.com/site/bctnet/](https://sites.google.com/site/bctnet/)
- Function reference: [https://sites.google.com/site/bctnet/measures/list](https://sites.google.com/site/bctnet/list-of-measures?authuser=0)

**Documentation Quick Reference:**

| Category | Key Functions | Description |
|----------|---------------|-------------|
| Clustering | `clustering_coef_bu`, `clustering_coef_wu` | Calculate clustering coefficients |
| Paths | `distance_bin`, `distance_wei`, `charpath` | Calculate path lengths and efficiency |
| Centrality | `betweenness_bin`, `eigenvector_centrality_und` | Identify important nodes |
| Community | `community_louvain`, `modularity_und` | Detect network modules |
| Random Networks | `randmio_und`, `randmio_dir` | Generate random networks |
| Visualization | *not included* | Use external tools |

**Interactive BCT Explorer**:
- An unofficial interactive web tool for exploring BCT functions and their relationships: [BCT Explorer](https://example.com/bct-explorer) (Note: This is a hypothetical resource)

### 8.2 Books and Reviews

- **Fornito, A., Zalesky, A., & Bullmore, E. (2016). Fundamentals of Brain Network Analysis. Academic Press.**
  - *Comprehensive textbook covering all aspects of brain network analysis*
  - *Includes detailed explanations of graph theory concepts and their neurobiological interpretations*
  - *Provides practical examples and case studies*
  - *Recommended chapters for beginners*: 1-4 (fundamentals), 6 (functional networks), 7 (structural networks)

- **Bassett, D. S., & Sporns, O. (2017). Network neuroscience. Nature Neuroscience, 20(3), 353-364.**
  - *Accessible review of network neuroscience principles*
  - *Covers recent advances and emerging directions*
  - *Good starting point for understanding the field's scope*

- **Bullmore, E., & Sporns, O. (2009). Complex brain networks: Graph theoretical analysis of structural and functional systems. Nature Reviews Neuroscience, 10(3), 186-198.**
  - *Classic review that helped establish network neuroscience*
  - *Explains fundamental concepts and early findings*
  - *Provides historical context for the field*

**Additional Essential Reading:**

- **Sporns, O. (2018). Graph theory methods: applications in brain networks. Dialogues in Clinical Neuroscience, 20(2), 111-121.**
  - *Concise introduction to graph theory applications in neuroscience*
  - *Accessible to clinical researchers and beginners*

- **van den Heuvel, M. P., & Hulshoff Pol, H. E. (2010). Exploring the brain network: a review on resting-state fMRI functional connectivity. European Neuropsychopharmacology, 20(8), 519-534.**
  - *Focused on functional connectivity analysis methods*
  - *Discusses clinical applications*

- **Rubinov, M., & Sporns, O. (2010). Complex network measures of brain connectivity: uses and interpretations. NeuroImage, 52(3), 1059-1069.**
  - *The original BCT paper, explaining the theoretical foundation*
  - *Details network measures and their interpretations*

### 8.3 Related Toolboxes

- **Network Based Statistic (NBS)**: Statistical testing for brain networks
  [https://sites.google.com/site/bctnet/comparison/nbs](https://www.nitrc.org/projects/nbs/)
  - *Purpose*: Statistical testing of network differences while controlling for multiple comparisons
  - *Advantages*: More sensitive than mass-univariate testing with FDR/FWE correction
  - *Integration with BCT*: Uses similar network formats, easy pipeline integration
  - *Example use case*: Identifying connection patterns that differ between patients and controls
  
- **Brain Connectivity Toolbox for Python (bctpy)**:
  [https://github.com/aestrivex/bctpy](https://github.com/aestrivex/bctpy)
  - *Purpose*: Python implementation of BCT functions
  - *Advantages*: Integration with Python ecosystem (NumPy, SciPy, scikit-learn)
  - *Usage*: Nearly identical function names and parameters as MATLAB version
  - *Example use case*: Building automated network analysis pipelines in Python

- **GraphVar**: GUI-based graph analysis of brain connectivity
  [https://www.nitrc.org/projects/graphvar/](https://www.nitrc.org/projects/graphvar/)
  - *Purpose*: GUI for network construction, analysis, and statistics
  - *Advantages*: User-friendly interface, no programming required
  - *Features*: Network visualization, statistical testing, multiple comparison correction
  - *Example use case*: Exploratory analysis of network differences between conditions

**Additional Valuable Resources:**

- **Connectome Computation System (CCS)**:
  [https://github.com/zuoxinian/CCS](https://github.com/zuoxinian/CCS)
  - *Purpose*: Comprehensive pipeline for functional and structural connectome analysis
  - *Features*: Preprocessing, network construction, analysis, visualization
  
- **CONN Toolbox**:
  [https://web.conn-toolbox.org/](https://web.conn-toolbox.org/)
  - *Purpose*: Functional connectivity analysis and network construction
  - *Features*: Denoising, first-level and second-level connectivity analysis
  
- **BrainNet Viewer**:
  [https://www.nitrc.org/projects/bnv/](https://www.nitrc.org/projects/bnv/)
  - *Purpose*: Visualization of brain networks
  - *Features*: 3D visualizations of nodes, edges, and network metrics

### 8.4 Online Learning Resources

- **Tutorials and Workshops**:
  - **OHBM Educational Courses**: Annual courses covering network neuroscience (slides often available online)
  - **Neurohackademy**: Network neuroscience tutorials and practical sessions
  - **Coursera - Network Science**: Not brain-specific but covers fundamental graph theory concepts

- **Video Lectures**:
  - [Olaf Sporns: Introduction to Network Neuroscience](https://www.youtube.com/watch?v=aOWDZI6W2qM) (YouTube)
  - [Danielle Bassett: Network Neuroscience](https://www.youtube.com/watch?v=NODk718Fk6A) (YouTube)

- **Code Examples and Tutorials**:
  - GitHub repositories with example analyses
  - Neurostars.org for community questions and answers
  - Open Neuro datasets with network analysis examples

### 8.5 Sample Datasets for Practice

- **Public Brain Connectivity Datasets**:
  - **Human Connectome Project (HCP)**: High-quality structural and functional connectivity data
  - **ABIDE (Autism Brain Imaging Data Exchange)**: Autism and control connectivity data
  - **SchizConnect**: Schizophrenia neuroimaging data

- **Preprocessed Connectivity Matrices**:
  - USC Multimodal Connectivity Database
  - BCT website example datasets
  - Open Connectome Project
  