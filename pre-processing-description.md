specXplore is run with two input files:

1) An experimental mgf file containing experimental spectra.
2) A second mgf file containing reference standards with known structure.

The pre-processing for specxplore happens in multiple steps. 

# Step 1

The raw spectral data are cleaned. An updated version of each file is stored separately.

# Step 2

The spectral data are annotated. For the experimental data, ms2query is used to create a results csv file that contains analog matches and chemical classifications of the latter. The analog chemical classifications will be used in the tool as proxies for chemical classes of the unknown spectrum. 

The reference standard spectra are assumed to have inchi entries. Those are used to query GNPS libraries for chemical classification ontology using both classyfire and npclassifier. This step can take quite a while.

# Step 3

spectra lists are merged. Each spectrum remains in their original order, and received an integer location identifier. 

# Step 4

classifications tables and any additional metadata from csv files are merged. Rows correspond to spectrum integer location identifiers that are added as identifier column. is_standard identifiers are added to the standards for further use inside specXplore.

# Step 5

The spectra list of step 3 is used to generate pairwise similarity matrices.

# Step 6

K-medoid clustering is performed to provide a means of generating clusters akin to molecular families. 

# Step 7 

t-SNE embedding is performed to provide emebedding coordinates.

# Step 6 

All data is stored in a specXplore file format. The classyfire and npclassifier columns are extracted into a classes column for visualization use. Other classification schemes may also be used.
