Predicting zoonotic potential with DNA sequences

For the current best models, select a model from the curr_models section and test it out for yourself with the metrics at the bottom of validate.ipynb.

**REMINDER: data was transformed in the way that GBM and XGBoost was fit in order for it to perform properly.

Also - some data is gitignored for the purposes of conserving storage - you should be able to generate the sequences folder, as well as downloading virome_contigs and virome_reads from: http://www.hli-opendata.com/Virome/

Used ncbi-genome-download for quick downloads from NCBI

ncbi-genome-download --taxids virus_tax_ids.txt viral

for dataset 1: 
ncbi-genome-download --taxids ids-0.txt viral --parallel 4