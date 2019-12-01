# ERGO :construction:

## Report 27.11.19 :green_apple:
### 
### Evaluation Methods
We suggest several evaluation methods, all for the same trained model (for every model type and dataset).
1. AUC Per peptide: AUC score of positive (TCRs binding this peptide)
and negative (TCRs that do not bind it) test samples paired with a specific peptide. 
2. Multiclass peptide classification of unseen TCRs: We first take the N most
frequent peptides in the test samples. For every 2≤k≤N, we get the TCRs in the test
set that bind one of the first k peptides, and that were not seen during training.
We classify those TCRs to one of the k peptides by taking argmax of k scores from the model.
3. Unseen pairs AUC: This is the original evaluation. We check for AUC score of test pairs.
4. Unseen TCRs AUC: We check for AUC score of test pairs that contain TCRs that were not seen during training.
5. Unseen TCRs and peptides AUC: We check for AUC score of test pairs that contain
TCRs and peptides that were not seen during training.

### Results for LSTM, McPAS model (not so optimized)
#### AUC Per peptide:
Peptide | AUC
--- | ---
LPRRSGAAGA |	0.741
GILGFVFTL |	0.779
GLCTLVAML |	0.764
NLVPMVATV |	0.775
SSLENFRAYV |	0.809

#### Multiclass peptide classification of unseen TCRs:
Peptide |	Accuracy
--- | ---
LPRRSGAAGA+GILGFVFTL|	0.694
+GLCTLVAML|	0.565
+NLVPMVATV|	0.496
+SSLENFRAYV|	0.485

Unseen pairs AUC: **0.824**

Unseen TCRs AUC: **0.777**

Unseen TCRs and peptides AUC: **None (bug)**

## Report 28.11.19 :red_car:
### Weighted loss
Since that the number of negative samples was equal to the number of positive examples,
we were over-sampling frequent peptides.
Now there are more negative examples than positive ones,
so we multiply the loss of positive samples by a factor. 
#### Results for LSTM, McPAS model (not so optimized)
##### AUC Per peptide:
Peptide | AUC
--- | ---
LPRRSGAAGA |	0.772
GILGFVFTL |	0.806
GLCTLVAML |	0.804
NLVPMVATV |	0.800
SSLENFRAYV |	0.925

##### Multiclass peptide classification of unseen TCRs:
Peptides |	Accuracy
--- | ---
LPRRSGAAGA+GILGFVFTL|	0.645
+GLCTLVAML|	0.523
+NLVPMVATV|	0.464
+SSLENFRAYV|	0.486

Unseen pairs AUC: **0.860**

Unseen TCRs AUC: **0.801**

Unseen TCRs and peptides AUC: **0.511**
### Peptides list
Our goal is to get SOTA results in the single peptide AUC evaluation.
We would like to compare previous methods.

Glanville peptides:

Peptide | AUC
--- | ---
VTEHDTLLY| 0.792
CTELKLSDY| none
NLVPMVATV| 0.800
GLCTLVAML| 0.804
GILGFVFTL| 0.806
TPRVTGGGAM| 0.786
LPRRSGAAGA| 0.772

Dash peptides (mentioned in TCRGP table):
* GLCTLVAML
* NLVPMVATV
* GILGFVFTL