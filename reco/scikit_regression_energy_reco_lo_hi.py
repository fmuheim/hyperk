import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model, ensemble
from sklearn.cross_validation import cross_val_score
import xgboost as xgb
#import sys
#sys.argv.append( '-b-' )
import ROOT
ROOT.gROOT.SetBatch(True)
from root_numpy import root2array, tree2array, fill_hist
import os


chain = ROOT.TChain('nu_eneNEW')
chain.Add('/Disk/ds-sopa-group/PPE/titus/ts-WChRecoSandBox/scripts/editing_ene/outputs/nu_numu_1000_1039_CCQE_12in_energy_studies_recoquant_tree.root')
for i in range(1040,1099):
    input_file = '/Disk/ds-sopa-group/PPE/titus/ts-WChRecoSandBox/scripts/editing_ene/outputs/nu_numu_'+str(i)+'_CCQE_12in_energy_studies_recoquant_tree_NEWlookupsB_for_training.root'
    if os.path.exists(input_file):
        chain.Add(input_file)
data = tree2array(chain)
data_reduced   = data[['total_hits2','total_ring_PEs2','recoDWallR2','recoDWallZ2','hits_pot_length2','lambda_max_2']]#,'hits_pot_length2']]
data_reduced_n = data_reduced.view(data_reduced.dtype[0]).reshape(data_reduced.shape + (-1,))
data_target    = data['trueKE']

# Training the ML algo
#clf = linear_model.SGDRegressor()
clf = xgb.XGBRegressor(max_depth=4, learning_rate=0.1, subsample=0.5, n_estimators=800)
scores = cross_val_score(clf, data_reduced_n, data_target, cv=5)
print scores
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


'''
f, ax2 = plt.subplots(1, 1)
ax2.scatter(test_data_trueKE_hi_E,clf_hi_E.predict(test_data_reduced_hi_E_n)-test_data_trueKE_hi_E)
ax2.set_ylim((-5000,1000))
ax2.set_xlabel("trueKE [MeV]")
ax2.set_ylabel("recoKE - trueKE [MeV]")
plt.savefig("comparision.pdf")
'''
