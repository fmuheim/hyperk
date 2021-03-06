import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model, ensemble
from sklearn.model_selection import cross_val_score, cross_val_predict
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
data_target    = data['trueKE']/1e3

# Training the ML algo
#clf = linear_model.SGDRegressor()
clf = xgb.XGBRegressor(max_depth=4, learning_rate=0.1, n_estimators=400)
scores = cross_val_score(clf, data_reduced_n, data_target, cv=5, scoring='neg_mean_squared_error')
print scores
print("MSE: %0.3f (+/- %0.3f)" % (scores.mean(), scores.std() * 2))

predicted = cross_val_predict(clf, data_reduced_n, data_target, cv=5)

from sklearn import metrics
scores2 = metrics.mean_squared_error(data_target, predicted)
print scores
print("MSE 2: %0.3f (+/- %0.3f)" % (scores2.mean(), scores2.std() * 2))

fig, ax = plt.subplots()
ax.scatter(data_target, predicted)
ax.plot([data_target.min(), data_target.max()], [data_target.min(), data_target.max()], 'k--', lw=4)
ax.set_xlabel('True KE [GeV]')
ax.set_ylabel('Predicted KE [GeV]')
ax.set_xlim([data_target.min()*0.9, data_target.max()*1.1])
ax.set_ylim([data_target.min()*0.9, data_target.max()*1.1])
plt.savefig("xgb_cross_val_comparision.pdf")

ROOT.gStyle.SetOptStat(1)
hist = ROOT.TH1D("hist","hist", 100, -1, 1)
diff = (data_target - predicted)/data_target
fill_hist(hist, diff)
hist.GetXaxis().SetTitle("#DeltaE/E")
canvas = ROOT.TCanvas()
hist.Draw()
canvas.SaveAs("xgb_cross_val_DeltaE.pdf")
