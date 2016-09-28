
import ROOT
rfile = ROOT.TFile('/Disk/ds-sopa-group/PPE/titus/ts-WChRecoSandBox/scripts/editing_ene/outputs/nu_numu_1000_1039_CCQE_12in_energy_studies_recoquant_tree.root')
intree = rfile.Get('nu_eneNEW')
from root_numpy import root2array, tree2array
arr=tree2array(intree)

arr2=arr[['total_hits2','total_ring_PEs2','recoDWallR2','recoDWallZ2','pot_length2','hits_pot_length2','lambda_max_2']]
arr2_n=arr2.view(arr2.dtype[0]).reshape(arr2.shape + (-1,))

arr3=arr['trueKE']

from sklearn import linear_model
clf = linear_model.SGDRegressor()
clf.fit(arr2_n[:50],arr3[:50])

import matplotlib.pyplot as plt
plt.plot(arr3[:50],clf.predict(arr2_n[:50]))
plt.show()
