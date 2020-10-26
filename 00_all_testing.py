from glob import glob
import joblib
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix as cm
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import plot_roc_curve
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import plot_precision_recall_curve
from sklearn.metrics import auc

import warnings
warnings.filterwarnings("ignore")

def makeFeatures(file):
	f = open(file,"r")
	data = f.readline().split("\t")[:-1]
	f.close()
	return data


if __name__ == "__main__":
	folders = glob("../all_with*/")
	modelList = None
	model = None
	testSet = None
	features = None
	X1 = None
	y1 = None
	test = None
	separator = "\t"
	nTrees = ["500","1000","1500","2000"]
	for folder in folders:
		features = makeFeatures(folder+"/K562/all_one_title.tsv")
		X1 = pd.read_csv(folder+"/K562/all_one_title.tsv", sep = separator, dtype=np.float32, usecols=features)
		y1 = pd.read_csv(folder+"/K562/all_one_title.tsv", sep = separator, dtype=np.float32, usecols=["promoter"])
		for t in nTrees:
			fig, ax = plt.subplots()
			ax.set_aspect('equal', adjustable='box')
			fig.set_size_inches(11,8)
			print("Entering to folder: "+folder)
			print("\tloading K562 test set")
			features = makeFeatures(folder+"/K562/all_one_title.tsv")
			print("\tLooking for models")
			modelList = glob(folder+"sorted_with_no*_ntree_"+t+".pkl")

			#ROC
			roc = None
			mean_fpr = np.linspace(0, 1, 100)
			tprs = []
			aucs = []
			for m in modelList:
				print("\t\tloading model: "+m)
				fragment = folder.split("_")[-1]
				without = m.split("sorted_with_no")[1].split("_")[0]
				tree = m.split("ntree_")[1].split(".")[0]
				name = "without "+without+" "
				model = joblib.load(m)
				roc = plot_roc_curve(model, X1, y1, ax = ax, name = name, alpha = .3)

				#ROC
				interp_tpr = np.interp(mean_fpr, roc.fpr, roc.tpr)
				interp_tpr[0] = 0.0
				tprs.append(interp_tpr)
			print("Exiting of folder: "+folder)

			ax.set_title("ROC curves", fontsize = 14) 
			ax.plot([0, 1], [0, 1], linestyle='--', lw=1, color='r', label='Chance', alpha=.8)
			mean_tpr = np.mean(tprs, axis=0)
			mean_tpr[-1] = 1.0
			mean_auc = auc(mean_fpr, mean_tpr)
			std_auc = np.std(aucs)
			ax.plot(mean_fpr, mean_tpr, color='b', label=r'Mean ROC (AUC = %0.2f)' % (mean_auc), lw=1, alpha=.8)
			std_tpr = np.std(tprs, axis=0)
			tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
			tprs_lower = np.maximum(mean_tpr - std_tpr, 0)

			ax.legend()

			print("saving figure...")
			plt.savefig(folder[3:-1]+"_ntree_"+t+"_roc.jpg")
		
