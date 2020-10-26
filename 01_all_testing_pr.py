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

			#PR
			pr = None
			mean_recall = np.linspace(0,1, 100)
			recalls = []
			aucs_pr = []
			for m in modelList:
				print("\t\tloading model: "+m)
				fragment = folder.split("_")[-1]
				without = m.split("sorted_with_no")[1].split("_")[0]
				tree = m.split("ntree_")[1].split(".")[0]
				name = "without "+without+" "
				model = joblib.load(m)
				pr = plot_precision_recall_curve(model, X1, y1, ax=ax, name = name, alpha = .3)

				#PR
				interp_pr = np.interp(mean_recall, list(pr.recall)[::-1], list(pr.precision)[::-1])
				interp_pr[0] = 1.0
				recalls.append(interp_pr)
			print("Exiting of folder: "+folder)

			ax.set_title("PR curves", fontsize = 14)
			mean_pr = np.mean(recalls, axis=0)
			mean_pr[0] = 1.0
			mean_auc_pr = auc(mean_recall, mean_pr)
			std_pr = np.std(recalls)
			prs_upper = np.minimum(mean_recall + std_pr, 1)
			prs_lower = np.maximum(mean_recall - std_pr, 0)
			ax.plot(mean_recall, mean_pr, color='b', label=r'Mean PR (AP = %0.2f)' % (mean_auc_pr),lw=1, alpha=.8)
			ax.legend()

			print("saving figure...")
			plt.savefig(folder[3:-1]+"_ntree_"+t+"_pr.jpg")
		
