from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import warnings
import xlrd #读取excel的库
warnings.filterwarnings("ignore")

plt.rcParams['figure.dpi']=200
plt.rcParams['savefig.dpi']=200
font = {'family': 'Times New Roman',
        'weight': 'light'}
plt.rc("font", **font)

#Section 1: Load Breast data, i.e., Benign and Malignant
resArray=[] #先声明一个空list
# Load the dataset
data = xlrd.open_workbook(".//denoised50epochs(CycleGAN)summary.xlsx") #读取文件
table = data.sheet_by_index(0) #按索引获取工作表，0就是工作表1
normArray=np.random.rand(table.nrows,105)
for i in range(table.nrows): #table.nrows表示总行数
    line=table.row_values(i) #读取每行数据，保存在line里面，line是list
    for j in range(len(line)):
        line[j]=float(line[j])
    resArray.append(line) #将line加入到resArray中，resArray是二维list
resArray=np.array(resArray) #将resArray从二维list变成数组

for j in range(104):
    #print(len(resArray[:,j]))
    normmin=min(resArray[:,j])
    normmax=max(resArray[:,j])
    normArray[:,j]=(resArray[:,j]-normmin)/(normmax-normmin)
normArray[:,104]=resArray[:,104]
np.random.shuffle(normArray)

X = normArray[:,0:103]
y = normArray[:,104]
X_train,X_test,y_train,y_test=\
    train_test_split(X,y,test_size=0.2,stratify=y,random_state=1)

#Section 2: Construct model optimized via GridSearch
pipe_svc=make_pipeline(StandardScaler(),
                       SVC(random_state=1))

#Section 3: Optimize SVC model via Grid Search
param_range=[0.0001,0.001,0.01,0.1,1.0,10.0,100.0,1000]
param_grid=[{'svc__C':param_range,'svc__kernel':['linear']},
            {'svc__C':param_range,'svc__kernel':['rbf'],
             'svc__gamma':param_range}]

gs=GridSearchCV(estimator=pipe_svc,
                param_grid=param_grid,
                scoring='roc_auc',
                cv=2)

scores=cross_val_score(gs,X_train,y_train,scoring='roc_auc',cv=5)
print("CV AUC in Train Phase: %.3f +/- %.3f" % (np.mean(scores),np.std(scores)))

gs.fit(X_train,y_train)
print("AUC in Train Phase: %.3f" % gs.best_score_)
