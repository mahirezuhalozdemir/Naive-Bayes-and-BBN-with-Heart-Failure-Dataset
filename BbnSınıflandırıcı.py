import pandas as pd
import os
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator, BayesianEstimator
from sklearn.model_selection import train_test_split
from pgmpy.inference import VariableElimination
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

#sütun başlıklarını "A,B,C,D,E,F,G,H,I,J,K,L,RESULT" olarak güncelleyelim
names="A,B,C,D,E,F,G,H,I,J,K,L,RESULT"
names=names.split(",")

#dosyamızı açalım
os.chdir('C:\\Users\\zuhal\\Desktop\\bioinformaticProjects')
# Veri setini yükleyelim
data = pd.read_csv('heart_failure_clinical_records_dataset.csv',header=0,names=names)

#print(data.dtypes)

#data type float64'leri int64'e çevirdik
data["A"] = data["A"].astype('int64')
data["G"] = data["G"].astype('int64')
data["H"] = data["H"].astype('int64')


#print(data.head())

#verilerden result sonucunu çıkardık
X = data.drop(columns=["RESULT"])
y = data["RESULT"]
#print(X)
#print(y)

#verilerden train ve test datalarını ayırıyoruz
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#print(X_train)
#print(y_test)
#print(train_data)
#print(test_data)

model = BayesianNetwork([('A', 'B'), ('B', 'C'),('C','D'),('D','E'),('E','RESULT')])
train_data = pd.concat([pd.DataFrame(X_train), pd.DataFrame(y_train)], axis=1)
model.fit(train_data,estimator=BayesianEstimator)


infer=VariableElimination(model)
test_data = pd.concat([pd.DataFrame(X_test), pd.DataFrame(y_test)], axis=1)
y_pred = []

print("Nodes: ", model.nodes())
print("Edges: ", model.edges())


y_pred=[]
for index, row in test_data.iterrows():
    q = infer.query(variables=["RESULT"], evidence={"A": row["A"]})
    sonuc = q.state_names["RESULT"]
    y_pred.append(sonuc[0])


accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# hassasiyet hesapla
precision = precision_score(y_test, y_pred,zero_division=1)
print("Hassasiyet:", precision)

# özgüllük hesapla
recall = recall_score(y_test, y_pred,zero_division=1)
print("Özgüllük:", recall)

# F1 puanı hesapla
f1 = f1_score(y_test, y_pred)
print("F1 Puanı:", f1)


# ROC eğrisi hesapla
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
# AUC puanı hesapla
roc_auc = auc(fpr, tpr)

# ROC eğrisini çizdir
plt.plot(fpr, tpr, label='ROC Eğrisi (AUC = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Oranı')
plt.ylabel('True Positive Oranı')
plt.title('ROC Eğrisi')
plt.legend(loc="lower right")
plt.show()


confusionMatrix = confusion_matrix(y_test, y_pred)
sns.heatmap(confusionMatrix, annot=True, cmap="Reds",xticklabels=['T', 'F'], yticklabels=['T', 'F'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()