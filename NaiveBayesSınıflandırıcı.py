import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import seaborn as sns
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

os.chdir('C:\\Users\\zuhal\\Desktop\\bioinformaticProjects')
#"pandas" kütüphanesi kullanılarak "heart_failure_clinical_records_dataset.csv" adlı CSV dosyası okunur
dataset = pd.read_csv('heart_failure_clinical_records_dataset.csv')



dataset['high_serum_creatinine'] = np.where(dataset['serum_creatinine'] > 1.5, 1, 0)
#dataset['high_age'] = np.where(dataset['age'] > 60, 1, 0)
#dataset['low_ef'] = np.where(dataset['ejection_fraction'] < 60, 1, 0)

#"dataset" veri çerçevesinden belirli sütunları seçerek bağımsız değişkenleri "X" , bağımlı değişkeni "y" olarak ayırır.
X = dataset[['age','high_serum_creatinine']].values #başka bağımsız değişken alınabilir mi ejection_fraction ve serum_creatinine
y = dataset[['DEATH_EVENT']].values


# Age, renal dysfunction, blood pressure, ejection fraction and anemia were found as significant risk factors for mortality among
# heart failure patients.
#0,1,4,5,7 --> 0.74
#0,1,5 --> 0.68
#0,1,4,5 --> 0.71
#4,7,8 --> 0.75
#0,4,10 --> 0.70
#0,1,3,4 --> 0.73
#0,1,4 --> 0.73
#0,1,4,5 --> 0.71


y = dataset[['DEATH_EVENT']].values
#("bağımsız değişkenler: ",X)
#("bağımlı değişkenler: "y)

# "train_test_split" fonksiyonunu kullanarak veri kümesini rastgele bir şekilde eğitim ve test alt kümelerine ayırır.
# %70 eğitim veri kümesi ve %30 test veri kümesi
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 0)



#"StandardScaler" sınıfını kullanarak verilerin ölçeklendirilmesini sağlar.
#Verilerin ölçeklendirilmesi, her bir özellik için ortalaması sıfır ve standart sapması bir olan bir dağılıma dönüştürülmesini içerir.
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)



#lk olarak, "GaussianNB" sınıfından bir sınıflandırıcı nesnesi ("classifier") oluşturulur.
#"GaussianNB", Naive Bayes algoritması temel alınarak oluşturulmuş bir sınıflandırıcıdır.
classifier = GaussianNB()

#"fit" yöntemi, sınıflandırıcı nesnesinin eğitim veri kümesine uyarlanmasını sağlar.
classifier.fit(X_train, y_train)

# "y_pred" dizisi, her bir test verisi örneğinin "classifier" tarafından tahmin edilen sınıfını içerir.
y_pred = classifier.predict(X_test)


#"confusion_matrix" fonksiyonunu kullanarak gerçek ve tahmin edilen sınıfların karşılaştırıldığı bir karışıklık matrisi oluşturur.
confusionMatrix = confusion_matrix(y_test, y_pred)


#"classification_report" fonksiyonu, sınıflandırıcının performansını özetleyen bir rapor üretir. Bu rapor, doğruluk, hassasiyet,
# recall ve F1 skoru gibi performans ölçüleri gibi sınıflandırıcının performansını özetleyen farklı metrikler sağlar.
classificationReport = classification_report(y_test, y_pred)
print('Sınıflandırma Raporu: ' ,classificationReport)


#"precision_recall_fscore_support" fonksiyonunu kullanarak sınıflandırıcının hassasiyet, geri çağırma ve F1 skoru gibi performans
# ölçüleri hakkında bilgi sağlayan bir 3 elemanlı tuple (hassasiyet, recall, F1 skoru) içerir
precisionScore = precision_recall_fscore_support(y_test, y_pred, average='macro')
print('Tahmin Skoru (Hassasiyet,Recall,F1 Skoru): ' ,precisionScore)

#heatmap kütüphanesi kullanarak karmaşıklık matrisini ısı haritası olarak görselleştirir
#"annot" parametresi, sayısal değerlerin her hücrede görüntülenip görüntülenmeyeceğini kontrol eder. "True" olarak ayarlandığında,
# sayısal değerler her hücrede görüntülenir.
#"cmap" parametresi, ısı haritasının kullanacağı renk paletini belirler. Bu kodda, "Blues" paleti kullanılır.
sns.heatmap(confusionMatrix, annot=True, cmap="Reds",xticklabels=['T', 'F'], yticklabels=['T', 'F'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()

#accuracy_score fonksiyonu, doğru sınıflandırma oranını (accuracy) hesaplamak için kullanılan bir fonksiyondur.
accuracy = accuracy_score(y_test, y_pred)
print('Doğruluk Puanı:', accuracy)

#train ve test verilerinde doğru sınıflandırma oranlarını (accuracy_score) hesaplar.
train_accuracy = accuracy_score(y_train, classifier.predict(X_train))
test_accuracy = accuracy_score(y_test, classifier.predict(X_test))
print('Eğitim Doğruluk Puanı:',train_accuracy)
print('Test Doğruluk Puanı:' ,test_accuracy)


#sınıflandırma raporu (classification_report) çıktısını kullanarak f1-score değerini hesaplar.
f1_score = classification_report(y_test, y_pred,labels=[0,1], output_dict=True)['1']['f1-score']
print("F1 Skoru:", f1_score)


#sklearn.metrics modülünde yer alan roc_curve() fonksiyonunu kullanarak, ROC eğrisi için False Positive Rate (FPR), True Positive Rate (TPR)
# ve eşik (thresholds) değerlerini hesaplar.
fpr, tpr, thresholds = roc_curve(y_test, y_pred)

# ROC eğrisinin çizdirilmesi
plt.plot(fpr, tpr, color='blue',label='ROC Eğrisi')
plt.plot([0, 1], [0, 1], color='navy', linestyle='--',label='Rastgele Tahmin')
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('ROC Eğrisi')
plt.legend()
plt.show()


#roc_auc_score fonksiyonu, ROC eğrisinin altındaki alanı (AUC) hesaplar.
#bu değer 1'e ne kadar yakınsa model o kadar iyi eğitilmiştir
roc_auc = roc_auc_score(y_test, y_pred)
print('ROC AUC Skoru:', roc_auc)

#precision_recall_curve fonksiyonu, bir sınıflandırma modelinin hassasiyet ve geri çağırma oranı (recall) gibi performans ölçütlerini
# çeşitli eşik değerleri için hesaplar ve bu eşik değerleri için hassasiyet ve geri çağırma oranlarını döndürür.
precision, recall, thresholds = precision_recall_curve(y_test, y_pred)

# Hassasiyet-doğruluk grafiğinin çizdirilmesi
plt.plot(recall, precision, label='Hassasiyet-Doğruluk Grafiği')
plt.xlabel('Doğruluk (Recall)')
plt.ylabel('Hassasiyet (Precision)')
plt.title('Hassasiyet-Doğruluk Grafiği')
plt.legend()
plt.show()



# Test verileri üzerinde sınıflandırıcıyı çalıştırın ve ROC eğrisini çizin
y_score = classifier.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_score)
roc_auc = auc(fpr, tpr)
# ROC eğrisini çizin
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()


