# Laporan Proyek *Machine Learning* Terapan 1 - Klasifikasi Kualitas Air dengan Algoritma *K-Nearest Neighbor* dan *Decision Tree* 
oleh : Farin Istighfarizky

## **Domain Proyek**
Domain proyek _machine learning_ ini adalah di bidang kesehatan. Bidang kesehatan banyak memanfaatkan teknologi, seperti _machine learning_ dan _artificial intelligence_. _Machine learning_ adalah pemelajaran mesin yang sangat membantu dalam menyelesaikan masalah. Dibidang kesehatan, _machine learning_ dapat membuat mudah dalam mengerjakan sesuatu, contohnya dokter bisa mendiagnosa penyakit jantung dalam waktu cepat tanpa memakan waktu yang lama. ([Referensi Jurnal](http://jurnal.unprimdn.ac.id/index.php/JUTIKOMP/article/view/657)).

Salah satu pemanfaatan dari _machine learning_ yang dapat diterapkan dalam bidang kesehatan yaitu klasifikasi kualitas air. Adanya _machine learning_ akan memudahkan pihak PDAM dalam menentukan kualitas air tersebut aman untuk dikonsumsi atau tidak. Pada proyek ini, permasalahan yang akan dikerjakan yaitu klasifikasi kualitas air berdasarkan fitur-fitur yang merupakan faktor yang diperhitungkan dalam mengetahui apakah air tersebut aman atau tidak untuk digunakan. Harapan dari proyek ini adalah dapat membantu otomatisasi pengklasifikasian kualitas air sehingga dapat membantu pihak PDAM dan juga masyarakat yang ingin mengetahui status kualitas air aman dikomsumsi atau tidak.
  
Penelitian klasifikasi kualitas air ini sudah pernah dilakukan oleh beberapa peneliti sebelumnya, salah satunya adalah penelitian yang berjudul "Komparasi Metode Data Mining _Support Vector Machine_ dengan _Naive Bayes_ untuk Klasifikasi Status Kualitas Air" yang dilakukan oleh (Tumangger *et al*, 2020). Penelitian tersebut menggunakan algoritma _Support Vector Machine_ dan _Naive Bayes_ dalam melakukan klasifikasi kualitas air dan menghasilkan akurasi sebesar 78,70% pada _Support Vector Machine_, sedangkan pada_Naïve Bayes_ sebesar 85,78%. ([Referensi Jurnal](https://j-ptiik.ub.ac.id/index.php/j-ptiik/article/view/6482)).

## ***Business Understanding***
### ***Problem Statements***
Berdasarkan permasalahan diatas, proyek ini dibuat untuk menjawab permasalahan berikut:
- Algoritma manakah yang memiliki akurasi yang paling baik dalam mengklasifikasikan kualitas air?

### ***Goals***
- Mengetahui perbandingan performa dari algoritma _K-Nearest Neighbor_ dan _Decision Tree_ dalam klasifikasi kualitas air.

### ***Solution statements***
Pada proyek ini, algoritma *machine learning* yang digunakan adalah algoritma *K-Nearest Neighbor* dan *Decision Tree Classifier*.

### ***Data Understanding***
*Dataset* yang digunakan pada proyek ini adalah data sekunder yang bersumber dari situs Kaggle dan dapat diunduh melalui tautan [Dataset kualitas air](https://www.kaggle.com/mssmartypants/water-quality). Dataset terdiri dari 6.399 sampel dengan 5661 sampel not safe (label 0) dan 738 sampel safe (label 1).

*Detail* informasi *dataset* pada Kaggle :
* *License*            : *Data files © Original Authors*
* *Tags*               : *earth and nature, classification, energy, binary classification, pollution, and water transport*
* *Usability*          : 9.1
* *File Format / Size* : CSV 840.69 kB

Variabel-variabel pada dataset kualita air tersebut adalah sebagai berikut:
- *Aluminium*   : Berbahaya jika lebih besar dari 2.8
- *Ammonia*     : Berbahaya jika lebih besar dari 32.5
- *Arsenic*     : Berbahaya jika lebih besar dari 0.01
- *Barium*      : Berbahaya jika lebih besar dari 2
- *Cadmium*     : Berbahaya jika lebih besar dari 0.005
- *Chloramine*  : Berbahaya jika lebih besar dari 4
- *Chromium*    : Berbahaya jika lebih besar dari 0.1
- *Copper*      : Berbahaya jika lebih besar dari 1.3
- *Flouride*    : Berbahaya jika lebih besar dari 1.5
- *Bacteria*    : Berbahaya jika lebih besar dari 0
- *Viruses*     : Berbahaya jika lebih besar dari 0
- *Lead*        : Berbahaya jika lebih besar dari 0.015
- *Nitrates*    : Berbahaya jika lebih besar dari 10
- *Nitrites*    : Berbahaya jika lebih besar dari 1
- *Mercury*     : Berbahaya jika lebih besar dari 0.002
- *Perchlorate* : Berbahaya jika lebih besar dari 56
- *Radium*      : Berbahaya jika lebih besar dari 5
- *Selenium*    : Berbahaya jika lebih besar dari 0.5
- *Silver*      : Berbahaya jika lebih besar dari 0.1
- *Uranium*     : Berbahaya jika lebih besar dari 0.3
- *Is_safe*     : Label kualitas air (aman atau tidak). air yang aman ditandai dengan label 1 dan air yang tidak aman ditandai dengan label 0.

![alternate text](https://raw.githubusercontent.com/Farinist/MLT1/main/pic/pic1.png)


Setelah dilakukan proses _data understanding_, ternyata ditemukan bahwa pada kolom _ammonia_ terdapat anomali data dimana terdapat variabel "#NUM!" yang tidak diperlukan dalam proses (klasifikasi/prediksi), sehingga nantinya variabel tersebut akan dihapus terlebih dahulu sebelum pengecekan keberadaan _missing value_ pada dataset.

### _Exploratory Data Analysis_ (EDA) 
- ![alternate text](https://raw.githubusercontent.com/Farinist/MLT1/main/pic/pic2.png)

Hasil visualisasi dari *Univariate Analysis* menunjukkan bahwa:
  1. Variabel _cadmium_, _flouride_, _bacteria_, _lead_, _nitrates_, _nitrites_, _mercury_, _selenium_, dan _uranium_     berdistribusi normal.
  2. Variabel _aluminium_, _ammonia_, _arsenic_, _barium_, _chloramine_, _chromium_, _copper_, _perchlorate_, _radium_, dan _silver_ cenderung memiliki kemiringan positif.
  3. Variable _viruses_ dan _is_safe_ cenderung memiliki kemiringan negatif.

- ![alternate text](https://raw.githubusercontent.com/Farinist/MLT1/main/pic/pic3.png)

Hasil Visualisasi dari matrix korelasi menunjukkan bahwa pada variabel _chloramine_ dan _chromium_ memiliki korelasi yang tinggi.

### ***Data Preparation***
- Membagi dataset

  Data dibagi menjadi dua yaitu data testing (uji) dan data training (latih), 20% data testing dan 80% data training. Pada proses ini dilakukan dengan menggunakan fungsi *train_test_split* yaitu salah satu metode yang dapat digunakan untuk mengevaluasi performa model *machine learning*.

- Melakukan *undersampling*

  *Undersampling* digunakan untuk mengatasi data yang tidak seimbang yang merupakan salah satu teknik dari *data preparation*. Jumlah data yang berlabel 0 (tidak aman) tidak sama dengan data yang berlabel 1 (aman). Oleh karena itu diperlukan teknik *undersampling* untuk melakukan penyeimbangan data dan untuk mengurangi bias pada saat proses pengklasifikasian. Pada tahap *undersampling* menggunakan fungsi *resample* Fungsi dari *resample* ini digunakan pada data yang berlabel 0 (tidak aman) dengan menggunakan parameter jumlah dari label 1 (aman) sehingga jumlahnya akan menjadi sama dengan jumlah data yang berlabel 1 (aman). 

- Melakukan standarisasi data

  Pada tahap ini untuk menyeragamkan nilai-nilai data yang pada penginputannya formatnya tidak konsisten menggunakan suatu format tertentu, hingga seluruh data menjadi standar. Proses standarisasi data dengan menggunakan fungsi *StandardScaler*.

## ***Modeling***
- ***K-Nearest Neighbor*** merupakan algoritma yang berfungsi untuk melakukan klasifikasi suatu data berdasarkan data pembelajaran, yang diambil dari k tetangga terdekatnya (_nearest neighbors_), dengan k merupakan banyaknya tetangga terdekat. Pada proyek ini menggunakan parameter n_neighbors = 6, karena setelah dicoba dengan n=2, n=4, n=6, didapatkan bahwa n berjumlah 6 menghasilkan akurasi yg tertinggi. Proses *modeling* yang pertama menggunakan algoritma *K-Nearest Neighbor*. Tahap awal pada proses ini adalah memanggil fungsi *KNeighborClassifier*. Proses pelatihan data dilakukan dengan 80% data latih yang telah dibagi. Kemudian,hasil dari proses pelatihan tersebut akan menghasilkan model yang nantinya akan digunakan dalam pengujian data uji sebesar 20% dari jumlah dataset.

- Kelebihan dari Algoritma _K-Nearest Neighbor_:
1. Algoritma _K-Nearest Neighbor_ kuat dalam mentraining data yang _noisy_.
2. Algoritma _K-Nearest Neighbor_ sangat efektif jika datanya besar.
3. Mudah diimplementasikan.

- Kekurangan dari Algoritma _K-Nearest Neighbor_:
1. Algoritma _K-Nearest Neighbor_ perlu menentukan nilai parameter K.
2. Sensitif pada data pencilan.
3. Rentan pada variabel yang non-informatif.

- ***Decision tree*** adalah salah satu metode klasifikasi yang populer karena mudah diinterpretasikan. *Decision tree* digunakan untuk pengenalan pola dan termasuk dalam pengenalan pola secara statistik. Visual dari *Decision tree* adalah seperti pohon bercabang beserta ranting dari cabangnya.Proses *modeling* yang kedua menggunakan algoritma *Decision Tree*. Tahap awal pada proses ini adalah memanggil fungsi *DecisionTreeClassifier*. Proses pelatihan data dilakukan dengan 80% data latih yang telah dibagi. Kemudian,hasil dari proses pelatihan tersebut akan menghasilkan model yang nantinya akan digunakan dalam pengujian data uji sebesar 20% dari jumlah dataset.

- Kelebihan dari Algoritma _Decision tree_:
1. Area keputusan yang sebelumnya kompleks dapat dibuat lebih sederhana dan lebih spesifik.
2. Dapat menghilangkan perhitungan yang tidak diperlukan.
3. Pemilihan fitur yang fleksibel, sehingga fitur yang dipilih dapat membedakan kriteria dari kriteria lain di node yang sama.

- Kekurangan dari Algoritma _Decision tree_:
1. Tumpang tindih terjadi terutama ketika sangat banyak kelas dan kriteria digunakan.
2. Akumulasi jumlah kesalahan dari setiap level dalam pohon keputusan besar.
3. Kesulitan dalam merancang pohon keputusan yang optimal.

## ***Evaluation***
Pada tahap evaluasi, *confusion matrix* digunakan untuk menghitung *precision, recall, f1-score*, dan akurasi. *Confusion matrix* menyimpulkan performa klasifikasi dari sebuah *classifier* yang sehubungan dengan data uji.

***Confusion Matrix*** adalah tabel dengan 4 kombinasi berbeda dari nilai prediksi dan nilai aktual, yaitu:
- True Positive (TP)  : Jumlah data positif yang diprediksi benar
- True Negative (TN)  : Jumlah data negatif yang diprediksi benar
- False Negative (FN) : Jumlah data positif tetapi diprediksi sebagai negatif
- False Positive (FP) : Jumlah data negatif tetapi diprediksi sebagai positif

**Metrik evaluasi** yang digunakan adalah *precision, recall, f1-score*, dan akurasi.

- ***Precision*** adalah perbandingan antara *True Positive* dengan banyaknya data yang diprediksi positif. Rumus *precision* sebagai berikut :
  *Precision* = TP/(TP+FP)

- ***Recall*** adalah perbandingan antara *True Positive* dengan banyaknya data yang sebenarnya positif. Rumus *recall* sebagai berikut :
  *Recall* = TP/(TP+FN)

- ***F1-score*** adalah rata-rata geometris dari *precision* dan *recall*. Rumus *F1-score* sebagai berikut :
  *F1-score* = (2 x *recall* x *precision*)/(*recall*+*precision*)

- **Akurasi** adalah kedekatan nilai aktual atau nilai sebenarnya dengan nilai yang diprediksi. Rumus akurasi sebagai berikut :
  Akurasi = (TP+TN)/(TP+TN+FP+FN)

Hasil evaluasi dari algoritma *K-Nearest Neighbor* ditunjukkan pada tabel berikut :

Metrik       | Not safe | Safe
------------ | ---------| --------
*Precision*  | 97.89%   | 27.33%
*Recall*     | 71.66%   | 87.35%
*F1-Score*   | 82.75%   | 41.64%
Akurasi                            | 73.38%

Hasil evaluasi dari algoritma *Decision Tree* ditunjukkan pada tabel berikut :

Metrik       | Not safe  | Safe
------------ | ----------| -------
*Precision*  | 97.70%    | 22.36%
*Recall*     | 62.76%    | 87.93%
*F1-Score*   | 76.43%    | 35.66%
Akurasi                             | 65.5%

Berdasarkan hasil evaluasi diatas menunjukkan bahwa hasil akurasi kedua algoritma berbeda jauh. Algoritma *K-Nearest Neighbor* menghasilkan performa yang lebih baik dibandingkan dengan algoritma *Decision Tree*.

Algoritma *K-Nearest Neighbor* menghasilkan akurasi sebesar 73.38% ketika nilai k=6. Pada label "not safe", rata-rata dari nilai *precision* sebesar 97.89%, nilai *recall* sebesar 71.66%, dan nilai *f1-score* sebesar 82.75%. Sedangkan pada label "safe" rata-rata dari nilai *precision* sebesar 27.33%, nilai *recall* sebesar 87.35%, dan nilai *f1-score* sebesar 41.64%.

Algoritma *Decision Tree* menghasilkan akurasi sebesar 65.5%. Pada label "not safe", rata-rata dari nilai *precision* sebesar 97.70%, nilai *recall* sebesar 62.76%, dan nilai *f1-score* sebesar 76.43%. Sedangkan pada label "safe" rata-rata dari nilai *precision* sebesar 22.36%, nilai *recall* sebesar 87.93%, dan nilai *f1-score* sebesar 35.66%.   
