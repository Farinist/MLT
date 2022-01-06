# Laporan Proyek ke-2 *Machine Learning* Terapan - Sistem rekomendasi film 
oleh : Farin Istighfarizky

## ***Project Overview***
Film juga dikenal sebagai movie, gambar hidup, film teater atau foto bergerak, merupakan serangkaian gambar diam, yang ketika ditampilkan pada layar akan menciptakan ilusi gambar bergerak karena efek fenomena phi. Ilusi optik ini memaksa penonton untuk melihat gerakan berkelanjutan antar objek yang berbeda secara cepat dan berturut-turut. Proses pembuatan film merupakan gabungan dari seni dan industri. Sebuah film dapat dibuat dengan memotret adegan sungguhan dengan kamera film; memotret gambar atau model "miniatur" menggunakan teknik animasi tradisional, dengan CGI dan animasi komputer, atau dengan kombinasi beberapa teknik yang ada dan efek visual lainnya [Wikipedia](https://id.wikipedia.org/wiki/Film). Pertumbuhan pasar industri dari bidang perfilman di luar negeri hingga dalam negeri kian menjanjikan. Banyaknya film yang diproduksi membuat calon penonton kesulitan dalam menentukan film yang akan ditontonnya, hal tersebut tentunya akan memakan waktu. Oleh karena itu, pada proyek _machine learning_ terapan ke-2 ini saya membuat sistem rekomendasi film dengan menggunakan metode _content based filtering_.

## ***Business Understanding***
### ***Problem Statements***
Berdasarkan permasalahan diatas, proyek ini dibuat untuk menjawab permasalahan berikut:
- Bagaimana cara membuat model _machine learning_ untuk sistem rekomendasi film?

### ***Goals***
- Mengetahui cara membuat model _machine learning_ untuk sistem rekomendasi film.

### ***Solution statements***
Pada proyek kali ini, metode yang digunakan untuk membuat sistem rekomendasi adalah metode *Content-Based Filtering*.

### ***Data Understanding***
*Dataset* yang digunakan pada proyek ini adalah data sekunder yang bersumber dari situs Kaggle dan dapat diunduh melalui tautan [Dataset Industri Film](https://www.kaggle.com/danielgrijalvas/movies). Dataset terdiri dari 7512 judul film. Data film ini diambil dari IMDb.

*Detail* informasi *dataset* pada Kaggle :
* *License*            : CC0: *Public Domain*
* *Tags*               : *movies and tv shows*
* *Usability*          : 10.0
* *File Format / Size* : CSV 1.35 MB

Variabel-variabel pada dataset film tersebut adalah sebagai berikut:
- *Budget*  : Anggaran sebuah film
- *Company* : Perusahaan produksi
- *Country* : Negara asal film
- *Director*: Sutradara
- *Genre*   : Genre film
- *Gross*   : Pendapatan film
- *Name*    : Judul film
- *Rating*  : Rating film
- *Released*: Tanggal rilis film
- *Runtime* : Durasi film
- *Score*   : Peringkat IMDb film dari pengguna
- *Votes*   : Jumlah *votes* film dari pengguna
- *Star*    : Aktor atau aktris utama dalam film 
- *Writer*  : Penulis film
- *Year*    : Tahun rilis pada film

![alternate text](https://raw.githubusercontent.com/Farinist/MLT/main/pic/04.png)

Pada dataset yang digunakan, ternyata ditemukan bahwa pada kolom *rating, released, score, votes, writer, star, country, budget, gross, company,* dan *runtime* terdapat missing value.

## ***Data Preparation***
### ***Univariate Exploratory Data Analysis***
![alternate text](https://raw.githubusercontent.com/Farinist/MLT/main/pic/pic4.png)

3 genre film paling banyak berdasarkan hasil visualisasi diatas adalah *Comedy, Action* dan *Drama*. 

![alternate text](https://raw.githubusercontent.com/Farinist/MLT/main/pic/pic5.png)

3 rating film paling banyak berdasarkan hasil visualisasi diatas adalah R *(Restricted)*, PG *(Parental Guidance)* dan PG-13 *(Parental Guidance-13)*. 

![alternate text](https://raw.githubusercontent.com/Farinist/MLT/main/pic/pic6.png)

3 Negara asal yang memproduksi film paling banyak berdasarkan hasil visualisasi diatas adalah *United States*, *United Kingdom* dan *France*. 

### Pengecekan *missing value*

  _Missing value_ harus ditangani sebelum dataset diproses dengan _machine learning_. Pada tahap ini proses pengecekan *missing value* dengan menggunakan fungsi _isnull()_. Fungsi _isnull()_ yaitu fungsi dari pandas untuk memeriksa apakah dataframe ada nilai null. _Missing value_ ditemukan pada kolom _rating_ sebanyak 77, pada kolom _released_ sebanyak 2, pada kolom _score_ sebanyak 3, pada kolom _votes_ sebanyak 3, pada kolom _writer_ sebanyak 3, pada kolom _star_ sebanyak 1, pada kolom _country_ sebanyak 3, pada kolom _budget_ sebanyak 2171, pada kolom _gross_ sebanyak 189, pada kolom _company_ sebanyak 17, dan pada kolom _runtime_ sebanyak 4. Karena adanya _missing value_ pada data film tersebut, dapat membuat film tidak bisa diidentifikasikan ke dalam salah satu genre film. Oleh karena itu, tahap selanjutnya adalah membersihkan *missing value* menggunakan fungsi *dropna()* dan kemudian melakukan pengecekan _missing value_ kembali.

## ***Modeling***
Pada proyek ini proses *modeling* dilakukan menggunakan *Content-Based Filtering*.

Tahap pertama proses modeling pada proyek ini adalah vektorisasi dengan TF-IDF. Tf-Idf adalah perhitungan yang menggambarkan seberapa pentingnya kata dalam sebuah dokumen dan korpus. Proses ini digunakan untuk menilai bobot relevansi term dari sebuah dokumen terhadap seluruh dokumen dalam korpus. Tahap dari TF-IDF sebagai berikut:
  - inisialisasi TfidfVectorizer dengan memanggil fungsi *TfidfVectorizer()*.
  - Kemudian melakukan perhitungan IDF pada genre film dan melakukan proses *mapping array*.
  - Melakukan *fit*, kemudian ditransformasikan kedalam bentuk matriks dengan menggunakan fungsi *fit_transform()*.
 
Tahap kedua adalah menghitung *cosine similarity*. Kemudian membuat *dataframe* dari variabel *cosine_sim* berupa judul film, dan untuk melihat matriks *similarity* disetiap judul film dapat dilihat dengan menggunakan fungsi *sample()*, *sample* yang digunakan pada proyek ini adalah sebanyak 20 *sample*.

Tahap ketiga adalah membuat fungsi baru yaitu *movie_recommendations* untuk mendapatkan hasil rekomendasi dengan parameter sebagai berikut:
  - name       : yang merupakan judul film dengan tipe data *string*.
  - *sim_data* : kesamaan *dataframe*, simetrik, dengan film (tipe data *pd.DataFrame*).
  - *items*    : berisi judul film dan fitur-fitur lainnya yang digunakan untuk mendefinisikan kemiripan (tipe data *pd.DataFrame*).
  - k          : jumlah rekomendasi yang diberikan (tipe data *integer*).
Kemudian melakukan proses *argpartition* untuk melakukan partisi secara tidak langsung. Lalu mengambil data dengan *similarity* terbesar, menghapus judul film yang dicari agar tidak ikut menjadi rekomendasi judul film, mencari rekomendasi film yang memiliki genre yang sama dengan judul film yang di input dan tahap yang terakhir adalah memanggil fungsi *movie_recommendations* dengan memasukkan judul film yang ingin dicari, contoh "*The Blue Lagoon*" yang memiliki genre "*Adventure*". Hasil yang didapat adalah 5 rekomendasi judul film yang memiliki genre yang sama yaitu "*Adventure*".

Berikut adalah hasil 5 rekomendasi film yang memiliki genre yang sama dengan judul film yang diinput yaitu "*The Blue Lagoon*" :

Judul Film                      | Genre Film
--------------------------------| -----------
Shanghai Surprise	              | Adventure
Dunston Checks In	              | Adventure
Stand by Me	                    | Adventure
Robin Hood: Men in Tights	      | Adventure
Black Beauty	                  | Adventure


## ***Evaluation***
Metrik yang dipakai dalam mengerjakan sistem rekomendasi kali ini adalah *precision* karena hanya menggunakan satu model saja yaitu _content based filtering_ dengan menggunakan _TF-IDF Vectorize_. Rumus *precision* sebagai berikut :

*precision* = jumlah rekomendasi relevan yang berhasil ditemukan/ jumlah seluruh rekomendasi.

Pada *project* yang dikerjakan kali ini, jumlah rekomendasi pada film yang sesuai dengan genre sebanyak 5 film, dengan total rekomendasi yang ditampilkan adalah 5. Sehingga nilai dari *precision* dari sistem rekomendasi ini adalah 5/5 = 1 atau sama dengan 100%.
