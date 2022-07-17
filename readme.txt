Susunan folder

-- CNN
    |-- CaptureFaceDataset.ipynb
    |-- Data-Trainer.ipynb

-- dataset
    |-- list dataset

-- haarcascade
    |-- haarcascade_frontalface_default.xml

-- output
    |-- labels.npy
    |-- trained-model.h5


Folder CNN berisi file capture dataset dan data trainer, untuk file capture dataset digunakan untuk membuat dataset secara real dengan
foto. sedangkan data trainer digunakan untuk training dataset dengan menggunakan algoritma CNN

Folder daataset berisi file file data yang akan digunakan untuk training

Folder haarcascade berisi model machine learning untuk pendeteksi wajah

Folder output berisi file file output dari hasil training

untuk melakukan presensi yaitu dengan menjalankan file Presensi.py, maka aplikasi akan terbuka dengan webcam dan mendeteksi wajah 