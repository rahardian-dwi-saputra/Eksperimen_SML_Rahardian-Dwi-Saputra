import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from joblib import dump


def preprocess_data(filepath_or_df, target_column, save_path, file_path):

    # Memuat Data dari file/dataframe
    if isinstance(filepath_or_df, str):
        df_clean = pd.read_csv(filepath_or_df)
    else:
        df_clean = filepath_or_df.copy()


    # Imputasi nilai 0 invalid pada RestingBP dan Cholesterol
    for col in ["RestingBP", "Cholesterol"]:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].replace(0, np.nan)
            # Menggunakan tren dari dataset asal (df_clean) untuk fillna
            df_clean[col] = df_clean[col].fillna(
                df_clean.groupby("Sex")[col].transform("median")
            )

    # Menentukan fitur numerik dan kategoris
    numeric_features = df_clean.select_dtypes(include=['float64', 'int64']).columns.tolist()

    # Menyertakan 'object' dan 'string' secara eksplisit sesuai rekomendasi Pandas terbaru
    categorical_features = df_clean.select_dtypes(include=['object', 'string']).columns.tolist()

    # Pastikan target_column tidak ada di numeric_features atau categorical_features
    if target_column in numeric_features:
        numeric_features.remove(target_column)
    if target_column in categorical_features:
        categorical_features.remove(target_column)
   
    # Deteksi kolom binary numerik dan pisahkan kolom kontinu
    binary_numeric_cols = [
        col for col in numeric_features if df_clean[col].nunique() == 2
    ]
    continuous_cols = [
        col for col in numeric_features if col not in binary_numeric_cols
    ]

    # Penanganan nilai outlier
    for col in continuous_cols:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Melakukan Capping (mengganti nilai di luar batas ke nilai batas atas/bawah)
        df_clean[col] = np.where(
            df_clean[col] < lower_bound, lower_bound, df_clean[col]
        )
        df_clean[col] = np.where(
            df_clean[col] > upper_bound, upper_bound, df_clean[col]
        )

    
    # Memisahkan target
    X = df_clean.drop(columns=[target_column])
    y = df_clean[target_column]

    # Membagi data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    # Pipeline untuk fitur kontinu (Hanya Scaling)
    numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])

    # Pipeline untuk fitur kategoris (One-Hot Encoding)
    categorical_transformer = Pipeline(
        steps=[("encoder", OneHotEncoder(drop="first", sparse_output=False))]
    )

    # Menggabungkan transformer ke ColumnTransformer
    # remainder='passthrough' memastikan kolom binary_numeric_cols tidak diapa-apakan (lolos otomatis)
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, continuous_cols),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="passthrough",
    )

    # Fitting dan transformasi data melalui Pipeline resmi
    X_train_res = preprocessor.fit_transform(X_train)
    X_test_res = preprocessor.transform(X_test)

    # Simpan pipeline objek preprocessor agar bisa digunakan saat deployment
    dump(preprocessor, save_path)
    print(f"Pipeline preprocessor berhasil disimpan ke: {save_path}")

    
    # Mendapatkan susunan nama kolom baru setelah One-Hot Encoding selesai
    encoded_cat_cols = (
        preprocessor.named_transformers_["cat"]
        .named_steps["encoder"]
        .get_feature_names_out(categorical_features)
        .tolist()
    )

    # Susunan kolom hasil remainder='passthrough' (yaitu binary_numeric_cols)
    remainder_cols = binary_numeric_cols

    # Urutan kolom final di dalam array hasil ColumnTransformer
    final_column_names = continuous_cols + encoded_cat_cols + remainder_cols

    # Mengubah kembali array training & testing menjadi DataFrame Pandas
    df_train_final = pd.DataFrame(X_train_res, columns=final_column_names)
    df_train_final[target_column] = y_train.values  # Memasukkan kembali target

    df_test_final = pd.DataFrame(X_test_res, columns=final_column_names)
    df_test_final[target_column] = y_test.values  # Memasukkan kembali target

    # Menggabungkan kembali data Train dan Test menjadi satu dataset utuh hasil preprocess
    df_all_final = pd.concat([df_train_final, df_test_final], axis=0).reset_index(
        drop=True
    )

    # Menyimpan hasil akhir transformasi penuh ke file CSV
    df_all_final.to_csv(file_path, index=False, header=True)
    print(f"Seluruh hasil akhir preprocessing disimpan ke: {file_path}")

    return X_train_res, X_test_res, y_train, y_test


X_train, X_test, y_train, y_test = preprocess_data(
    filepath_or_df="heart_raw/heart.csv",
    target_column="HeartDisease",
    save_path="preprocessing/preprocessor_pipeline.joblib",
    file_path="preprocessing/heart_preprocessing.csv",
)