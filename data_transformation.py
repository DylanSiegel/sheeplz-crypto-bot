# data_transformation.py
import cudf
import cupy as cp
from cuml.decomposition import PCA as cuPCA
from sklearn.preprocessing import StandardScaler
from scipy.fft import fft


def calculate_fft(series):
    fft_values = fft(series.values)
    return cudf.DataFrame({
        'FFT_Real': fft_values.real,
        'FFT_Imag': fft_values.imag,
        'FFT_Magnitude': cp.abs(fft_values),
        'FFT_Phase': cp.angle(fft_values)
    })


def perform_pca(df, n_components=5):
    num_cols = df.select_dtypes(include=[cp.float64, cp.float32]).columns
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[num_cols].to_pandas())
    pca = cuPCA(n_components=n_components)
    pca_result = pca.fit_transform(scaled_data)
    return cudf.DataFrame({f'PCA_{i+1}': pca_result[:, i] for i in range(n_components)})