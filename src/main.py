import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
import warnings
import os

# Mengabaikan peringatan (warnings) agar output konsol tetap bersih
warnings.filterwarnings('ignore')

def main():
    # ---------------------------------------------------------
    # 0. PENGATURAN GAYA VISUALISASI (SCIENTIFIC PUBLICATION STYLE)
    # ---------------------------------------------------------
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except:
        try:
            plt.style.use('seaborn-whitegrid')
        except:
            pass # Fallback jika seaborn tidak terpasang di versi environment tertentu

    sns.set_context("paper", font_scale=1.2)

    # ---------------------------------------------------------
    # 1. MEMBACA DAN MEMPERSIAPKAN DATA UTAMA
    # ---------------------------------------------------------
    file_utama = 'data_bersih_XY.csv'
    
    if not os.path.exists(file_utama):
        print(f"Error: File '{file_utama}' tidak ditemukan di direktori ini.")
        return

    # Membaca data harga
    df = pd.read_csv(file_utama)
    col_x = [c for c in df.columns if c.upper() == 'X'][0]
    col_y = [c for c in df.columns if c.upper() == 'Y'][0]

    X_raw = df[col_x].values
    y = df[col_y].values
    X = X_raw.reshape(-1, 1)

    print("Data utama berhasil dimuat. Memulai pemrosesan data iklim...\n")

    # ---------------------------------------------------------
    # 2. PARSING OTOMATIS DATA CURAH HUJAN (OPSI 2)
    # ---------------------------------------------------------
    file_iklim = 'laporan iklim harian jan jun.csv'
    hujan_ditemukan = False

    if os.path.exists(file_iklim):
        try:
            # Baca file csv dengan delimiter ';' dan lewati 7 baris header metadata
            df_iklim = pd.read_csv(file_iklim, sep=';', skiprows=7)
            
            # Temukan kolom-kolom yang berpasangan TANGGAL dan RR
            tanggal_cols = [c for c in df_iklim.columns if 'TANGGAL' in str(c).upper()]
            rr_cols = [c for c in df_iklim.columns if 'RR' in str(c).upper()]
            
            all_dates = []
            all_rr = []
            
            # Looping untuk mem-flatten (mendatarkan) format matriks menjadi 1 dimensi waktu berurut
            for d_col, r_col in zip(tanggal_cols, rr_cols):
                for d, r in zip(df_iklim[d_col], df_iklim[r_col]):
                    if pd.notna(d) and str(d).strip() != '':
                        all_dates.append(d)
                        # Format angka desimal Indonesia pakai koma, ubah ke titik
                        r_val = str(r).replace(',', '.').strip()
                        try:
                            all_rr.append(float(r_val))
                        except:
                            all_rr.append(0.0) # Jika data kosong/strip, jadikan 0
            
            # Konversi ke DataFrame, parsing tipe tanggal, urutkan, dan reset indeks
            df_hujan = pd.DataFrame({'TANGGAL': all_dates, 'RR': all_rr})
            df_hujan['TANGGAL'] = pd.to_datetime(df_hujan['TANGGAL'], format='%d/%m/%Y', errors='coerce')
            df_hujan = df_hujan.dropna(subset=['TANGGAL']).sort_values('TANGGAL').reset_index(drop=True)
            
            # Gabungkan curah hujan (RR) ke dalam dataframe utama sesuai urutan index X
            # (Diasumsikan urutan X 1 sampai N sejalan dengan rentang harian yang ada)
            min_len = min(len(df), len(df_hujan))
            df['RR'] = 0.0 # Default
            df.loc[:min_len-1, 'RR'] = df_hujan['RR'].values[:min_len]
            hujan_ditemukan = True
            
            print(f"[INFO] Data curah hujan sukses diparsing dan digabungkan ({min_len} baris harian).")
        except Exception as e:
            print(f"[WARNING] Gagal memproses ekstraksi file iklim: {e}")
    else:
        print(f"[WARNING] File '{file_iklim}' tidak ditemukan. Grafik 4 akan dilewati.")

    # ---------------------------------------------------------
    # 3. GRAFIK 1: PIECEWISE REGRESSION (DETEKSI TITIK KRITIS)
    # ---------------------------------------------------------
    best_mse = float('inf')
    best_bp_x = None
    best_model = None

    for bp in X_raw[5:-5]: # Hindari 5 titik ujung agar segmen tidak error
        X2 = np.maximum(0, X_raw - bp).reshape(-1, 1)
        X_piecewise = np.hstack([X, X2])
        
        model = LinearRegression().fit(X_piecewise, y)
        y_pred = model.predict(X_piecewise)
        mse = mean_squared_error(y, y_pred)
        
        if mse < best_mse:
            best_mse = mse
            best_bp_x = bp
            best_model = model

    X2_best = np.maximum(0, X_raw - best_bp_x).reshape(-1, 1)
    X_pw_best = np.hstack([X, X2_best])
    y_pred_pw = best_model.predict(X_pw_best)
    best_bp_y = best_model.predict(np.array([[best_bp_x, 0]]))[0]

    plt.figure(figsize=(12, 6), dpi=300)
    plt.scatter(X_raw, y, color='gray', alpha=0.6, label='Data Aktual')
    
    mask_left = X_raw <= best_bp_x
    mask_right = X_raw >= best_bp_x
    plt.plot(X_raw[mask_left], y_pred_pw[mask_left], color='green', linewidth=2.5, label='Segmen 1 (Sebelum Kritis)')
    plt.plot(X_raw[mask_right], y_pred_pw[mask_right], color='red', linewidth=2.5, label='Segmen 2 (Setelah Kritis)')
    
    plt.scatter(best_bp_x, best_bp_y, color='gold', edgecolor='black', s=400, marker='*', zorder=5, label='Titik Kritis')
    plt.annotate(
        f'TITIK KRITIS (Early Warning System)\nX: {best_bp_x:.1f}, Y: Rp {best_bp_y:,.0f}',
        xy=(best_bp_x, best_bp_y),
        xytext=(best_bp_x - len(X_raw)*0.15, best_bp_y + (y.max() - y.min())*0.15),
        arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=7),
        fontsize=10, weight='bold',
        bbox=dict(boxstyle="round,pad=0.4", fc="lightyellow", ec="orange", lw=1.5)
    )

    plt.title("Deteksi Titik Kritis Fluktuasi Harga Bawang Merah\nMenggunakan Model Piecewise Regression", fontsize=14, weight='bold', pad=15)
    plt.xlabel("Indeks Waktu (X)")
    plt.ylabel("Harga (Y)")
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig('grafik_piecewise.png')
    plt.close()

    # ---------------------------------------------------------
    # 4. GRAFIK 2: KEGAGALAN REGRESI LINEAR (DIREVISI)
    # ---------------------------------------------------------
    model_lin = LinearRegression().fit(X, y)
    y_pred_lin = model_lin.predict(X)
    
    residuals = np.abs(y - y_pred_lin)
    max_res_idx = np.argmax(residuals)
    max_res_x = X_raw[max_res_idx]
    max_res_y = y[max_res_idx]

    plt.figure(figsize=(12, 6), dpi=300)
    plt.scatter(X_raw, y, color='gray', alpha=0.6, label='Data Aktual')
    plt.plot(X_raw, y_pred_lin, color='blue', linestyle='--', linewidth=2, label='Regresi Linear Orde 1')
    
    # REVISI: Menambahkan padding ekstrim sebesar 35% di atas agar teks tidak memotong judul
    y_range = y.max() - y.min()
    plt.ylim(y.min() - (y_range * 0.05), y.max() + (y_range * 0.35)) 
    
    # REVISI: Dinamis melempar teks ke kiri atau kanan agar tidak memotong tepi grafik
    if max_res_x > np.median(X_raw):
        x_offset = max_res_x - (len(X_raw) * 0.25) # Tarik teks ke kiri
    else:
        x_offset = max_res_x + (len(X_raw) * 0.05) # Tarik teks ke kanan
        
    y_offset = max_res_y + (y_range * 0.15) # Teks di atas titik

    plt.annotate(
        "GAGAL MENANGKAP LONJAKAN EKSTREM",
        xy=(max_res_x, max_res_y),
        xytext=(x_offset, y_offset),
        arrowprops=dict(facecolor='red', shrink=0.05, width=2, headwidth=8),
        fontsize=11, color='darkred', weight='bold',
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="red", lw=1.5, alpha=0.9)
    )

    # REVISI: pad=20 akan memberi margin yang aman antara grafik dan teks judul
    plt.title("Kegagalan Regresi Linear Tunggal\nMemodelkan Dinamika Harga Ekstrem", fontsize=14, weight='bold', pad=20)
    plt.xlabel("Indeks Waktu (X)")
    plt.ylabel("Harga (Y)")
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig('grafik_linear_failure.png')
    plt.close()

    # ---------------------------------------------------------
    # 5. GRAFIK 3: REGRESI POLINOMIAL (ORDE 2)
    # ---------------------------------------------------------
    poly = PolynomialFeatures(degree=2)
    X_poly = poly.fit_transform(X)
    model_poly = LinearRegression().fit(X_poly, y)
    y_pred_poly = model_poly.predict(X_poly)

    X_smooth = np.linspace(X_raw.min(), X_raw.max(), 500).reshape(-1, 1)
    X_smooth_poly = poly.transform(X_smooth)
    y_smooth = model_poly.predict(X_smooth_poly)

    plt.figure(figsize=(12, 6), dpi=300)
    plt.scatter(X_raw, y, color='gray', alpha=0.6, label='Data Aktual')
    plt.plot(X_smooth, y_smooth, color='red', linewidth=2.5, label='Regresi Polinomial (Degree 2)')

    plt.title("Implementasi Regresi Polinomial pada Dinamika Harga Ekstrem", fontsize=14, weight='bold', pad=15)
    plt.xlabel("Indeks Waktu (X)")
    plt.ylabel("Harga (Y)")
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig('grafik_polynomial.png')
    plt.close()

    # Metrik Evaluasi
    r2_lin = r2_score(y, y_pred_lin)
    rmse_lin = np.sqrt(mean_squared_error(y, y_pred_lin))
    r2_poly = r2_score(y, y_pred_poly)
    rmse_poly = np.sqrt(mean_squared_error(y, y_pred_poly))

    # ---------------------------------------------------------
    # 6. GRAFIK 4: HUBUNGAN CURAH HUJAN DAN HARGA (TWINX)
    # ---------------------------------------------------------
    if hujan_ditemukan:
        fig, ax1 = plt.subplots(figsize=(12, 6), dpi=300)
        
        # Grafik Sumbu Kiri: Curah Hujan (Bar)
        # alpha=0.3 agar transparan dan tidak mendominasi grafik harga
        bar_plot = ax1.bar(X_raw, df['RR'], color='royalblue', alpha=0.3, label='Curah Hujan (mm)')
        ax1.set_xlabel('Indeks Waktu Harian (X)')
        ax1.set_ylabel('Curah Hujan (mm)', color='royalblue', weight='bold')
        ax1.tick_params(axis='y', labelcolor='royalblue')
        
        # Grafik Sumbu Kanan: Harga (Line)
        ax2 = ax1.twinx()
        line_plot = ax2.plot(X_raw, y, color='crimson', linewidth=2.5, label='Harga Bawang Merah')
        ax2.set_ylabel('Harga (Rp)', color='crimson', weight='bold')
        ax2.tick_params(axis='y', labelcolor='crimson')
        
        # Gabungkan legenda
        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')

        plt.title("Dinamika Curah Hujan vs Fluktuasi Harga Bawang Merah", fontsize=14, weight='bold', pad=15)
        plt.tight_layout()
        plt.savefig('grafik_hujan_harga.png')
        plt.close()

    # ---------------------------------------------------------
    # 7. OUTPUT KONSOL (RINGKASAN STATISTIK)
    # ---------------------------------------------------------
    print("\n" + "="*55)
    print("      RINGKASAN STATISTIK HASIL PEMODELAN")
    print("="*55)
    print("Regresi Linear Tunggal:")
    print(f" - R²   : {r2_lin:.4f}")
    print(f" - RMSE : Rp {rmse_lin:,.2f}")
    print("-" * 30)
    print("Regresi Polinomial (Orde 2):")
    print(f" - R²   : {r2_poly:.4f}")
    print(f" - RMSE : Rp {rmse_poly:,.2f}")
    print("-" * 30)
    print("Titik Kritis (Piecewise Breakpoint):")
    print(f" - Indeks X : {best_bp_x:.1f}")
    print(f" - Harga Y  : Rp {best_bp_y:,.2f}")
    print("="*55)
    
    if hujan_ditemukan:
        print("\n[SUCCESS] Seluruh 4 grafik (termasuk Curah Hujan vs Harga) telah berhasil disimpan.")
    else:
        print("\n[INFO] Selesai dengan 3 grafik. Grafik hujan dilewati karena ketiadaan data.")

if __name__ == '__main__':
    main()