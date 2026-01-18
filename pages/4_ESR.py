import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.signal import find_peaks

# --- 定数 ---
H_PLANCK = 6.62607015e-34  # J·s
BOHR_MAGNETON = 9.27401007e-24  # J/T

def calculate_g_factor(magnetic_field_mt, frequency_ghz):
    """g値を計算する (磁場: mT, 周波数: GHz)"""
    if magnetic_field_mt == 0: return 0
    b_tesla = magnetic_field_mt * 1e-3
    freq_hz = frequency_ghz * 1e9
    g = (H_PLANCK * freq_hz) / (BOHR_MAGNETON * b_tesla)
    return g

def main():
    st.set_page_config(page_title="Advanced ESR Analyzer", layout="wide")
    st.title("🧲 Advanced ESR Spectrum Analyzer")

    # --- サイドバー：設定 ---
    st.sidebar.header("1. 測定条件・補正")
    uploaded_file = st.sidebar.file_uploader("ESRデータ (CSV/TXT) をアップロード", type=['csv', 'txt', 'dat'])
    
    freq_ghz = st.sidebar.number_input("測定周波数 (GHz)", value=9.450, format="%.4f")
    
    st.sidebar.subheader("磁場軸の補正")
    offset_mt = st.sidebar.number_input("磁場オフセット補正 (mT)", value=0.0, step=0.01, help="横軸を全体的に左右にずらします。")
    
    skip_head = st.sidebar.number_input("ヘッダー行数", value=0, min_value=0)
    
    st.sidebar.markdown("---")
    st.sidebar.header("2. 解析オプション")
    do_baseline = st.sidebar.checkbox("ベースライン補正", value=True)
    
    st.sidebar.subheader("ピーク検出設定 (HF用)")
    peak_prominence = st.sidebar.slider("ピーク検出感度", 0.01, 1.0, 0.1)

    if uploaded_file is not None:
        try:
            # 文字コードの候補
            encodings = ['cp932', 'utf-8', 'latin1']
            df = None
            
            for enc in encodings:
                try:
                    uploaded_file.seek(0)
                    df = pd.read_csv(uploaded_file, skiprows=skip_head, header=None, sep=None, engine='python', encoding=enc)
                    break # 読み込めたらループを抜ける
                except:
                    continue
            
            if df is None:
                st.error("ファイルの読み込みに失敗しました。文字コードを確認してください。")
                return

            df = df.apply(pd.to_numeric, errors='coerce').dropna()
            raw_field = df.iloc[:, 0].values  # 磁場 (mT)
            signal = df.iloc[:, 1].values     # 信号強度
            
            # --- 磁場補正 ---
            field = raw_field + offset_mt

            # --- ベースライン補正 ---
            if do_baseline:
                baseline = np.linspace(signal[0], signal[-1], len(signal))
                signal = signal - baseline

            # --- ハイパーファイン分裂・ピーク検出 ---
            # 1次微分波形の「山」と「谷」を検出
            peaks_pos, _ = find_peaks(signal, prominence=peak_prominence * np.max(signal))
            peaks_neg, _ = find_peaks(-signal, prominence=peak_prominence * np.max(np.abs(signal)))
            
            all_peak_indices = np.sort(np.concatenate([peaks_pos, peaks_neg]))
            
            # --- メイン表示エリア ---
            col1, col2 = st.columns([2, 1])

            with col1:
                st.subheader("スペクトル表示")
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(field, signal, label="1st Derivative", color='black', lw=1)
                
                # 検出されたピークをプロット
                if len(all_peak_indices) > 0:
                    ax.scatter(field[all_peak_indices], signal[all_peak_indices], color='red', s=30, label="Detected Peaks")
                
                ax.set_xlabel("Magnetic Field (mT)")
                ax.set_ylabel("Intensity (a.u.)")
                ax.axhline(0, color='gray', lw=0.5)
                ax.legend()
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)

                # 積分表示
                st.subheader("吸収波形 (1回積分)")
                abs_signal = cumulative_trapezoid(signal, field, initial=0)
                fig2, ax2 = plt.subplots(figsize=(10, 3))
                ax2.plot(field, abs_signal, color='forestgreen')
                ax2.fill_between(field, abs_signal, color='forestgreen', alpha=0.2)
                ax2.set_xlabel("Magnetic Field (mT)")
                ax2.grid(True, alpha=0.3)
                st.pyplot(fig2)

            with col2:
                st.subheader("📊 基本解析結果")
                if len(peaks_pos) > 0 and len(peaks_neg) > 0:
                    # 最も強い山と谷からg値を計算
                    f_max = field[peaks_pos[np.argmax(signal[peaks_pos])]]
                    f_min = field[peaks_neg[np.argmax(-signal[peaks_neg])]]
                    center_f = (f_max + f_min) / 2
                    g_val = calculate_g_factor(center_f, freq_ghz)
                    
                    st.metric("g値 (中心)", f"{g_val:.5f}")
                    st.metric("ΔHpp (ピーク間幅)", f"{abs(f_max - f_min):.3f} mT")
                
                st.divider()
                st.subheader("🧬 ハイパーファイン分裂 ($A$値)")
                
                if len(all_peak_indices) >= 2:
                    hf_data = []
                    # 隣り合うピーク間の距離を計算
                    for i in range(len(all_peak_indices) - 1):
                        idx1 = all_peak_indices[i]
                        idx2 = all_peak_indices[i+1]
                        dist_mt = abs(field[idx1] - field[idx2])
                        
                        # A値をMHzに変換 ( A[MHz] = g * (Bohr Magneton / h) * dist[mT] * 1e-3 )
                        # 近似的に A(MHz) ≒ 28.025 * (g/2.0023) * dist(mT)
                        # ここでは簡便に各ピーク間の中心g値を使用
                        avg_field = (field[idx1] + field[idx2]) / 2
                        current_g = calculate_g_factor(avg_field, freq_ghz)
                        a_mhz = current_g * BOHR_MAGNETON * (dist_mt * 1e-3) / H_PLANCK / 1e6
                        
                        hf_data.append({
                            "Peak Pair": f"{i+1}-{i+2}",
                            "分裂幅 (mT)": round(dist_mt, 4),
                            "A値 (MHz)": round(a_mhz, 2)
                        })
                    
                    st.table(pd.DataFrame(hf_data))
                    st.caption("※山と谷の両方をピークとして検出しています。分裂幅は隣接する赤点間の距離です。")
                else:
                    st.warning("複数のピークが検出されませんでした。サイドバーの「感度」を調整してください。")

                if st.button("2回積分値を計算"):
                    area = np.trapz(abs_signal, field)
                    st.write(f"相対強度 (Area): **{area:.2e}**")

        except Exception as e:
            st.error(f"エラー: {e}")
    else:
        st.info("👈 左のサイドバーからESRファイルを読み込んでください。")

if __name__ == "__main__":
    main()