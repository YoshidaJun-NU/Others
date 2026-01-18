import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.signal import find_peaks

# --- 定数 ---
H_PLANCK = 6.62607015e-34
BOHR_MAGNETON = 9.27401007e-24

def calculate_g_factor(magnetic_field_mt, frequency_ghz):
    if magnetic_field_mt == 0: return 0
    b_tesla = magnetic_field_mt * 1e-3
    freq_hz = frequency_ghz * 1e9
    g = (H_PLANCK * freq_hz) / (BOHR_MAGNETON * b_tesla)
    return g

def load_data_robust(uploaded_file, skip_rows):
    """文字コードと区切り文字を自動判別して読み込む"""
    encodings = ['cp932', 'shift_jis', 'utf-8', 'latin1']
    
    for enc in encodings:
        try:
            uploaded_file.seek(0)
            # 1列でも読み込めるように設定
            df = pd.read_csv(uploaded_file, skiprows=skip_rows, header=None, sep=None, engine='python', encoding=enc)
            
            # 文字列を数値に変換（変換できない行はNaNにして削除）
            df_numeric = df.apply(pd.to_numeric, errors='coerce').dropna()
            
            if df_numeric.shape[0] > 0:
                return df_numeric
        except Exception:
            continue
    return None

def main():
    st.set_page_config(page_title="ESR Analyzer Ultimate", layout="wide")
    st.title("🧲 ESR Spectrum Analyzer (1列データ対応版)")

    # --- サイドバー：読み込み設定 ---
    st.sidebar.header("1. データ読み込み設定")
    uploaded_file = st.sidebar.file_uploader("データファイルをアップロード", type=['txt', 'csv', 'dat'])
    
    # ヘッダー行数の調整（重要）
    st.sidebar.caption("※読み込みエラーが出る場合、ここを調整してください")
    skip_head = st.sidebar.number_input("ヘッダー行数 (スキップ)", value=0, min_value=0, step=1, help="データの冒頭にあるテキスト行の数。Gnuplotに'80行目から'とあった場合は約80にしてください。")

    st.sidebar.markdown("---")
    st.sidebar.header("2. 測定パラメータ")
    freq_ghz = st.sidebar.number_input("測定周波数 (GHz)", value=9.450, format="%.4f")
    
    # 磁場範囲の設定（1列データ用）
    st.sidebar.subheader("磁場軸 (X軸) の設定")
    st.sidebar.caption("※データが「強度のみ」の場合に使われます")
    manual_x_start = st.sidebar.number_input("開始磁場 (mT)", value=270.0)
    manual_x_range = st.sidebar.number_input("掃引幅 (Range) (mT)", value=100.0)
    
    st.sidebar.markdown("---")
    st.sidebar.header("3. 解析オプション")
    peak_prominence = st.sidebar.slider("ピーク検出感度", 0.01, 1.0, 0.1)
    do_baseline = st.sidebar.checkbox("ベースライン補正", value=True)

    if uploaded_file is not None:
        # データ読み込み実行
        df = load_data_robust(uploaded_file, skip_head)
        
        if df is None:
            st.error("エラー: データを読み込めませんでした。ヘッダー行数を増やしてみてください。")
        else:
            st.success(f"読み込み成功: {len(df)} 行のデータ")
            
            # --- データの列数判定とX軸生成 ---
            raw_field = None
            signal = None
            
            if df.shape[1] >= 2:
                # 2列以上ある場合（1列目=磁場、2列目=強度 とみなす）
                st.info("💡 2列のデータを検出しました (X:磁場, Y:強度)")
                raw_field = df.iloc[:, 0].values
                signal = df.iloc[:, 1].values
            else:
                # 1列しかない場合（強度のみ → X軸を作成）
                st.warning("⚠️ 1列のデータ（強度のみ）を検出しました。サイドバーの設定値で磁場軸を生成します。")
                signal = df.iloc[:, 0].values
                # linspaceでX軸を作る
                manual_x_end = manual_x_start + manual_x_range
                raw_field = np.linspace(manual_x_start, manual_x_end, len(signal))

            # --- 解析処理 ---
            # ベースライン補正
            if do_baseline:
                baseline = np.linspace(signal[0], signal[-1], len(signal))
                signal = signal - baseline

            # ピーク検出
            peaks_pos, _ = find_peaks(signal, prominence=peak_prominence * np.max(signal))
            peaks_neg, _ = find_peaks(-signal, prominence=peak_prominence * np.max(np.abs(signal)))
            all_peak_indices = np.sort(np.concatenate([peaks_pos, peaks_neg]))

            # --- グラフ表示 ---
            col1, col2 = st.columns([2, 1])

            with col1:
                st.subheader("スペクトル (1次微分)")
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(raw_field, signal, color='black', lw=1.2, label='Signal')
                
                # ピークプロット
                if len(all_peak_indices) > 0:
                    ax.scatter(raw_field[all_peak_indices], signal[all_peak_indices], color='red', zorder=5)
                
                ax.set_xlabel("Magnetic Field (mT)")
                ax.set_ylabel("Intensity (a.u.)")
                ax.grid(True, linestyle=':', alpha=0.6)
                ax.legend()
                st.pyplot(fig)
                
                # 積分波形
                st.subheader("吸収波形 (積分)")
                abs_signal = cumulative_trapezoid(signal, raw_field, initial=0)
                fig2, ax2 = plt.subplots(figsize=(10, 3))
                ax2.fill_between(raw_field, abs_signal, color='forestgreen', alpha=0.3)
                ax2.plot(raw_field, abs_signal, color='forestgreen')
                ax2.set_xlabel("Magnetic Field (mT)")
                st.pyplot(fig2)

            with col2:
                st.subheader("📊 解析結果")
                
                if len(peaks_pos) > 0 and len(peaks_neg) > 0:
                    # g値 (最大-最小の中心)
                    f_max = raw_field[peaks_pos[np.argmax(signal[peaks_pos])]]
                    f_min = raw_field[peaks_neg[np.argmax(-signal[peaks_neg])]]
                    center_field = (f_max + f_min) / 2
                    g_val = calculate_g_factor(center_field, freq_ghz)
                    
                    st.metric("中心 g値", f"{g_val:.5f}")
                    st.metric("中心磁場", f"{center_field:.2f} mT")
                    st.metric("線幅 ΔHpp", f"{abs(f_max - f_min):.2f} mT")
                
                st.divider()
                st.write("**ハイパーファイン分裂 ($A$)**")
                
                if len(all_peak_indices) >= 2:
                    hf_list = []
                    for i in range(len(all_peak_indices) - 1):
                        idx1 = all_peak_indices[i]
                        idx2 = all_peak_indices[i+1]
                        
                        dist = abs(raw_field[idx1] - raw_field[idx2])
                        # A値換算
                        avg_f = (raw_field[idx1] + raw_field[idx2]) / 2
                        curr_g = calculate_g_factor(avg_f, freq_ghz)
                        a_mhz = curr_g * BOHR_MAGNETON * (dist * 1e-3) / H_PLANCK / 1e6
                        
                        hf_list.append({
                            "Pair": f"{i+1}-{i+2}",
                            "幅 (mT)": f"{dist:.2f}",
                            "A (MHz)": f"{a_mhz:.1f}"
                        })
                    st.table(pd.DataFrame(hf_list))
                else:
                    st.caption("ピークが2つ以上検出されませんでした。")

if __name__ == "__main__":
    main()