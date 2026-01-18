import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.signal import find_peaks
import re

# --- 定数 ---
H_PLANCK = 6.62607015e-34
BOHR_MAGNETON = 9.27401007e-24

def calculate_g_factor(magnetic_field_mt, frequency_ghz):
    if magnetic_field_mt == 0: return 0
    b_tesla = magnetic_field_mt * 1e-3
    freq_hz = frequency_ghz * 1e9
    g = (H_PLANCK * freq_hz) / (BOHR_MAGNETON * b_tesla)
    return g

def main():
    st.set_page_config(page_title="ESR Analyzer Special", layout="wide")
    st.title("🧲 ESR Spectrum Analyzer (カスタム読み込み版)")

    # --- サイドバー：読み込み設定 ---
    st.sidebar.header("1. 読み込み範囲設定")
    
    # デフォルト値をリクエスト通りに設定
    # ユーザーにとっての「80行目」は、プログラム(0始まり)ではインデックス79
    default_start = 80
    default_end = 65615

    start_line = st.sidebar.number_input("データ開始行 (行番号)", value=default_start, min_value=1)
    end_line = st.sidebar.number_input("データ終了行 (行番号)", value=default_end, min_value=1)
    
    st.sidebar.markdown("---")
    st.sidebar.header("2. 磁場パラメータ (X軸)")
    # デフォルト値
    default_xmin = 270.0
    default_xrange = 100.0
    
    x_min = st.sidebar.number_input("X-range min (mT)", value=default_xmin, format="%.2f")
    x_range = st.sidebar.number_input("X-range (mT)", value=default_xrange, format="%.2f")
    
    st.sidebar.markdown("---")
    st.sidebar.header("3. その他設定")
    freq_ghz = st.sidebar.number_input("測定周波数 (GHz)", value=9.450, format="%.4f")
    peak_prominence = st.sidebar.slider("ピーク検出感度", 0.01, 1.0, 0.1)
    do_baseline = st.sidebar.checkbox("ベースライン補正", value=True)

    # --- メインエリア：ファイルアップロード ---
    uploaded_file = st.file_uploader("データファイル (.txt) をアップロード", type=['txt', 'csv', 'dat'])

    if uploaded_file is not None:
        try:
            # 1. ファイルを行ごとに読み込む (Shift-JIS/cp932対応)
            # バイナリとして読んでデコード
            content_bytes = uploaded_file.read()
            
            # 文字コード判定（cp932優先、だめならutf-8）
            try:
                content_text = content_bytes.decode('cp932')
            except UnicodeDecodeError:
                content_text = content_bytes.decode('utf-8', errors='ignore')
            
            lines = content_text.splitlines()

            # 2. ヘッダー情報の確認 (4, 6, 7行目)
            # Pythonのリストは0始まりなので、4行目はindex 3
            st.info("ℹ️ ファイルヘッダー情報 (確認用)")
            header_col1, header_col2, header_col3 = st.columns(3)
            
            # 安全にアクセスするために長さをチェック
            if len(lines) >= 7:
                # 行の内容を表示（前後の空白除去）
                with header_col1:
                    st.text(f"4行目: {lines[3].strip()}")
                with header_col2:
                    st.text(f"6行目: {lines[5].strip()}")
                with header_col3:
                    st.text(f"7行目: {lines[6].strip()}")
            else:
                st.warning("ファイル行数が短いため、指定のヘッダー行を確認できませんでした。")

            # 3. データ部分の抽出
            # ユーザー指定の行番号をPythonのインデックスに変換（-1する）
            idx_start = start_line - 1
            idx_end = end_line  # スライスは終了を含まないのでそのまま

            if idx_start < 0 or idx_end > len(lines):
                st.error(f"指定された行範囲 ( {start_line} 〜 {end_line} ) がファイル行数を超えています。")
                return

            raw_data_lines = lines[idx_start:idx_end]
            
            # 数値変換処理
            y_values = []
            for line in raw_data_lines:
                line = line.strip()
                if not line: continue  # 空行はスキップ
                try:
                    # 数値としてパース (区切り文字がある場合は最初の要素を取得)
                    # "123.45" や "123.45  0.00" のような形式に対応
                    parts = re.split(r'[,\s\t]+', line)
                    val = float(parts[0])
                    y_values.append(val)
                except ValueError:
                    continue # 数値変換できない行は無視

            signal = np.array(y_values)
            n_points = len(signal)

            if n_points == 0:
                st.error("指定範囲内に有効な数値データが見つかりませんでした。")
                return

            st.success(f"データ読み込み成功: {n_points} 点 (行 {start_line} 〜 {end_line})")

            # 4. X軸 (磁場) の生成
            # x_min から x_min + x_range までを等分割
            # Gnuplotの仕様に合わせて生成
            field = np.linspace(x_min, x_min + x_range, n_points)

            # --- 解析処理 (既存ロジック) ---
            
            # ベースライン補正
            if do_baseline:
                baseline = np.linspace(signal[0], signal[-1], n_points)
                signal = signal - baseline

            # ピーク検出
            # 信号の絶対値の最大値を基準にプロミネンスを設定
            max_amp = np.max(np.abs(signal))
            if max_amp == 0: max_amp = 1.0 # ゼロ除算防止
            
            peaks_pos, _ = find_peaks(signal, prominence=peak_prominence * max_amp)
            peaks_neg, _ = find_peaks(-signal, prominence=peak_prominence * max_amp)
            all_peak_indices = np.sort(np.concatenate([peaks_pos, peaks_neg]))

            # --- グラフ表示 ---
            col1, col2 = st.columns([2, 1])

            with col1:
                st.subheader("スペクトル (1次微分)")
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(field, signal, color='blue', lw=1.0, label='Signal')
                
                # ピークプロット
                if len(all_peak_indices) > 0:
                    ax.scatter(field[all_peak_indices], signal[all_peak_indices], color='red', s=20, zorder=5)
                
                ax.set_xlabel("Magnetic Field (mT)")
                ax.set_ylabel("Intensity (a.u.)")
                ax.set_xlim(x_min, x_min + x_range) # X軸範囲を固定
                ax.grid(True, linestyle=':', alpha=0.6)
                ax.legend()
                st.pyplot(fig)
                
                # 積分波形
                st.subheader("吸収波形 (積分)")
                abs_signal = cumulative_trapezoid(signal, field, initial=0)
                fig2, ax2 = plt.subplots(figsize=(10, 3))
                ax2.fill_between(field, abs_signal, color='green', alpha=0.3)
                ax2.plot(field, abs_signal, color='green', lw=1)
                ax2.set_xlabel("Magnetic Field (mT)")
                ax2.set_xlim(x_min, x_min + x_range)
                st.pyplot(fig2)

            with col2:
                st.subheader("📊 解析結果")
                
                if len(peaks_pos) > 0 and len(peaks_neg) > 0:
                    # g値 (最大-最小の中心)
                    # 最も強度の高いペアを探す簡易ロジック
                    idx_max_int = peaks_pos[np.argmax(signal[peaks_pos])]
                    idx_min_int = peaks_neg[np.argmax(-signal[peaks_neg])]
                    
                    f_max = field[idx_max_int]
                    f_min = field[idx_min_int]
                    
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
                        
                        dist = abs(field[idx1] - field[idx2])
                        
                        # A値換算
                        avg_f = (field[idx1] + field[idx2]) / 2
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

        except Exception as e:
            st.error(f"エラーが発生しました: {e}")

if __name__ == "__main__":
    main()