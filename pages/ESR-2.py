import streamlit as st
import pandas as pd
import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
import re
import plotly.graph_objects as go
import plotly.express as px

# --- 定数 ---
H_PLANCK = 6.62607015e-34
BOHR_MAGNETON = 9.27401007e-24

def calculate_g_factor(magnetic_field_mt, frequency_ghz):
    if magnetic_field_mt == 0: return 0
    b_tesla = magnetic_field_mt * 1e-3
    freq_hz = frequency_ghz * 1e9
    g = (H_PLANCK * freq_hz) / (BOHR_MAGNETON * b_tesla)
    return g

def lorentzian_derivative(x, amp, center, width):
    return -amp * (x - center) / ((width**2) + (x - center)**2)**2

# --- ヘッダー解析関数 ---
def parse_header_params(lines):
    """
    ファイルのヘッダー行から x-range min, x-range を探す。
    見つからなければ None を返す。
    """
    params = {}
    
    # 探索するキーワードと正規表現
    # 例: "x-range min = 295" または "x-range min=295" などに対応
    patterns = {
        "x_min": r"x-range\s*min\s*=\s*([0-9\.]+)",
        "x_range": r"x-range\s*=\s*([0-9\.]+)"
    }

    # 最初の20行くらいを走査
    header_check_limit = 20
    for i in range(min(len(lines), header_check_limit)):
        line = lines[i].strip()
        for key, pattern in patterns.items():
            if key not in params:
                match = re.search(pattern, line, re.IGNORECASE)
                if match:
                    try:
                        params[key] = float(match.group(1))
                    except:
                        pass
    
    return params.get("x_min"), params.get("x_range")

def main():
    st.set_page_config(page_title="ESR Multi-Plot Analyzer", layout="wide")
    st.title("🧲 ESR Multi-Spectrum Analyzer (重ね書き対応)")

    # --- サイドバー：読み込み設定 ---
    st.sidebar.header("1. 読み込み共通設定")
    
    # 読み込み行のデフォルト
    default_start = 80
    default_end = 65615
    start_line = st.sidebar.number_input("データ開始行", value=default_start, min_value=1)
    end_line = st.sidebar.number_input("データ終了行", value=default_end, min_value=1)

    st.sidebar.caption("※ヘッダーから磁場範囲(x-range)を自動取得しますが、取得できない場合は以下のデフォルト値を使用します。")
    fallback_xmin = st.sidebar.number_input("デフォルト X-min (mT)", value=295.0, format="%.4f")
    fallback_xrange = st.sidebar.number_input("デフォルト X-range (mT)", value=50.0, format="%.4f")
    
    st.sidebar.markdown("---")
    st.sidebar.header("2. 表示オプション")
    do_normalize = st.sidebar.checkbox("正規化 (Normalize)", value=False, help="最大強度を1に揃えます")
    y_offset = st.sidebar.slider("Y軸オフセット (Waterfall)", 0.0, 2.0, 0.0, step=0.1, help="波形を縦にずらして表示します")
    
    freq_ghz = st.sidebar.number_input("測定周波数 (GHz)", value=9.450, format="%.4f")

    # --- メインエリア：複数ファイルアップロード ---
    uploaded_files = st.file_uploader(
        "データファイルをアップロード (複数選択可)", 
        type=['txt', 'csv', 'dat'], 
        accept_multiple_files=True
    )

    if uploaded_files:
        # 全データの格納用リスト
        dataset_list = []

        # --- 各ファイルをループ処理 ---
        for u_file in uploaded_files:
            try:
                # 読み込み
                u_file.seek(0)
                content_bytes = u_file.read()
                try:
                    content_text = content_bytes.decode('cp932')
                except UnicodeDecodeError:
                    content_text = content_bytes.decode('utf-8', errors='ignore')
                
                lines = content_text.splitlines()

                # 1. ヘッダーからパラメータ自動取得
                auto_xmin, auto_xrange = parse_header_params(lines)
                
                # 自動取得できなければサイドバーの値を使う
                current_xmin = auto_xmin if auto_xmin is not None else fallback_xmin
                current_xrange = auto_xrange if auto_xrange is not None else fallback_xrange

                # 2. データ抽出
                idx_start = start_line - 1
                idx_end = end_line
                
                if idx_start < 0 or idx_end > len(lines):
                    continue # 行数不足ならスキップ

                raw_lines = lines[idx_start:idx_end]
                vals = []
                for ln in raw_lines:
                    ln = ln.strip()
                    if not ln: continue
                    try:
                        parts = re.split(r'[,\s\t]+', ln)
                        vals.append(float(parts[0]))
                    except: continue
                
                signal = np.array(vals)
                n_points = len(signal)
                
                if n_points == 0: continue

                # 3. X軸生成 (Incr = Range / Points)
                incr = current_xrange / n_points
                field = current_xmin + np.arange(n_points) * incr

                # データをリストに追加
                dataset_list.append({
                    "filename": u_file.name,
                    "field": field,
                    "signal": signal,
                    "xmin": current_xmin,
                    "xrange": current_xrange
                })

            except Exception as e:
                st.error(f"{u_file.name} の読み込みに失敗: {e}")

        # --- 重ね書きグラフの描画 ---
        if len(dataset_list) > 0:
            st.subheader("📈 スペクトル重ね書き (Overlay)")
            
            fig = go.Figure()
            colors = px.colors.qualitative.Plotly # 色パレット

            for i, data in enumerate(dataset_list):
                y_data = data["signal"]
                
                # 正規化処理
                if do_normalize:
                    max_val = np.max(np.abs(y_data))
                    if max_val > 0:
                        y_data = y_data / max_val
                
                # オフセット処理 (新しいファイルほど上にずらす、あるいは下にずらす)
                # ここでは単純に i * offset
                display_y = y_data + (i * y_offset)

                fig.add_trace(go.Scatter(
                    x=data["field"],
                    y=display_y,
                    mode='lines',
                    name=data["filename"],
                    line=dict(width=1.5),
                    hovertemplate=f"<b>{data['filename']}</b><br>B: %{{x:.2f}}<br>I: %{{y:.3f}}<extra></extra>"
                ))

            fig.update_layout(
                xaxis_title="Magnetic Field (mT)",
                yaxis_title="Intensity (Normalized/Offset)" if do_normalize or y_offset > 0 else "Intensity (a.u.)",
                height=600,
                legend=dict(x=1.02, y=1, xanchor='left', yanchor='top'),
                margin=dict(r=150) # 凡例のために右マージンを空ける
            )
            st.plotly_chart(fig, use_container_width=True)

            # --- 個別解析セクション ---
            st.divider()
            st.subheader("🔍 個別スペクトルの詳細解析")
            
            # 解析対象を選択
            filenames = [d["filename"] for d in dataset_list]
            selected_name = st.selectbox("解析するファイルを選択", filenames)
            
            # 選択されたデータを取り出す
            target_data = next((d for d in dataset_list if d["filename"] == selected_name), None)

            if target_data:
                field = target_data["field"]
                signal = target_data["signal"]
                
                # 解析用設定
                col_opt1, col_opt2 = st.columns(2)
                with col_opt1:
                    peak_prominence = st.slider("ピーク検出感度", 0.01, 1.0, 0.1, key="prominence")
                with col_opt2:
                    do_fitting = st.checkbox("カーブフィッティング (Lorentzian)", value=False, key="fitting")

                # ベースライン補正（個別解析時のみ適用）
                baseline = np.linspace(signal[0], signal[-1], len(signal))
                signal_corrected = signal - baseline

                # ピーク検出
                max_amp = np.max(np.abs(signal_corrected))
                peaks_pos, _ = find_peaks(signal_corrected, prominence=peak_prominence * max_amp)
                peaks_neg, _ = find_peaks(-signal_corrected, prominence=peak_prominence * max_amp)
                
                # --- フィッティング ---
                popt = None
                fit_y = None
                r2 = None
                
                if do_fitting and len(peaks_pos) > 0 and len(peaks_neg) > 0:
                    try:
                        # 初期値推定
                        idx_max = peaks_pos[np.argmax(signal_corrected[peaks_pos])]
                        idx_min = peaks_neg[np.argmax(-signal_corrected[peaks_neg])]
                        init_center = (field[idx_max] + field[idx_min]) / 2
                        init_width = abs(field[idx_max] - field[idx_min]) * np.sqrt(3) / 2
                        init_amp = np.max(np.abs(signal_corrected)) * (init_width**3) * 5 # 係数調整

                        p0 = [init_amp, init_center, init_width]
                        popt, _ = curve_fit(lorentzian_derivative, field, signal_corrected, p0=p0, maxfev=5000)
                        
                        fit_y = lorentzian_derivative(field, *popt)
                        
                        # R2
                        ss_res = np.sum((signal_corrected - fit_y)**2)
                        ss_tot = np.sum((signal_corrected - np.mean(signal_corrected))**2)
                        r2 = 1 - (ss_res / ss_tot)
                    except:
                        st.warning("フィッティングに失敗しました。")

                # --- 結果表示 ---
                res_col1, res_col2 = st.columns([2, 1])
                
                with res_col1:
                    fig_single = go.Figure()
                    fig_single.add_trace(go.Scatter(x=field, y=signal_corrected, name="Raw (Baseline Corrected)", line=dict(color='black')))
                    if fit_y is not None:
                        fig_single.add_trace(go.Scatter(x=field, y=fit_y, name="Fit", line=dict(color='orange', width=2)))
                    
                    # ピーク
                    all_peaks = np.concatenate([peaks_pos, peaks_neg])
                    if len(all_peaks) > 0:
                        fig_single.add_trace(go.Scatter(x=field[all_peaks], y=signal_corrected[all_peaks], mode='markers', name='Peaks', marker=dict(color='red')))

                    fig_single.update_layout(height=400, xaxis_title="Magnetic Field (mT)", margin=dict(l=20, r=20, t=20, b=20))
                    st.plotly_chart(fig_single, use_container_width=True)

                with res_col2:
                    st.markdown(f"**ファイル:** `{selected_name}`")
                    st.caption(f"X-range: {target_data['xmin']} 〜 {target_data['xmin']+target_data['xrange']} mT")

                    if popt is not None:
                        f_center = popt[1]
                        width_param = abs(popt[2])
                        delta_hpp = 2 * width_param / np.sqrt(3)
                        g_val = calculate_g_factor(f_center, freq_ghz)
                        
                        st.success("✅ Fitting Result")
                        st.metric("g値", f"{g_val:.5f}")
                        st.metric("ΔHpp (mT)", f"{delta_hpp:.3f}")
                        st.metric("R² (一致度)", f"{r2:.4f}")
                    
                    elif len(peaks_pos) > 0 and len(peaks_neg) > 0:
                        idx_max = peaks_pos[np.argmax(signal_corrected[peaks_pos])]
                        idx_min = peaks_neg[np.argmax(-signal_corrected[peaks_neg])]
                        f_pp = abs(field[idx_max] - field[idx_min])
                        c_pp = (field[idx_max] + field[idx_min]) / 2
                        g_pp = calculate_g_factor(c_pp, freq_ghz)
                        
                        st.info("🔹 Peak-to-Peak Result")
                        st.metric("g値 (仮)", f"{g_pp:.5f}")
                        st.metric("ΔHpp (mT)", f"{f_pp:.3f}")

    else:
        st.info("👈 サイドバーからファイルをアップロードしてください（複数選択可能です）。")

if __name__ == "__main__":
    main()