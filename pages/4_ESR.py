import streamlit as st
import pandas as pd
import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
import re
import plotly.graph_objects as go

# --- 定数 ---
H_PLANCK = 6.62607015e-34
BOHR_MAGNETON = 9.27401007e-24

def calculate_g_factor(magnetic_field_mt, frequency_ghz):
    if magnetic_field_mt == 0: return 0
    b_tesla = magnetic_field_mt * 1e-3
    freq_hz = frequency_ghz * 1e9
    g = (H_PLANCK * freq_hz) / (BOHR_MAGNETON * b_tesla)
    return g

# --- フィッティング用関数 (ローレンツ関数の1次微分) ---
def lorentzian_derivative(x, amp, center, width):
    """
    x: 磁場
    amp: 振幅係数
    center: 中心磁場 (B0)
    width: 半値半幅に近いパラメータ (HWHM)
    
    式: y = - A * (x - x0) / ( w^2 + (x - x0)^2 )^2
    ※この定義の場合、ピーク間幅 Delta_Hpp = 2 * width / sqrt(3) となる
    """
    return -amp * (x - center) / ((width**2) + (x - center)**2)**2

def main():
    st.set_page_config(page_title="ESR Analyzer with Fitting", layout="wide")
    st.title("🧲 ESR Analyzer (Curve Fitting Edition)")

    # --- サイドバー：読み込み設定 ---
    st.sidebar.header("1. 読み込み範囲設定")
    default_start = 80
    default_end = 65615
    start_line = st.sidebar.number_input("データ開始行", value=default_start, min_value=1)
    end_line = st.sidebar.number_input("データ終了行", value=default_end, min_value=1)
    
    st.sidebar.markdown("---")
    st.sidebar.header("2. 磁場パラメータ (X軸)")
    x_min = st.sidebar.number_input("X-range min (mT)", value=295.0, format="%.4f")
    x_range = st.sidebar.number_input("X-range (mT)", value=50.0, format="%.4f")
    
    st.sidebar.markdown("---")
    st.sidebar.header("3. 解析・Fitting設定")
    freq_ghz = st.sidebar.number_input("測定周波数 (GHz)", value=9.450, format="%.4f")
    
    # フィッティング有効化スイッチ
    do_fitting = st.sidebar.checkbox("カーブフィッティングを実行", value=False)
    
    st.sidebar.markdown("---")
    peak_prominence = st.sidebar.slider("ピーク検出感度", 0.01, 1.0, 0.1)
    do_baseline = st.sidebar.checkbox("ベースライン補正", value=True)

    # --- メインエリア：ファイルアップロード ---
    uploaded_file = st.file_uploader("データファイル (.txt) をアップロード", type=['txt', 'csv', 'dat'])

    if uploaded_file is not None:
        try:
            # 1. ファイル読み込み
            content_bytes = uploaded_file.read()
            try:
                content_text = content_bytes.decode('cp932')
            except UnicodeDecodeError:
                content_text = content_bytes.decode('utf-8', errors='ignore')
            lines = content_text.splitlines()

            # ヘッダー情報表示
            st.info("ℹ️ ファイルヘッダー情報")
            header_col1, header_col2, header_col3 = st.columns(3)
            if len(lines) >= 7:
                with header_col1: st.text(f"4行目: {lines[3].strip()}")
                with header_col2: st.text(f"6行目: {lines[5].strip()}")
                with header_col3: st.text(f"7行目: {lines[6].strip()}")

            # データ抽出
            idx_start = start_line - 1
            idx_end = end_line
            if idx_start < 0 or idx_end > len(lines):
                st.error("行指定が範囲外です。")
                return

            raw_data_lines = lines[idx_start:idx_end]
            y_values = []
            for line in raw_data_lines:
                line = line.strip()
                if not line: continue
                try:
                    parts = re.split(r'[,\s\t]+', line)
                    val = float(parts[0])
                    y_values.append(val)
                except ValueError: continue

            signal = np.array(y_values)
            n_points = len(signal)
            if n_points == 0:
                st.error("データが見つかりません。")
                return

            # X軸生成
            incr = x_range / n_points
            field = x_min + np.arange(n_points) * incr
            st.caption(f"🔧 Data Points: {n_points}, Incr: {incr:.5e} mT")

            # ベースライン補正
            if do_baseline:
                baseline = np.linspace(signal[0], signal[-1], n_points)
                signal = signal - baseline

            # ピーク検出 (簡易)
            max_amp = np.max(np.abs(signal))
            peaks_pos, _ = find_peaks(signal, prominence=peak_prominence * max_amp)
            peaks_neg, _ = find_peaks(-signal, prominence=peak_prominence * max_amp)
            all_peak_indices = np.sort(np.concatenate([peaks_pos, peaks_neg]))

            # --- フィッティング処理 ---
            fit_y = None
            popt = None
            r_squared = None
            
            if do_fitting:
                try:
                    # 初期値の推定
                    # 中心: 最大と最小の中点
                    if len(peaks_pos) > 0 and len(peaks_neg) > 0:
                        idx_max = peaks_pos[np.argmax(signal[peaks_pos])]
                        idx_min = peaks_neg[np.argmax(-signal[peaks_neg])]
                        init_center = (field[idx_max] + field[idx_min]) / 2
                        init_width_pp = abs(field[idx_max] - field[idx_min])
                    else:
                        init_center = np.mean(field)
                        init_width_pp = x_range / 10

                    # モデル式の width パラメータに換算 (width = Delta_Hpp * sqrt(3) / 2)
                    init_w_param = init_width_pp * np.sqrt(3) / 2
                    
                    # 振幅の推定 (概算)
                    init_amp = np.max(np.abs(signal)) * (init_w_param**3) # 次元の辻褄合わせの係数

                    p0 = [init_amp, init_center, init_w_param]
                    
                    # カーブフィッティング実行
                    # bounds: 振幅は正負ありうる, 中心は範囲内, 幅は正
                    popt, pcov = curve_fit(lorentzian_derivative, field, signal, p0=p0, maxfev=10000)
                    
                    fit_amp, fit_center, fit_w_param = popt
                    fit_y = lorentzian_derivative(field, *popt)

                    # R2値 (決定係数) の計算
                    residuals = signal - fit_y
                    ss_res = np.sum(residuals**2)
                    ss_tot = np.sum((signal - np.mean(signal))**2)
                    r_squared = 1 - (ss_res / ss_tot)

                except Exception as e:
                    st.warning(f"フィッティングに失敗しました: {e}")

            # --- グラフ描画 ---
            col1, col2 = st.columns([2, 1])

            with col1:
                st.subheader("スペクトル解析")
                fig = go.Figure()
                
                # 生データ
                fig.add_trace(go.Scatter(
                    x=field, y=signal, mode='lines', name='Experimental',
                    line=dict(color='black', width=1.5),
                    hovertemplate='Exp<br>B: %{x:.2f}<br>I: %{y:.2f}<extra></extra>'
                ))
                
                # フィッティング結果
                if fit_y is not None:
                    fig.add_trace(go.Scatter(
                        x=field, y=fit_y, mode='lines', name='Fitted (Lorentzian)',
                        line=dict(color='orange', width=2.0, dash='solid'),
                        hovertemplate='Fit<br>B: %{x:.2f}<br>I: %{y:.2f}<extra></extra>'
                    ))
                    # 残差プロット（オプション：コメントアウトを外せば表示可）
                    # fig.add_trace(go.Scatter(x=field, y=signal-fit_y, mode='lines', name='Residual', line=dict(color='gray', width=0.5)))

                # ピークマーカー
                if len(all_peak_indices) > 0:
                    fig.add_trace(go.Scatter(
                        x=field[all_peak_indices], y=signal[all_peak_indices],
                        mode='markers', name='Peaks',
                        marker=dict(color='red', size=8, symbol='circle-open')
                    ))

                fig.update_layout(
                    xaxis_title="Magnetic Field (mT)", yaxis_title="Intensity",
                    height=500, margin=dict(l=40, r=40, t=20, b=40),
                    legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.8)')
                )
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.subheader("📊 解析結果")
                
                if popt is not None:
                    st.markdown("### ✅ Fitting Result")
                    
                    # パラメータ計算
                    f_center_fit = popt[1]
                    f_width_param = abs(popt[2])
                    
                    # ピーク間幅 Delta Hpp = 2 * w / sqrt(3)
                    delta_hpp_fit = 2 * f_width_param / np.sqrt(3)
                    
                    # g値換算
                    g_fit = calculate_g_factor(f_center_fit, freq_ghz)
                    
                    st.metric("g値 (Fitted)", f"{g_fit:.6f}")
                    st.metric("中心磁場 (B0)", f"{f_center_fit:.3f} mT")
                    st.metric("線幅 (ΔHpp)", f"{delta_hpp_fit:.3f} mT")
                    st.metric("決定係数 (R²)", f"{r_squared:.4f}")
                    
                    if r_squared < 0.9:
                        st.warning("⚠️ フィッティング精度が低いです。データがローレンツ型ではないか、ノイズが多い可能性があります。")
                
                elif len(peaks_pos) > 0 and len(peaks_neg) > 0:
                    st.markdown("### 🔹 Peak Picking Result")
                    # 単純な最大最小法
                    idx_max = peaks_pos[np.argmax(signal[peaks_pos])]
                    idx_min = peaks_neg[np.argmax(-signal[peaks_neg])]
                    f_max, f_min = field[idx_max], field[idx_min]
                    
                    center_simple = (f_max + f_min)/2
                    g_simple = calculate_g_factor(center_simple, freq_ghz)
                    
                    st.metric("g値 (Peak-to-Peak)", f"{g_simple:.5f}")
                    st.metric("線幅 (ΔHpp)", f"{abs(f_max - f_min):.3f} mT")
                
                st.divider()
                st.caption("Fittingモデル: ローレンツ関数1次微分")

        except Exception as e:
            st.error(f"エラー: {e}")

if __name__ == "__main__":
    main()