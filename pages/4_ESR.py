import streamlit as st
import pandas as pd
import numpy as np
# NumPy 2.0対応
from scipy.integrate import cumulative_trapezoid, trapezoid 
from scipy.signal import find_peaks, convolve
from scipy.optimize import curve_fit
import re
import plotly.graph_objects as go

# --- 定数 ---
H_PLANCK = 6.62607015e-34
BOHR_MAGNETON = 9.27401007e-24

# --- メモ定数 ---
DEFAULT_MEMO = """
### 📌 解析パラメータ・メモ
**ファイルごとのパラメータ値の入力ルール**
* **4行目**: data length
* **6行目**: x-range min
* **7行目**: x-range

**Gain自動読み取りについて**
* ファイルヘッダー内に `amplitude` または `gain` という項目があれば、それを「試料のGain」として自動取得します。
"""

# --- 物理計算関数 ---
def calculate_g_factor(magnetic_field_mt, frequency_ghz):
    if magnetic_field_mt == 0: return 0
    b_tesla = magnetic_field_mt * 1e-3
    freq_hz = frequency_ghz * 1e9
    g = (H_PLANCK * freq_hz) / (BOHR_MAGNETON * b_tesla)
    return g

def calculate_field_from_g(g_value, frequency_ghz):
    if g_value == 0: return 0
    freq_hz = frequency_ghz * 1e9
    b_tesla = (H_PLANCK * freq_hz) / (BOHR_MAGNETON * g_value)
    return b_tesla * 1e3

def lorentzian_derivative(x, amp, center, width):
    return -amp * (x - center) / ((width**2) + (x - center)**2)**2

# --- シミュレーション関数 ---
def generate_isotope_pattern(n_nuclei, spin_I):
    if spin_I == 0.5:
        base = np.array([1, 1])
    elif spin_I == 1.0:
        base = np.array([1, 1, 1])
    else:
        len_vec = int(2*spin_I + 1)
        base = np.ones(len_vec)
    pattern = np.array([1.0])
    for _ in range(n_nuclei):
        pattern = convolve(pattern, base)
    return pattern

def simulate_isotropic(x_axis, g_val, freq, width_mT, a_val_mT, n_nuclei, spin_I):
    center_field = calculate_field_from_g(g_val, freq)
    intensities = generate_isotope_pattern(n_nuclei, spin_I)
    total_spin_len = len(intensities)
    indices = np.arange(total_spin_len) - (total_spin_len - 1) / 2
    peak_positions = center_field + indices * a_val_mT
    
    y_sim = np.zeros_like(x_axis)
    w_param = width_mT * np.sqrt(3) / 2
    amp_factor = 1.0 / np.max(intensities) * (w_param**2) * 5 
    for pos, intensity in zip(peak_positions, intensities):
        y_sim += lorentzian_derivative(x_axis, intensity * amp_factor, pos, w_param)
    if np.max(np.abs(y_sim)) > 0:
        y_sim = y_sim / np.max(np.abs(y_sim))
    return y_sim, peak_positions

# --- メインアプリ ---
def main():
    st.set_page_config(page_title="ESR Ultimate Analyzer", layout="wide")
    st.title("🧲 ESR Ultimate Analyzer (Gain自動取得版)")
    
    tab1, tab2, tab3 = st.tabs(["📊 実験データ解析 & 定量", "🧪 シミュレーション", "📝 メモ・測定条件"])

    # ==========================================
    # Tab 1: 実験データ解析 & 定量
    # ==========================================
    with tab1:
        st.header("実験データの解析・定量")
        
        with st.sidebar:
            st.header("1. [解析] 読み込み設定")
            start_line = st.number_input("開始行", value=80)
            end_line = st.number_input("終了行", value=65615)
            
            st.markdown("---")
            st.header("2. [解析] 磁場設定")
            x_min_in = st.number_input("X-min (mT)", value=295.0, format="%.2f")
            x_range_in = st.number_input("X-range (mT)", value=50.0, format="%.2f")
            freq_ghz = st.number_input("周波数 (GHz)", value=9.450, format="%.4f")
            do_baseline = st.checkbox("ベースライン補正", value=True)

        uploaded_file = st.file_uploader("データファイル (.txt) をアップロード", type=['txt', 'csv'])

        if uploaded_file:
            try:
                # ファイル読み込み
                content = uploaded_file.read()
                try:
                    text = content.decode('cp932')
                except:
                    text = content.decode('utf-8', errors='ignore')
                lines = text.splitlines()

                # --- ヘッダー自動解析 (パラメータ取得) ---
                auto_xmin = None
                auto_xrange = None
                auto_gain = None  # Gain格納用

                # 先頭50行程度を走査
                for i in range(min(50, len(lines))):
                    line_lower = lines[i].lower()
                    
                    # X-range min
                    if "x-range min" in line_lower:
                        m = re.search(r"=\s*([0-9\.]+)", line_lower)
                        if m: auto_xmin = float(m.group(1))
                    
                    # X-range (width)
                    if "x-range" in line_lower and "min" not in line_lower:
                        m = re.search(r"=\s*([0-9\.]+)", line_lower)
                        if m: auto_xrange = float(m.group(1))
                    
                    # Gain (Amplitude または Receiver Gain)
                    if "amplitude" in line_lower or "receiver gain" in line_lower:
                        # "amplitude = 100" のような形式を探す
                        m = re.search(r"=\s*([0-9\.]+)", line_lower)
                        if m: auto_gain = float(m.group(1))

                # 採用するパラメータ
                cur_xmin = auto_xmin if auto_xmin else x_min_in
                cur_xrange = auto_xrange if auto_xrange else x_range_in

                # --- データ抽出 ---
                idx_s = start_line - 1
                idx_e = end_line
                raw_data = []
                for ln in lines[idx_s:idx_e]:
                    ln = ln.strip()
                    if not ln: continue
                    try:
                        val = float(re.split(r'[,\s]+', ln)[0])
                        raw_data.append(val)
                    except: continue
                
                signal = np.array(raw_data)
                n_pts = len(signal)
                
                if n_pts > 0:
                    incr = cur_xrange / n_pts
                    field = cur_xmin + np.arange(n_pts) * incr
                    
                    if do_baseline:
                        baseline = np.linspace(signal[0], signal[-1], n_pts)
                        signal = signal - baseline

                    # 解析実行
                    integ1 = cumulative_trapezoid(signal, field, initial=0)
                    integ1 = integ1 - np.linspace(integ1[0], integ1[-1], n_pts)
                    area_val = trapezoid(integ1, field)

                    # --- グラフ描画 ---
                    col_g1, col_g2 = st.columns([2, 1])
                    with col_g1:
                        st.subheader("スペクトル")
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=field, y=signal, name='Signal', line=dict(color='black')))
                        fig.add_trace(go.Scatter(x=field, y=integ1, name='Absorption', line=dict(color='green', dash='dot'), visible='legendonly'))
                        fig.update_layout(height=450, xaxis_title="Magnetic Field (mT)", yaxis_title="Intensity")
                        st.plotly_chart(fig, use_container_width=True)

                    with col_g2:
                        st.subheader("📊 解析データ")
                        st.metric("Area (2回積分)", f"{area_val:.4e}")
                        
                        # --- Gain情報の表示 ---
                        if auto_gain:
                            st.success(f"ℹ️ ファイルからGain値を検出: {int(auto_gain)}")
                        
                        st.divider()
                        st.markdown("#### 🧪 スピン濃度定量")
                        
                        with st.form("quant_form"):
                            st.write("**試料情報**")
                            sample_mass = st.number_input("試料質量 (mg)", value=1.0, format="%.2f")
                            
                            st.write("**標準試料 (Standard)**")
                            std_area = st.number_input("標準のArea", value=1.0e5, format="%.2e")
                            std_spins = st.number_input("標準の総スピン数", value=1.0e15, format="%.2e")
                            
                            st.write("**Gain補正**")
                            # 補正チェックボックス
                            use_correction = st.checkbox("Gain補正を行う", value=(auto_gain is not None))
                            
                            col_c1, col_c2 = st.columns(2)
                            # 自動取得したGainがあればそれをデフォルト値にする
                            default_sample_gain = auto_gain if auto_gain else 100.0
                            
                            gain_sample = col_c1.number_input("試料のGain", value=default_sample_gain)
                            gain_std = col_c2.number_input("標準のGain", value=100.0)

                            calc_btn = st.form_submit_button("計算実行")
                        
                        if calc_btn:
                            factor = (gain_std / gain_sample) if use_correction else 1.0
                            n_sample_total = std_spins * (area_val / std_area) * factor
                            spin_conc = n_sample_total / (sample_mass * 1e-3)
                            
                            st.success(f"総スピン数: {n_sample_total:.2e} spins")
                            st.error(f"濃度: {spin_conc:.2e} spins/g")
                            if use_correction:
                                st.caption(f"Gain補正係数: {factor:.2f} (Std/Sample)")

            except Exception as e:
                st.error(f"解析エラー: {e}")

    # ==========================================
    # Tab 2 & 3: シミュレーション / メモ (変更なし)
    # ==========================================
    with tab2:
        st.header("ESR シミュレーション")
        col_sim_set, col_sim_plot = st.columns([1, 2])
        with col_sim_set:
            sim_freq = st.number_input("周波数 (GHz)", value=9.450, format="%.4f", key="sim_freq")
            sim_g = st.number_input("中心 g値", value=2.0060, format="%.5f")
            sim_width = st.number_input("線幅 (mT)", value=0.5, step=0.1)
            st.divider()
            nuc_type = st.radio("核スピン I", [0.5, 1.0], format_func=lambda x: "I=1/2" if x==0.5 else "I=1")
            sim_n = st.number_input("核の数 n", value=1, min_value=0)
            sim_a = st.number_input("A値 (mT)", value=1.5)
            st.divider()
            sim_center_mT = calculate_field_from_g(sim_g, sim_freq)
            sim_range = st.number_input("表示幅 (mT)", value=10.0)
        with col_sim_plot:
            x_axis_sim = np.linspace(sim_center_mT - sim_range/2, sim_center_mT + sim_range/2, 2000)
            y_sim, peaks_sim = simulate_isotropic(x_axis_sim, sim_g, sim_freq, sim_width, sim_a, int(sim_n), nuc_type)
            fig_sim = go.Figure()
            fig_sim.add_trace(go.Scatter(x=x_axis_sim, y=y_sim, name='Sim', line=dict(color='blue')))
            st.plotly_chart(fig_sim, use_container_width=True)

    with tab3:
        st.header("📝 メモ・測定条件")
        st.info("ℹ️ 解析ルール")
        st.markdown(DEFAULT_MEMO)
        st.text_area("Memo Pad", height=300)

if __name__ == "__main__":
    main()