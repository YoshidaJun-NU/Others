import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import io
import zipfile
from scipy.signal import find_peaks
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.colors as mcolors

# ---------------------------------------------------------
# 関数: JASCO形式等のファイル読み込み
# ---------------------------------------------------------
def load_spectral_data(uploaded_file):
    """
    JASCO形式のテキストファイルなどを読み込み、
    {'filename': str, 'x': array, 'ir': array, 'vcd': array, 'noise': array, 'head': df} を返す。
    """
    try:
<<<<<<< HEAD
        content = uploaded_file.getvalue().decode("utf-8", errors="ignore")
        lines = content.splitlines()
=======
        uploaded_file.seek(0)
        if sep == 'auto':
            try:
                df = pd.read_csv(uploaded_file, skiprows=skip_rows, header=None, engine='python', encoding=encoding)
                if df.shape[1] <= 1:
                    uploaded_file.seek(0)
                    df = pd.read_csv(uploaded_file, skiprows=skip_rows, header=None, sep=r'\s+', engine='python', encoding=encoding)
            except:
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, skiprows=skip_rows, header=None, sep=r'\s+', engine='python', encoding=encoding)
        else:
            df = pd.read_csv(uploaded_file, skiprows=skip_rows, header=None, sep=sep, engine='python', encoding=encoding)
        df = df.apply(pd.to_numeric, errors='coerce').dropna()
        return df
    except Exception:
        return None

def smooth_data(y, window_length=11, polyorder=3):
    try:
        if window_length > len(y): window_length = len(y) // 2 * 2 + 1 
        return savgol_filter(y, window_length, polyorder)
    except:
        return y

def detect_multiple_peaks(v, i, prominence_val=0.0):
    amplitude = np.max(i) - np.min(i)
    prom = amplitude * prominence_val if amplitude > 0 else None
    peaks_top_idx, _ = find_peaks(i, prominence=prom)
    peaks_top = [{"E": v[idx], "I": i[idx], "Type": "Anodic"} for idx in peaks_top_idx]
    peaks_btm_idx, _ = find_peaks(-i, prominence=prom)
    peaks_btm = [{"E": v[idx], "I": i[idx], "Type": "Cathodic"} for idx in peaks_btm_idx]
    peaks_top.sort(key=lambda x: x["E"])
    peaks_btm.sort(key=lambda x: x["E"])
    return peaks_top, peaks_btm

def split_cycles_by_voltage(v, i, v_init, v_max, v_min):
    peaks_high, _ = find_peaks(v, height=v_max - abs(v_max)*0.1)
    peaks_low, _ = find_peaks(-v, height=-(v_min + abs(v_min)*0.1))
    n_cycles = min(len(peaks_high), len(peaks_low))
    if n_cycles == 0: return [{"v": v, "i": i}]
    cycles = []
    cycle_start_idx = 0
    for k in range(n_cycles):
        p_h = peaks_high[k]
        p_l = peaks_low[k]
        last_extremum_idx = max(p_h, p_l)
        search_start = last_extremum_idx + 10
        if search_start >= len(v): cycle_end_idx = len(v)
        else:
            diff = np.abs(v[search_start:] - v_init)
            local_min_idx = np.argmin(diff)
            cycle_end_idx = search_start + local_min_idx + 1
        if cycle_end_idx > len(v): cycle_end_idx = len(v)
        cycles.append({"v": v[cycle_start_idx:cycle_end_idx], "i": i[cycle_start_idx:cycle_end_idx]})
        cycle_start_idx = cycle_end_idx
        if cycle_start_idx >= len(v) - 10: break
    if len(cycles) == 0: return [{"v": v, "i": i}]
    return cycles

def update_fig_layout(fig, title, x_title, y_title, show_grid, show_mirror, show_ticks, axis_width, font_size):
    fig.update_layout(
        title=title, xaxis_title=x_title, yaxis_title=y_title,
        height=600, font=dict(size=font_size), hovermode="closest", template="simple_white",
        xaxis=dict(showgrid=show_grid, gridcolor='lightgray', showline=True, linewidth=axis_width, linecolor='black', mirror=show_mirror, ticks="outside" if show_ticks else "", tickwidth=axis_width, tickcolor='black', showticklabels=show_ticks),
        yaxis=dict(showgrid=show_grid, gridcolor='lightgray', showline=True, linewidth=axis_width, linecolor='black', mirror=show_mirror, ticks="outside" if show_ticks else "", tickwidth=axis_width, tickcolor='black', showticklabels=show_ticks)
    )
    return fig

# ==========================================
# サイドバー設定
# ==========================================
st.sidebar.header("📂 データ設定")
st.sidebar.caption("※以下の設定は「個別解析・比較」タブのサンプルデータに適用されます。")
with st.sidebar.expander("列番号・フォーマット", expanded=False):
    c1, c2 = st.columns(2)
    with c1: x_col_idx = st.number_input("横軸 (E) 列", value=2, min_value=1)
    with c2: y_col_idx = st.number_input("縦軸 (I) 列", value=3, min_value=1)
    skip_rows = st.number_input("ヘッダー行数", value=1, min_value=0)
    data_sep = st.selectbox("区切り文字", ['auto', ',', '\t', ' '], index=0)
    smoothing = st.checkbox("スムージング", value=True)

st.sidebar.markdown("---")
st.sidebar.subheader("📥 ファイルアップロード")
sample_files = st.sidebar.file_uploader("サンプルデータ選択", type=['csv', 'txt', 'dat'], accept_multiple_files=True)

st.sidebar.markdown("---")
with st.sidebar.expander("📊 グラフ設定", expanded=False):
    line_width = st.slider("線太さ", 0.5, 5.0, 2.0, 0.1)
    color_mode = st.radio("配色", ["自動", "単色"], horizontal=True)
    custom_color = st.color_picker("色指定", "#000000")
    st.markdown("**軸設定**")
    show_grid = st.checkbox("グリッド", value=True)
    show_ticks = st.checkbox("目盛ラベル", value=True)
    show_mirror = st.checkbox("枠線 (Mirror)", value=True)
    axis_width = st.slider("軸太さ", 1.0, 5.0, 2.0, 0.5)
    font_size = st.number_input("フォントサイズ", value=14, min_value=8)

# ==========================================
# タブ構成
# ==========================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "1️⃣ 校正", "2️⃣ 個別解析", "3️⃣ 比較", "📝 HOMO/LUMO", "ℹ️ メモ・原理"
])

# ==========================================
# Tab 1: 校正 (修正箇所)
# ==========================================
with tab1:
    st.header("標準物質による基準電位の決定")

    # --- 修正: フェロセン専用の設定エリアを追加 ---
    with st.expander("⚙️ フェロセン読み込み設定 (列・ヘッダー)", expanded=False):
        st.caption("サンプルデータと形式が異なる場合、ここで指定してください。")
        fc_cols = st.columns(3)
        with fc_cols[0]:
            fc_x_col = st.number_input("横軸 (E) 列", value=2, min_value=1, key="fc_x")
        with fc_cols[1]:
            fc_y_col = st.number_input("縦軸 (I) 列", value=3, min_value=1, key="fc_y")
        with fc_cols[2]:
            fc_skip_rows = st.number_input("ヘッダー行数", value=1, min_value=0, key="fc_skip")
    # ---------------------------------------------

    fc_file = st.file_uploader("標準物質 (例: Ferrocene)", type=['csv', 'txt', 'dat'], key="fc_u")
    
    if fc_file:
        # 修正: fc_skip_rows を使用して読み込み
        df_fc = load_data(fc_file, fc_skip_rows, sep=data_sep)
        
        # 修正: fc_x_col, fc_y_col を使用してデータ抽出
        if df_fc is not None and df_fc.shape[1] >= max(fc_x_col, fc_y_col):
            v_fc = df_fc.iloc[:, fc_x_col-1].values
            i_fc = df_fc.iloc[:, fc_y_col-1].values
            
            if smoothing: i_fc = smooth_data(i_fc)
            
            c_fc1, c_fc2 = st.columns(2)
            min_v, max_v = float(np.min(v_fc)), float(np.max(v_fc))
            with c_fc1: s_min = st.number_input("探索 Min (V)", value=min_v, format="%.2f", key="fc_min")
            with c_fc2: s_max = st.number_input("探索 Max (V)", value=max_v, format="%.2f", key="fc_max")
            
            mask = (v_fc >= s_min) & (v_fc <= s_max)
            v_roi, i_roi = v_fc[mask], i_fc[mask]
            
            if len(v_roi) > 0:
                idx_max, idx_min = np.argmax(i_roi), np.argmin(i_roi)
                E_pa, I_pa = v_roi[idx_max], i_roi[idx_max]
                E_pc, I_pc = v_roi[idx_min], i_roi[idx_min]
                E_half = (E_pa + E_pc)/2
                
                res1, res2, res3 = st.columns(3)
                res1.metric("酸化 Epa", f"{E_pa:.3f} V")
                res2.metric("還元 Epc", f"{E_pc:.3f} V")
                res3.metric("式量電位 E1/2", f"{E_half:.3f} V")
                
                if st.button("基準 (0 V) に設定"):
                    st.session_state['calibration_shift'] = E_half
                    st.session_state['is_calibrated'] = True
                    st.success(f"校正完了: Shift = {E_half:.4f} V")
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=v_fc, y=i_fc, mode='lines', name='Raw', line=dict(color='gray')))
                fig.add_trace(go.Scatter(x=[E_pa], y=[I_pa], mode='markers', marker=dict(color='red', size=10), name='Anodic'))
                fig.add_trace(go.Scatter(x=[E_pc], y=[I_pc], mode='markers', marker=dict(color='blue', size=10), name='Cathodic'))
                fig.add_vline(x=E_half, line_dash='dash', line_color='green')
                fig = update_fig_layout(fig, f"Standard ({fc_file.name})", "V", "A", show_grid, show_mirror, show_ticks, axis_width, font_size)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.error(f"データの列数が不足しています。指定: 横{fc_x_col}列目 / 縦{fc_y_col}列目 (データ列数: {df_fc.shape[1] if df_fc is not None else 0})")

# ==========================================
# Tab 2: 個別解析 (ピーク検出 & ペア登録)
# ==========================================
with tab2:
    st.header("サンプル解析と $E_{1/2}$ ペア算出")
    shift = st.session_state['calibration_shift']
    if st.session_state['is_calibrated']: st.success(f"補正値: {shift:.4f} V")
    else: st.warning("未校正")

    if sample_files:
        sel_file = st.selectbox("ファイル選択", sample_files, format_func=lambda x: x.name)
        if sel_file:
            df_s = load_data(sel_file, skip_rows, sep=data_sep)
            if df_s is not None and df_s.shape[1] >= max(x_col_idx, y_col_idx):
                v_full = df_s.iloc[:, x_col_idx-1].values
                i_full = df_s.iloc[:, y_col_idx-1].values
                if smoothing: i_full = smooth_data(i_full)
                v_calib = v_full - shift

                # サイクル分割
                with st.expander("🔄 サイクル分割設定", expanded=False):
                    use_cy = st.checkbox("有効にする")
                    def_init, def_max, def_min = float(v_full[0]), float(np.max(v_full)), float(np.min(v_full))
                    cy1, cy2, cy3 = st.columns(3)
                    c_init = cy1.number_input("初期電圧", value=def_init, format="%.2f")
                    c_max = cy2.number_input("最大電圧", value=def_max, format="%.2f")
                    c_min = cy3.number_input("最小電圧", value=def_min, format="%.2f")
                
                active_v, active_i = v_calib, i_full
                cy_info = "All Data"
                if use_cy:
                    cycles = split_cycles_by_voltage(v_full, i_full, c_init, c_max, c_min)
                    if cycles:
                        opts = ["All"] + [f"Cycle {k+1}" for k in range(len(cycles))]
                        sel_cy = st.selectbox("サイクル", opts)
                        if sel_cy != "All":
                            idx = int(sel_cy.split(" ")[1]) - 1
                            active_v = cycles[idx]["v"] - shift
                            active_i = cycles[idx]["i"]
                            cy_info = sel_cy

                st.divider()
                col_L, col_R = st.columns([1, 1.3])
                
                with col_L:
                    st.subheader("1. ピーク検出")
                    if len(active_v) > 0:
                        pm, pM = float(np.min(active_v)), float(np.max(active_v))
                        c_p1, c_p2 = st.columns(2)
                        p_min = c_p1.number_input("Min (V)", value=pm, step=0.1, format="%.2f")
                        p_max = c_p2.number_input("Max (V)", value=pM, step=0.1, format="%.2f")
                        prom = st.slider("感度 (Prominence)", 0.0, 0.5, 0.01, 0.005)

                        mask = (active_v >= p_min) & (active_v <= p_max)
                        v_r, i_r = active_v[mask], active_i[mask]
                        d_top, d_btm = [], []
                        if len(v_r) > 0:
                            d_top, d_btm = detect_multiple_peaks(v_r, i_r, prom)
                        
                        st.caption(f"検出: 酸化{len(d_top)} / 還元{len(d_btm)}")

                        st.subheader("2. ペア作成・登録")
                        if not d_top and not d_btm:
                            st.warning("ピークなし")
                        else:
                            c_s1, c_s2 = st.columns(2)
                            ox_map = {f"{p['E']:.3f} V": p for p in d_top}
                            red_map = {f"{p['E']:.3f} V": p for p in d_btm}
                            k_ox = c_s1.selectbox("酸化ピーク", list(ox_map.keys())) if ox_map else None
                            k_red = c_s2.selectbox("還元ピーク", list(red_map.keys())) if red_map else None

                            if k_ox and k_red:
                                s_ox, s_red = ox_map[k_ox], red_map[k_red]
                                val_half = (s_ox['E'] + s_red['E']) / 2
                                st.success(f"**$E_{{1/2}}$ = {val_half:.4f} V**")
                                if st.button("このペアを登録 💾"):
                                    st.session_state['pair_results'].append({
                                        "File": sel_file.name, "Cycle": cy_info,
                                        "E_1/2": val_half, "E_pa": s_ox['E'], "E_pc": s_red['E'],
                                        "I_pa": s_ox['I'], "I_pc": s_red['I']
                                    })
                    else:
                         st.error("データ範囲エラー")

                with col_R:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=active_v, y=active_i, mode='lines', line=dict(color='black', width=2), name="Current"))
                    if 'v_r' in locals() and len(v_r) > 0:
                        fig.add_trace(go.Scatter(x=v_r, y=i_r, mode='lines', line=dict(color='orange', width=4), opacity=0.3, showlegend=False))
                    
                    if d_top:
                        fig.add_trace(go.Scatter(x=[p['E'] for p in d_top], y=[p['I'] for p in d_top], mode='markers', marker=dict(color='red', size=7, symbol='circle-open'), name="Ox Cand."))
                    if d_btm:
                        fig.add_trace(go.Scatter(x=[p['E'] for p in d_btm], y=[p['I'] for p in d_btm], mode='markers', marker=dict(color='blue', size=7, symbol='circle-open'), name="Red Cand."))
                    
                    saved = [p for p in st.session_state['pair_results'] if p['File'] == sel_file.name]
                    for sp in saved:
                        fig.add_vline(x=sp["E_1/2"], line_dash="dot", line_color="green", opacity=0.6)
                        fig.add_trace(go.Scatter(x=[sp["E_pa"], sp["E_pc"]], y=[sp["I_pa"], sp["I_pc"]], mode='markers+lines', marker=dict(color='green', size=10, symbol='star'), line=dict(dash='dot', width=1), name=f"E1/2={sp['E_1/2']:.2f}"))

                    fig = update_fig_layout(fig, f"{sel_file.name} ({cy_info})", "V vs Fc/Fc+", "A", show_grid, show_mirror, show_ticks, axis_width, font_size)
                    st.plotly_chart(fig, use_container_width=True)

                st.divider()
                st.subheader("📋 登録リスト")
                if st.session_state['pair_results']:
                    st.dataframe(pd.DataFrame(st.session_state['pair_results']), use_container_width=True)
                    if st.button("全削除 🗑️"):
                        st.session_state['pair_results'] = []
                        st.rerun()

# ==========================================
# Tab 3: 比較 (簡易表示)
# ==========================================
with tab3:
    st.header("比較・重ね書き")
    if sample_files:
        comp_data = {}
        for f in sample_files:
            d = load_data(f, skip_rows, sep=data_sep)
            if d is not None and d.shape[1] >= max(x_col_idx, y_col_idx):
                v_r = d.iloc[:, x_col_idx-1].values - shift
                i_r = d.iloc[:, y_col_idx-1].values
                if smoothing: i_r = smooth_data(i_r)
                comp_data[f.name] = {"v": v_r, "i": i_r}
>>>>>>> f1540c708c7cb31080c3e5d8be7a2fae1bff613e
        
        skip_rows = 0
        header_found = False
        
        # 'XYDATA' 行を探す
        for i, line in enumerate(lines):
            if "XYDATA" in line:
                skip_rows = i + 1
                header_found = True
                break
        
        # 読み込み処理
        try:
            if header_found:
                df = pd.read_csv(io.StringIO(content), skiprows=skip_rows, sep='\t', header=None, engine='python')
                if df.shape[1] < 2:
                     df = pd.read_csv(io.StringIO(content), skiprows=skip_rows, sep='\s+', header=None, engine='python')
            else:
                df = pd.read_csv(io.StringIO(content), sep=None, engine='python', header=None)
        except Exception as e:
            return None, f"パースエラー: {e}"

        df = df.apply(pd.to_numeric, errors='coerce').dropna()
        
        if df.shape[1] < 3:
            return None, "列数が不足しています (最低3列必要)"

        # データ抽出
        x = df.iloc[:, 0].values
        col2 = df.iloc[:, 1].values # IR or Abs
        col3 = df.iloc[:, 2].values # VCD or LD
        
        # 4列目がある場合は取得、なければ0で埋める
        if df.shape[1] >= 4:
            col4 = df.iloc[:, 3].values
        else:
            col4 = np.zeros_like(x)
        
        # 先頭5行を取得（確認用）
        head_df = df.head(5)
        
        return {
            'filename': uploaded_file.name,
            'x': x,
            'ir': col2,  
            'vcd': col3,
            'noise': col4,
            'head': head_df # 追加: 先頭データ
        }, None

    except Exception as e:
        return None, f"読み込み例外: {e}"

# ---------------------------------------------------------
# 関数: Gnuplot用パッケージ作成
# ---------------------------------------------------------
def create_gnuplot_package(data_list, style_dict, x_lim, y1_lim, y2_lim, y3_lim, 
                           label_y1="Signal", label_y2="Absorbance", label_y3="Noise", include_noise=False):
    if not data_list: return None
    
    all_x = []
    for d in data_list:
        all_x.extend(d['x'])
    common_x = np.sort(np.unique(all_x))[::-1] # 降順
    
    df_out = pd.DataFrame({'Wavenumber': common_x})
    plot_cmds_y1 = []
    plot_cmds_y2 = []
    plot_cmds_y3 = []
    
    current_col = 2
    for i, d in enumerate(data_list):
        fname = d['filename']
        style = style_dict.get(fname, {'color': 'black', 'scale': 1.0})
        color = style['color']
        scale = style['scale']
        
        y2_interp = np.interp(common_x, d['x'][::-1], d['ir'][::-1])          
        y1_interp = np.interp(common_x, d['x'][::-1], d['vcd'][::-1]) * scale 
        y3_interp = np.interp(common_x, d['x'][::-1], d['noise'][::-1]) * scale 
        
        safe_name = f"File_{i+1}"
        df_out[f"{safe_name}_Abs"] = y2_interp
        df_out[f"{safe_name}_Sig"] = y1_interp
        df_out[f"{safe_name}_Nse"] = y3_interp
        
        title = fname.replace('_', '\\_')
        if scale != 1.0: title += f" (x{scale})"
        
        plot_cmds_y2.append(f"'data.dat' u 1:{current_col} w l lc rgb '{color}' title '{title}'")
        plot_cmds_y1.append(f"'data.dat' u 1:{current_col+1} w l lc rgb '{color}' notitle")
        if include_noise:
            plot_cmds_y3.append(f"'data.dat' u 1:{current_col+2} w l lc rgb '{color}' notitle")
        
        current_col += 3

    data_str = df_out.to_csv(sep='\t', index=False, float_format='%.6f')

    xr = f"[{x_lim[0]}:{x_lim[1]}]"
    yr_y1 = f"[{y1_lim[0]}:{y1_lim[1]}]" if y1_lim[0] is not None else "[:]"
    yr_y2 = f"[{y2_lim[0]}:{y2_lim[1]}]" if y2_lim[0] is not None else "[:]"
    yr_y3 = f"[{y3_lim[0]}:{y3_lim[1]}]" if y3_lim[0] is not None else "[:]"

    if include_noise:
        layout_cfg = "3,1 margins 0.15, 0.95, 0.1, 0.95 spacing 0.05"
        p1 = f"""
set ylabel "{label_y1}"
set yrange {yr_y1}
set bmargin 0
set format x ""
set xzeroaxis lt 1 lc rgb "black" lw 1
plot {', '.join(plot_cmds_y1)}
"""
        p2 = f"""
set ylabel "{label_y2}"
set yrange {yr_y2}
set bmargin 0
set format x ""
plot {', '.join(plot_cmds_y2)}
"""
        p3 = f"""
set ylabel "{label_y3}"
set yrange {yr_y3}
set xlabel "Wavenumber (cm^{{-1}})"
set bmargin 4
set format x "%g"
plot {', '.join(plot_cmds_y3)}
"""
        plot_body = p1 + p2 + p3
    else:
        layout_cfg = "2,1 margins 0.15, 0.95, 0.1, 0.95 spacing 0.05"
        p1 = f"""
set ylabel "{label_y1}"
set yrange {yr_y1}
set bmargin 0
set format x ""
set xzeroaxis lt 1 lc rgb "black" lw 1
plot {', '.join(plot_cmds_y1)}
"""
        p2 = f"""
set ylabel "{label_y2}"
set yrange {yr_y2}
set xlabel "Wavenumber (cm^{{-1}})"
set bmargin 4
set format x "%g"
plot {', '.join(plot_cmds_y2)}
"""
        plot_body = p1 + p2

    script = f"""
set terminal pngcairo size 800,{900 if include_noise else 800} font "Arial,12"
set output 'plot.png'
set multiplot layout {layout_cfg}
set xrange {xr}
set grid ls 1 lc rgb "gray" lw 0.5 dt 2
set lmargin 12
set tmargin 0
{plot_body}
unset multiplot
    """
    
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zf:
        zf.writestr("data.dat", data_str)
        zf.writestr("plot.plt", script)
    zip_buffer.seek(0)
    return zip_buffer

# ---------------------------------------------------------
# メインアプリ
# ---------------------------------------------------------
def main():
    st.set_page_config(page_title="VCD/LD Analyzer", layout="wide")
    st.title("VCD / LD Spectra Analyzer")

    if 'vcd_data' not in st.session_state: st.session_state['vcd_data'] = []
    if 'ld_data' not in st.session_state: st.session_state['ld_data'] = []

    # ==========================================
    # 1. サイドバー: データ読み込み
    # ==========================================
    st.sidebar.header("📂 ファイル読み込み")
    
    st.sidebar.subheader("VCD解析用 (Tab 1, 2)")
    uploaded_vcd = st.sidebar.file_uploader(
        "VCDファイルをアップロード", 
        accept_multiple_files=True,
        key="up_vcd",
        type=['txt', 'csv', 'dat'],
        help="波数, IR, VCD, (Noise) のデータ"
    )
    if uploaded_vcd:
        data_list = []
        for f in uploaded_vcd:
            data, error_msg = load_spectral_data(f)
            if data: data_list.append(data)
            else: st.sidebar.error(f"VCD Error {f.name}: {error_msg}")
        if data_list:
            st.session_state['vcd_data'] = data_list
            st.sidebar.success(f"VCD: {len(data_list)}件 読込完了")

    st.sidebar.markdown("---")

    st.sidebar.subheader("LD解析用 (Tab 3)")
    uploaded_ld = st.sidebar.file_uploader(
        "LDファイルをアップロード", 
        accept_multiple_files=True,
        key="up_ld",
        type=['txt', 'csv', 'dat'],
        help="波数, Abs, LD のデータ"
    )
    if uploaded_ld:
        data_list = []
        for f in uploaded_ld:
            data, error_msg = load_spectral_data(f)
            if data: data_list.append(data)
            else: st.sidebar.error(f"LD Error {f.name}: {error_msg}")
        if data_list:
            st.session_state['ld_data'] = data_list
            st.sidebar.success(f"LD: {len(data_list)}件 読込完了")
    
    # === 追加機能: ファイル先頭行の確認 ===
    all_loaded = st.session_state['vcd_data'] + st.session_state['ld_data']
    if all_loaded:
        st.sidebar.markdown("---")
        with st.sidebar.expander("📄 読み込みデータの確認 (先頭5行)"):
            file_opts = [d['filename'] for d in all_loaded]
            sel_check = st.selectbox("確認するファイル", file_opts)
            for d in all_loaded:
                if d['filename'] == sel_check:
                    st.caption("※パース後の数値データ")
                    st.dataframe(d['head'])
                    break

    # ==========================================
    # タブ構成
    # ==========================================
    tab1, tab2, tab3 = st.tabs(["📊 VCD: 個別解析", "📈 VCD: 比較プロット", "📏 LD解析 (Linear Dichroism)"])

    vcd_data = st.session_state['vcd_data']
    ld_data = st.session_state['ld_data']

    # ==========================================
    # Tab 1: VCD 個別解析 (Interactive)
    # ==========================================
    with tab1:
        if not vcd_data:
            st.info("サイドバーからVCDファイルを読み込んでください。")
        else:
            st.subheader("VCD: Single Spectrum Analysis")
            col_sel, col_peak = st.columns([1, 2])
            
            with col_sel:
                file_names = [d['filename'] for d in vcd_data]
                selected_idx = st.selectbox("ファイル選択", range(len(file_names)), format_func=lambda x: file_names[x], key="vcd_sel")
                selected_data = vcd_data[selected_idx]
                
                with st.expander("軸範囲設定", expanded=False):
                    man_t1 = st.checkbox("手動設定", key="t1_man")
                    c1, c2 = st.columns(2)
                    t1_x_high = c1.number_input("X Left", value=2000.0, key="t1_xh")
                    t1_x_low = c2.number_input("X Right", value=800.0, key="t1_xl")
                    t1_vcd_min, t1_vcd_max = None, None
                    t1_ir_min, t1_ir_max = None, None
                    if man_t1:
                        t1_vcd_max = c1.number_input("VCD Max", value=0.001, format="%.5f", key="t1_vmax")
                        t1_vcd_min = c2.number_input("VCD Min", value=-0.001, format="%.5f", key="t1_vmin")
                        t1_ir_max = c1.number_input("IR Max", value=1.5, key="t1_imax")
                        t1_ir_min = c2.number_input("IR Min", value=0.0, key="t1_imin")

            with col_peak:
                do_peak = st.checkbox("ピーク検出", value=True, key="vcd_peak")
                peak_th = st.slider("しきい値", 0.0, 2.0, 0.05, 0.01, key="vcd_th")

            if selected_data:
                x, ir, vcd = selected_data['x'], selected_data['ir'], selected_data['vcd']
                peaks, _ = find_peaks(ir, height=peak_th, distance=10)
                peak_x = x[peaks]
                peak_ir = ir[peaks]
                peak_vcd = vcd[peaks]

                fig_p = make_subplots(
                    rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.15, 
                    subplot_titles=(f"VCD: {selected_data['filename']}", "IR Spectrum"),
                    row_heights=[0.5, 0.5]
                )
                fig_p.add_trace(go.Scatter(x=x, y=vcd, mode='lines', name='VCD', line=dict(color='#00008B', width=1.5)), row=1, col=1)
                fig_p.add_trace(go.Scatter(x=x, y=ir, mode='lines', name='IR', line=dict(color='#8B0000', width=1.5)), row=2, col=1)
                
                if do_peak and len(peak_x) > 0:
                    fig_p.add_trace(go.Scatter(x=peak_x, y=peak_vcd, mode='markers', marker=dict(symbol='x', color='black'), showlegend=False), row=1, col=1)
                    fig_p.add_trace(go.Scatter(x=peak_x, y=peak_ir, mode='markers', marker=dict(symbol='circle', color='red'), showlegend=False), row=2, col=1)

                fig_p.update_layout(height=600, hovermode="x unified", showlegend=False)
                fig_p.update_xaxes(title_text="Wavenumber (cm⁻¹)", range=[t1_x_high, t1_x_low], row=2, col=1)
                fig_p.update_xaxes(range=[t1_x_high, t1_x_low], row=1, col=1)
                if man_t1:
                    fig_p.update_yaxes(range=[t1_vcd_min, t1_vcd_max], row=1, col=1)
                    fig_p.update_yaxes(range=[t1_ir_min, t1_ir_max], row=2, col=1)
                fig_p.add_hline(y=0, line_width=1, line_color="black", row=1, col=1)
                st.plotly_chart(fig_p, use_container_width=True)

    # ==========================================
    # Tab 2: VCD 比較プロット (Comparison)
    # ==========================================
    with tab2:
        if not vcd_data:
            st.info("サイドバーからVCDファイルを読み込んでください。")
        else:
            st.subheader("VCD: Multi-Spectra Comparison")
            render_comparison_plot(vcd_data, "vcd", "VCD Intensity", "Absorbance", allow_noise=True)

    # ==========================================
    # Tab 3: LD解析 (Linear Dichroism)
    # ==========================================
    with tab3:
        if not ld_data:
            st.info("サイドバーの「LD解析用」エリアからファイルを読み込んでください。")
        else:
            st.subheader("LD (Linear Dichroism) Analysis")
            render_comparison_plot(ld_data, "ld", "LD Signal (3rd Col)", "Absorbance (2nd Col)", allow_noise=False)


# ---------------------------------------------------------
# 共通描画ロジック (VCD/LD共用)
# ---------------------------------------------------------
def render_comparison_plot(data_source, prefix, label_y1, label_y2, allow_noise=False):
    """
    比較プロットを描画する共通関数
    フォームを使用して「再プロットボタン」による更新を実現
    """
    col_c_sel, col_c_opt = st.columns([1, 2])
    
    with col_c_sel:
        st.markdown("##### データ選択")
        all_filenames = [d['filename'] for d in data_source]
        # データ選択はフォームの外に出す（選択した瞬間に下のスタイル設定欄を更新するため）
        selected_files = st.multiselect(
            "プロット対象", all_filenames, default=all_filenames, key=f"{prefix}_multi"
        )
        target_data = [d for d in data_source if d['filename'] in selected_files]
    
    with col_c_opt:
        st.markdown("##### グラフ設定")
        # 設定とプロットをフォームで囲む
        with st.form(key=f"{prefix}_plot_form"):
            c_leg, c_noise = st.columns(2)
            show_legend = c_leg.checkbox("凡例を表示", value=True, key=f"{prefix}_leg")
            
            show_noise = False
            if allow_noise:
                show_noise = c_noise.checkbox("ノイズ (4列目) を表示", value=False, key=f"{prefix}_nse")
            
            with st.expander("軸範囲設定", expanded=False):
                c1, c2 = st.columns(2)
                x_high = c1.number_input("X Left", value=2000.0, key=f"{prefix}_xh")
                x_low = c2.number_input("X Right", value=800.0, key=f"{prefix}_xl")
                
                man_y = st.checkbox("Y軸範囲固定", key=f"{prefix}_many")
                y1_min, y1_max = None, None
                y2_min, y2_max = None, None
                y3_min, y3_max = None, None
                
                if man_y:
                    y1_max = c1.number_input(f"1段目({label_y1}) Max", value=0.0005, format="%.5f", key=f"{prefix}_y1x")
                    y1_min = c2.number_input(f"1段目({label_y1}) Min", value=-0.0005, format="%.5f", key=f"{prefix}_y1n")
                    y2_max = c1.number_input(f"2段目({label_y2}) Max", value=1.0, key=f"{prefix}_y2x")
                    y2_min = c2.number_input(f"2段目({label_y2}) Min", value=0.0, key=f"{prefix}_y2n")
                    # ノイズ用 (表示時のみ有効だが入力欄は常設しておくか、show_noise連動させるか。フォーム内なので連動が難しい場合がある)
                    y3_max = c1.number_input("3段目(Noise) Max", value=0.0005, format="%.5f", key=f"{prefix}_y3x")
                    y3_min = c2.number_input("3段目(Noise) Min", value=-0.0005, format="%.5f", key=f"{prefix}_y3n")

            st.markdown("---")
            st.markdown("##### 🎨 スタイル設定 (色・太さ・倍率)")
            
            default_colors = list(mcolors.TABLEAU_COLORS.values())
            plot_styles = {} # 辞書で保持 (Key: Filename)

            if target_data:
                with st.expander("設定パネルを開く", expanded=True):
                    cols = st.columns(3)
                    for i, d in enumerate(target_data):
                        fname = d['filename']
                        default_c = default_colors[i % len(default_colors)]
                        with cols[i % 3]:
                            st.caption(f"**{fname}**")
                            cc, cw, cs = st.columns([1, 1, 1])
                            # キーを一意にする
                            p_color = cc.color_picker("Col", value=default_c, key=f"{prefix}_c_{fname}")
                            p_width = cw.number_input("Wid", value=1.5, step=0.5, key=f"{prefix}_w_{fname}")
                            p_scale = cs.number_input("Scl", value=1.0, step=0.5, key=f"{prefix}_s_{fname}")
                            plot_styles[fname] = {'color': p_color, 'width': p_width, 'scale': p_scale}

            # === 再プロットボタン ===
            submit_btn = st.form_submit_button("グラフを更新 (再プロット)")

    # フォームの送信ボタンが押されたとき、または初回読み込み時に描画したい場合
    # StreamlitのFormはボタンを押さないと中の値が確定しないため、初回はデフォルト値で動くか、
    # ユーザーに「更新」を押させるUIになる。
    # ここでは、target_dataがあるなら描画処理へ進む（ボタン押下をトリガーにする）
    
    if submit_btn:
        if not target_data:
            st.warning("表示するデータがありません。")
            return

        # NoiseONなら3行、OFFなら2行
        if show_noise:
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(10, 10), 
                                                gridspec_kw={'height_ratios': [1, 1, 1]})
        else:
            fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 8), 
                                           gridspec_kw={'height_ratios': [1, 1]})
            ax3 = None

        plt.subplots_adjust(hspace=0.05)
        
        for d in target_data:
            fname = d['filename']
            # スタイル取得 (フォームから送信された値)
            style = plot_styles.get(fname, {'color': 'black', 'width': 1.0, 'scale': 1.0})
            color = style['color']
            width = style['width']
            factor = style['scale']
            
            x_vals = d['x']
            y1_vals = d['vcd'] * factor
            y2_vals = d['ir']
            y3_vals = d['noise'] * factor
            
            label = f"{fname}" + (f" (x{factor})" if factor != 1.0 else "")
            
            # Plot
            ax1.plot(x_vals, y1_vals, color=color, linewidth=width, label=label)
            ax2.plot(x_vals, y2_vals, color=color, linewidth=width)
            if show_noise and ax3 is not None:
                ax3.plot(x_vals, y3_vals, color=color, linewidth=width)
        
        # Style
        ax1.axhline(0, color='black', linewidth=0.8)
        ax1.set_ylabel(label_y1)
        ax1.set_xlim(x_high, x_low)
        if man_y: ax1.set_ylim(y1_min, y1_max)
        if show_legend: ax1.legend(loc='upper right', fontsize='small', framealpha=0.5)
        
        ax2.set_ylabel(label_y2)
        if man_y: ax2.set_ylim(y2_min, y2_max)
        
        if show_noise and ax3 is not None:
            ax3.axhline(0, color='black', linewidth=0.8)
            ax3.set_ylabel("Noise (4th Col)")
            ax3.set_xlabel("Wavenumber ($cm^{-1}$)")
            if man_y: ax3.set_ylim(y3_min, y3_max)
        else:
            ax2.set_xlabel("Wavenumber ($cm^{-1}$)")
        
        st.pyplot(fig)
        
        # Download
        st.markdown("---")
        c1, c2 = st.columns(2)
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
        buf.seek(0)
        c1.download_button(f"画像保存 ({prefix}_plot.png)", buf, f"{prefix}_plot.png", "image/png")
        
        zip_dat = create_gnuplot_package(
            target_data, plot_styles, (x_high, x_low), 
            (y1_min, y1_max), (y2_min, y2_max), (y3_min, y3_max),
            label_y1, label_y2, "Noise", include_noise=show_noise
        )
        if zip_dat:
            c2.download_button("Gnuplotデータ (.zip)", zip_dat, f"{prefix}_gnuplot.zip", "application/zip")
    
    elif target_data:
        st.info("👆 設定を変更し、「グラフを更新」ボタンを押してプロットしてください。")

if __name__ == "__main__":
    main()