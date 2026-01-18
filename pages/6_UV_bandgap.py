import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import linregress

# --- ページ設定 ---
st.set_page_config(page_title="Band Gap Calculator (Tauc Plot)", layout="wide")
st.title("🌈 Absorption Spectrum & Band Gap Calculator")
st.markdown("吸収スペクトルから **Tauc Plot** を作成し、バンドギャップ ($E_g$) を算出します。")

# --- 定数 ---
HC = 1239.84193  # Planck constant * speed of light [eV nm]

# --- 関数: データ読み込み ---
def load_data(uploaded_file, skip_rows, sep):
    try:
        uploaded_file.seek(0)
        if sep == 'auto':
            try:
                df = pd.read_csv(uploaded_file, skiprows=skip_rows, header=None, engine='python')
                if df.shape[1] <= 1:
                    uploaded_file.seek(0)
                    df = pd.read_csv(uploaded_file, skiprows=skip_rows, header=None, sep=r'\s+', engine='python')
            except:
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, skiprows=skip_rows, header=None, sep=r'\s+', engine='python')
        else:
            df = pd.read_csv(uploaded_file, skiprows=skip_rows, header=None, sep=sep, engine='python')
        
        df = df.apply(pd.to_numeric, errors='coerce').dropna()
        return df
    except Exception:
        return None

# --- サイドバー設定 ---
st.sidebar.header("📂 データ読み込み設定")
with st.sidebar.expander("列・フォーマット設定", expanded=True):
    col_x = st.number_input("波長 (nm) の列番号", value=1, min_value=1)
    col_y = st.number_input("吸光度 (Abs) の列番号", value=2, min_value=1)
    skip_rows = st.number_input("ヘッダー行数", value=0, min_value=0)
    sep_opt = st.selectbox("区切り文字", ['auto', ',', '\t', ' '], index=0)

st.sidebar.markdown("---")
st.sidebar.header("⚙️ 解析設定")
transition_type = st.sidebar.selectbox(
    "遷移タイプ (n)", 
    options=["Direct Allowed (直接遷移)", "Indirect Allowed (間接遷移)"],
    index=0
)
# Tauc式の指数決定: (ahv)^(1/n) -> Direct=1/2 -> 指数=2, Indirect=2 -> 指数=0.5
tauc_power = 2.0 if "Direct" in transition_type else 0.5

uploaded_file = st.sidebar.file_uploader("吸収スペクトルデータ (.txt, .csv)", type=['txt', 'csv', 'dat'])

# --- メイン処理 ---
tab1, tab2, tab3 = st.tabs(["📊 吸収スペクトル", "📉 Tauc Plot (Eg解析)", "📝 原理・解説"])

if uploaded_file:
    df = load_data(uploaded_file, skip_rows, sep_opt)
    
    if df is not None and df.shape[1] >= max(col_x, col_y):
        # データ取得
        wavelength = df.iloc[:, col_x-1].values
        absorbance = df.iloc[:, col_y-1].values
        
        # エネルギー (eV) への変換
        # E = hc / lambda
        # ゼロ除算回避
        with np.errstate(divide='ignore'):
            energy_ev = HC / wavelength
        
        # Tauc項の計算 (ahv)^(1/n)
        # alpha (吸光係数) は 吸光度(A) に比例すると仮定
        tauc_y = (absorbance * energy_ev) ** tauc_power
        
        # データフレーム化（便利のため）
        data = pd.DataFrame({
            "Wavelength": wavelength,
            "Absorbance": absorbance,
            "Energy": energy_ev,
            "Tauc": tauc_y
        }).sort_values("Energy") # エネルギー順にソート

        # ==========================================
        # Tab 1: 生データ (吸収スペクトル)
        # ==========================================
        with tab1:
            st.subheader("吸収スペクトル (Absorbance vs Wavelength)")
            fig_spec = go.Figure()
            fig_spec.add_trace(go.Scatter(x=data["Wavelength"], y=data["Absorbance"], mode='lines', name='Spectrum'))
            fig_spec.update_layout(
                xaxis_title="Wavelength / nm", yaxis_title="Absorbance",
                height=500, template="simple_white"
            )
            st.plotly_chart(fig_spec, use_container_width=True)

        # ==========================================
        # Tab 2: Tauc Plot
        # ==========================================
        with tab2:
            st.subheader("Tauc Plot & フィッティング")
            st.markdown("グラフの**直線部分（バンド端）**が含まれるように、下のスライダーでフィッティング範囲を指定してください。")

            col_fit1, col_fit2 = st.columns([1, 2])
            
            with col_fit1:
                st.markdown("#### フィッティング範囲 (eV)")
                min_e_limit = float(data["Energy"].min())
                max_e_limit = float(data["Energy"].max())
                
                # スライダーで範囲指定
                e_range = st.slider(
                    "エネルギー範囲を選択",
                    min_value=min_e_limit,
                    max_value=max_e_limit,
                    value=(min_e_limit + (max_e_limit-min_e_limit)*0.1, max_e_limit - (max_e_limit-min_e_limit)*0.1),
                    step=0.01
                )
                
                # 範囲内のデータを抽出
                mask = (data["Energy"] >= e_range[0]) & (data["Energy"] <= e_range[1])
                x_fit = data.loc[mask, "Energy"].values
                y_fit = data.loc[mask, "Tauc"].values
                
                # 線形回帰
                if len(x_fit) > 1:
                    slope, intercept, r_value, p_value, std_err = linregress(x_fit, y_fit)
                    
                    # Eg (x切片) の計算: y = ax + b -> 0 = a*Eg + b -> Eg = -b/a
                    if slope != 0:
                        eg_calc = -intercept / slope
                    else:
                        eg_calc = 0
                    
                    st.divider()
                    st.success(f"### Calculated $E_g$: {eg_calc:.3f} eV")
                    st.caption(f"決定係数 $R^2$: {r_value**2:.4f}")
                else:
                    st.warning("範囲内のデータが少なすぎます")
                    slope, intercept, eg_calc = 0, 0, 0

            with col_fit2:
                # Tauc Plot 描画
                fig_tauc = go.Figure()
                
                # 全データ
                fig_tauc.add_trace(go.Scatter(
                    x=data["Energy"], y=data["Tauc"], 
                    mode='lines', name='Data', line=dict(color='black', width=2)
                ))
                
                # フィッティング範囲の強調
                fig_tauc.add_trace(go.Scatter(
                    x=x_fit, y=y_fit,
                    mode='lines', name='Selected Range', line=dict(color='orange', width=4), opacity=0.5
                ))

                # 近似直線 (延長してX軸との交点を見せる)
                if len(x_fit) > 1:
                    # X軸の範囲を少し広げてプロット
                    x_line_min = max(0, eg_calc - 0.5)
                    x_line_max = e_range[1] + 0.5
                    x_line = np.linspace(x_line_min, x_line_max, 100)
                    y_line = slope * x_line + intercept
                    
                    fig_tauc.add_trace(go.Scatter(
                        x=x_line, y=y_line,
                        mode='lines', name='Fit Line', line=dict(color='red', dash='dash')
                    ))
                    
                    # Egの点
                    fig_tauc.add_trace(go.Scatter(
                        x=[eg_calc], y=[0],
                        mode='markers+text', 
                        marker=dict(color='blue', size=12, symbol='x'),
                        text=[f"Eg={eg_calc:.2f}eV"], textposition="top left",
                        name='Band Gap'
                    ))

                # Y軸ラベル (遷移タイプによって変わる)
                ylabel = r"$(\alpha h \nu)^2$" if tauc_power == 2 else r"$(\alpha h \nu)^{1/2}$"

                fig_tauc.update_layout(
                    title="Tauc Plot",
                    xaxis_title="Photon Energy ($h \\nu$) / eV",
                    yaxis_title=ylabel,
                    height=600,
                    template="simple_white",
                    xaxis=dict(showgrid=True), yaxis=dict(showgrid=True)
                )
                # Y=0の線
                fig_tauc.add_hline(y=0, line_color="black", line_width=1)
                
                st.plotly_chart(fig_tauc, use_container_width=True)

    else:
        st.error(f"指定された列（{col_x}, {col_y}）がデータに存在しません。設定を確認してください。")

else:
    with tab1:
        st.info("👈 サイドバーから吸収スペクトルデータをアップロードしてください。")

# ==========================================
# Tab 3: 原理・解説
# ==========================================
with tab3:
    st.header("📝 Tauc Plotの原理")
    st.markdown(r"""
    ### 1. Taucの式
    半導体や絶縁体の光吸収端近傍において、吸光係数 $\alpha$ と光エネルギー $h\nu$ の間には以下の関係（Taucの式）が成り立ちます。

    $$ (\alpha h \nu)^{1/n} = A (h \nu - E_g) $$

    * $\alpha$: 吸光係数（薄膜や溶液では吸光度 $Abs$ で代用することが多い）
    * $h\nu$: 光子のエネルギー ($= 1240 / \lambda \ [nm]$)
    * $A$: 定数
    * $E_g$: バンドギャップエネルギー
    * $n$: 遷移の種類によって決まる定数
        * **直接遷移 (Direct Allowed):** $n = 1/2$ $\to$ 縦軸を $(\alpha h \nu)^2$ にする
        * **間接遷移 (Indirect Allowed):** $n = 2$ $\to$ 縦軸を $(\alpha h \nu)^{1/2}$ にする

    ### 2. 解析手順
    1. 横軸をエネルギー $h\nu \ [eV]$ に変換します。
    2. 縦軸を $(\alpha h \nu)^{1/n}$ に変換してプロットします。
    3. 吸収が立ち上がる**直線領域**を見つけます。
    4. その領域を直線近似し、**X軸（y=0）との交点**を読み取ると、それが $E_g$ になります。

    ### 3. 注意点
    * **ベースライン:** ベースラインが浮いている場合は、事前に補正するか、フィッティング範囲を適切に選ぶ必要があります。
    * **遷移タイプ:** 有機半導体やペロブスカイト等は一般的に「直接遷移 ($n=1/2$)」を仮定することが多いですが、物質によります。
    """)