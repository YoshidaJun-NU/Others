import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import io
import math
import os

# ---------------------------------------------------------
# 定数定義 (1.25mm = 1250um 基準)
# ---------------------------------------------------------
FOV_WIDTH_100X_UM = 1250.0
FOV_WIDTH_40X_UM = FOV_WIDTH_100X_UM * (100 / 40)   # 3125.0 um
FOV_WIDTH_400X_UM = FOV_WIDTH_100X_UM * (100 / 400) # 312.5 um  <-- 追加

# ---------------------------------------------------------
# フォント読み込みヘルパー
# ---------------------------------------------------------
def load_font(font_type, size):
    if font_type == 'serif':
        candidates = ["times.ttf", "Times New Roman.ttf", "DejaVuSerif.ttf", "LiberationSerif-Regular.ttf", "/System/Library/Fonts/Times.ttc"]
    else:
        candidates = ["arial.ttf", "DejaVuSans.ttf", "LiberationSans-Regular.ttf", "/System/Library/Fonts/Helvetica.ttc", "Verdana.ttf"]

    for font_path in candidates:
        try:
            return ImageFont.truetype(font_path, size)
        except OSError:
            continue
    return ImageFont.load_default()

# ---------------------------------------------------------
# 描画関数
# ---------------------------------------------------------
def draw_arrowhead(draw, tip, direction, color, size):
    length = math.sqrt(direction[0]**2 + direction[1]**2)
    if length == 0: return
    ux, uy = direction[0] / length, direction[1] / length
    
    base_center_x = tip[0] - ux * size
    base_center_y = tip[1] - uy * size
    
    vx, vy = -uy, ux
    width_factor = 0.5 
    
    p1 = tip
    p2 = (base_center_x + vx * size * width_factor, base_center_y + vy * size * width_factor)
    p3 = (base_center_x - vx * size * width_factor, base_center_y - vy * size * width_factor)
    
    draw.polygon([p1, p2, p3], fill=color)

def draw_polarization_icon(draw, params, width):
    margin = int(width * 0.02)
    icon_size = int(width * 0.1) 
    
    color = params['arrow_color']
    thickness = params['arrow_thickness']
    head_size = params['arrow_head_size']
    
    start_x, start_y = margin, margin
    end_x, end_y = margin + icon_size, margin + icon_size
    center_x, center_y = (start_x + end_x) / 2, (start_y + end_y) / 2

    line_offset = 3 
    
    if params['is_crossed_nicols']:
        draw.line([(center_x, end_y), (center_x, start_y + line_offset)], fill=color, width=thickness)
        draw_arrowhead(draw, (center_x, start_y), (0, -1), color, head_size)

        draw.line([(start_x, center_y), (end_x - line_offset, center_y)], fill=color, width=thickness)
        draw_arrowhead(draw, (end_x, center_y), (1, 0), color, head_size)
    else:
        y1 = start_y + icon_size * 0.3
        draw.line([(start_x, y1), (end_x - line_offset, y1)], fill=color, width=thickness)
        draw_arrowhead(draw, (end_x, y1), (1, 0), color, head_size)

        y2 = start_y + icon_size * 0.7
        draw.line([(start_x, y2), (end_x - line_offset, y2)], fill=color, width=thickness)
        draw_arrowhead(draw, (end_x, y2), (1, 0), color, head_size)

    return end_y + margin

def process_image(image, params):
    img = image.copy()
    draw = ImageDraw.Draw(img)
    width, height = img.size
    
    icon_bottom_y = 0
    if params['show_polarization']:
        icon_bottom_y = draw_polarization_icon(draw, params, width)

    # 2. スケール計算 (倍率判定を拡張)
    if '400x' in params['magnification']:
        real_width_um = FOV_WIDTH_400X_UM
    elif '100x' in params['magnification']:
        real_width_um = FOV_WIDTH_100X_UM
    else:
        real_width_um = FOV_WIDTH_40X_UM
    
    pixels_per_um = width / real_width_um
    bar_length_px = params['scale_length_um'] * pixels_per_um
    bar_height = params['bar_thickness']

    margin_x = int(width * 0.05)
    margin_y = int(height * 0.05)
    position = params['scale_position']

    if position == "右下":
        bar_x_start = width - margin_x - bar_length_px
        bar_y_start = height - margin_y - bar_height
    elif position == "左下":
        bar_x_start = margin_x
        bar_y_start = height - margin_y - bar_height
    elif position == "右上":
        bar_x_start = width - margin_x - bar_length_px
        bar_y_start = margin_y
    elif position == "左上":
        bar_x_start = margin_x
        bar_y_start = max(margin_y, icon_bottom_y + margin_y/2)

    bar_x_end = bar_x_start + bar_length_px
    bar_y_end = bar_y_start + bar_height

    draw.rectangle([bar_x_start, bar_y_start, bar_x_end, bar_y_end], fill=params['bar_color'])

    font = load_font(params['font_type'], params['font_size'])
    text = f"{int(params['scale_length_um'])} µm"
    
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    text_x = bar_x_start + (bar_length_px - text_w) / 2
    text_y = bar_y_start - text_h - (height * 0.01)

    if params['use_outline']:
        o_color = params['outline_color']
        s_width = 2
        for adj_x in range(-s_width, s_width+1):
            for adj_y in range(-s_width, s_width+1):
                 draw.text((text_x+adj_x, text_y+adj_y), text, fill=o_color, font=font)

    draw.text((text_x, text_y), text, fill=params['text_color'], font=font)

    return img

# ---------------------------------------------------------
# メインUI
# ---------------------------------------------------------
def main():
    st.set_page_config(page_title="Microscope Scale App", layout="centered")
    st.title("🔬 顕微鏡画像 編集ツール")
    # インフォメーションを更新
    st.info(f"基準: 100倍 = {int(FOV_WIDTH_100X_UM)}µm / 400倍 = {FOV_WIDTH_400X_UM}µm")

    params = {}

    with st.expander("📸 1. 撮影・画像条件", expanded=True):
        # 400x を選択肢に追加
        params['magnification'] = st.radio(
            "倍率 (接眼10x含む総合倍率)", 
            ('40x (赤色)', '100x (黄色)', '400x (青色)'), 
            index=1
        )

    with st.expander("🔄 2. 偏光マーク設定", expanded=True):
        params['show_polarization'] = st.checkbox("偏光マークを表示", value=True)
        c1, c2 = st.columns(2)
        with c1:
            pol_state = st.radio("状態", ("直交 (クロスニコル)", "平行 (明視野)"))
            params['is_crossed_nicols'] = (pol_state == "直交 (クロスニコル)")
        with c2:
            params['arrow_color'] = st.color_picker("矢印の色", "#FFFFFF")
        params['arrow_thickness'] = st.slider("矢印の線の太さ", 1, 50, 30)
        params['arrow_head_size'] = st.slider("矢じり(三角)の大きさ", 10, 200, 90)

    with st.expander("📏 3. スケールバー設定", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            # 400倍の時は視野が312umなので、初期値を100umなどにすると使いやすいです
            default_scale = 100 if '400x' in params['magnification'] else 500
            params['scale_length_um'] = st.number_input("長さ (µm)", 1, 2000, default_scale, 10)
            params['bar_thickness'] = st.slider("バーの太さ", 1, 200, 50)
        with c2:
            params['scale_position'] = st.selectbox("位置", ["右下", "左下", "右上", "左上"])
            params['bar_color'] = st.color_picker("バーの色", "#FFFFFF")

    with st.expander("🔤 4. 文字フォント・デザイン", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            params['font_size'] = st.slider("文字サイズ", 10, 300, 200)
            font_choice = st.selectbox("フォント種類", ["Sans-serif (ゴシック系)", "Serif (明朝系)"])
            params['font_type'] = 'sans' if "Sans" in font_choice else 'serif'
        with c2:
            params['text_color'] = st.color_picker("文字色", "#FFFFFF")
        params['use_outline'] = st.checkbox("文字の縁取りあり", value=True)
        params['outline_color'] = st.color_picker("縁取りの色", "#000000")

    uploaded_files = st.file_uploader("画像をアップロード (複数選択可)", type=['jpg', 'jpeg', 'png', 'tif'], accept_multiple_files=True)

    if uploaded_files:
        st.header("処理結果一覧")
        for i, uploaded_file in enumerate(uploaded_files):
            st.markdown(f"### 画像 {i+1}: {uploaded_file.name}")
            image = Image.open(uploaded_file)
            processed_image = process_image(image, params)

            # 表示用の視野幅計算
            if '400x' in params['magnification']:
                current_fov = FOV_WIDTH_400X_UM
            elif '100x' in params['magnification']:
                current_fov = FOV_WIDTH_100X_UM
            else:
                current_fov = FOV_WIDTH_40X_UM

            st.image(processed_image, caption=f"計算上の視野幅: {current_fov:.1f} µm", use_container_width=True)

            buf = io.BytesIO()
            fmt = image.format if image.format else 'PNG'
            processed_image.save(buf, format=fmt)
            st.download_button(
                f"画像 {i+1} を保存", 
                data=buf.getvalue(), 
                file_name=f"processed_{uploaded_file.name}", 
                mime=f"image/{fmt.lower()}",
                key=f"dl_{i}"
            )
            st.markdown("---")

if __name__ == "__main__":
    main()