import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import io
import math

# ---------------------------------------------------------
# 定数定義 (倍率と視野の関係は反比例)
# ---------------------------------------------------------
# 基準: 100倍のときの視野幅 (1.25mm = 1250um)
BASE_MAG = 100.0
BASE_FOV_UM = 1250.0

# 各倍率の視野幅を計算 (BASE_FOV * (基準倍率 / 対象倍率))
FOV_WIDTH_40X_UM = BASE_FOV_UM * (BASE_MAG / 40.0)   # 3125.0 um (広く見える)
FOV_WIDTH_100X_UM = BASE_FOV_UM * (BASE_MAG / 100.0) # 1250.0 um
FOV_WIDTH_400X_UM = BASE_FOV_UM * (BASE_MAG / 400.0) # 312.5 um (狭く見える)

# ---------------------------------------------------------
# フォント読み込みヘルパー (省略なし)
# ---------------------------------------------------------
def load_font(font_type, size):
    if font_type == 'serif':
        candidates = ["times.ttf", "Times New Roman.ttf", "DejaVuSerif.ttf", "LiberationSerif-Regular.ttf", "/System/Library/Fonts/Times.ttc"]
    else:
        candidates = ["arial.ttf", "DejaVuSans.ttf", "LiberationSans-Regular.ttf", "/System/Library/Fonts/Helvetica.ttc", "Verdana.ttf"]

    for font_path in candidates:
        try: return ImageFont.truetype(font_path, size)
        except OSError: continue
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

    # 2. スケール計算 (判定順序を確実に)
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

    draw.rectangle([bar_x_start, bar_y_start, bar_x_start + bar_length_px, bar_y_start + bar_height], fill=params['bar_color'])

    font = load_font(params['font_type'], params['font_size'])
    text = f"{int(params['scale_length_um'])} µm"
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    text_x = bar_x_start + (bar_length_px - text_w) / 2
    text_y = bar_y_start - text_h - (height * 0.01)

    if params['use_outline']:
        o_color = params['outline_color']
        for adj_x in range(-2, 3):
            for adj_y in range(-2, 3):
                 draw.text((text_x+adj_x, text_y+adj_y), text, fill=o_color, font=font)

    draw.text((text_x, text_y), text, fill=params['text_color'], font=font)
    return img

# ---------------------------------------------------------
# メインUI
# ---------------------------------------------------------
def main():
    st.set_page_config(page_title="Microscope Scale App", layout="centered")
    st.title("🔬 顕微鏡画像 編集ツール")
    
    # 計算された視野幅を確認用に出す
    st.sidebar.header("計算設定の確認")
    st.sidebar.write(f"40x視野: {FOV_WIDTH_40X_UM:.1f} µm")
    st.sidebar.write(f"100x視野: {FOV_WIDTH_100X_UM:.1f} µm")
    st.sidebar.write(f"400x視野: {FOV_WIDTH_400X_UM:.1f} µm")

    params = {}

    with st.expander("📸 1. 撮影・画像条件", expanded=True):
        params['magnification'] = st.radio(
            "倍率 (接眼10x × 対物レンズ)", 
            ('40x (対物4x)', '100x (対物10x)', '400x (対物40x)'), 
            index=1
        )

    with st.expander("🔄 2. 偏光マーク設定", expanded=True):
        params['show_polarization'] = st.checkbox("偏光マークを表示", value=True)
        c1, c2 = st.columns(2)
        with c1:
            pol_state = st.radio("状態", ("直交", "平行"))
            params['is_crossed_nicols'] = (pol_state == "直交")
        with c2:
            params['arrow_color'] = st.color_picker("矢印の色", "#FFFFFF")
        params['arrow_thickness'] = st.slider("線の太さ", 1, 50, 20)
        params['arrow_head_size'] = st.slider("矢じりサイズ", 10, 200, 60)

    with st.expander("📏 3. スケールバー設定", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            # 倍率に合わせてデフォルトの長さを変える
            if '40x' in params['magnification']: def_val = 1000
            elif '100x' in params['magnification']: def_val = 500
            else: def_val = 100
            
            params['scale_length_um'] = st.number_input("表示する長さ (µm)", 1, 5000, def_val, 50)
            params['bar_thickness'] = st.slider("バーの太さ", 1, 100, 30)
        with c2:
            params['scale_position'] = st.selectbox("位置", ["右下", "左下", "右上", "左上"])
            params['bar_color'] = st.color_picker("バーの色", "#FFFFFF")

    with st.expander("🔤 4. 文字デザイン", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            params['font_size'] = st.slider("文字サイズ", 10, 300, 150)
            font_choice = st.selectbox("フォント", ["Sans-serif", "Serif"])
            params['font_type'] = 'sans' if "Sans" in font_choice else 'serif'
        with c2:
            params['text_color'] = st.color_picker("文字色", "#FFFFFF")
            params['use_outline'] = st.checkbox("縁取りあり", value=True)
            params['outline_color'] = st.color_picker("縁取り色", "#000000")

    files = st.file_uploader("画像をアップロード", type=['jpg','png','tif'], accept_multiple_files=True)

    if files:
        for i, f in enumerate(files):
            img = Image.open(f)
            processed = process_image(img, params)
            
            # 選択された倍率の視野幅を取得
            fov = FOV_WIDTH_40X_UM if '40x' in params['magnification'] else (FOV_WIDTH_100X_UM if '100x' in params['magnification'] else FOV_WIDTH_400X_UM)
            
            st.image(processed, caption=f"{f.name} (想定視野: {fov:.1f}µm)", use_container_width=True)
            
            buf = io.BytesIO()
            processed.save(buf, format="PNG")
            st.download_button(f"{f.name}を保存", buf.getvalue(), f"scale_{f.name}", "image/png", key=f"dl_{i}")

if __name__ == "__main__":
    main()