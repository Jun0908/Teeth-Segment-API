import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from tensorflow.keras.models import load_model
import csv
from PIL import Image
from io import BytesIO
import base64
from scipy.spatial import distance as dist
from imutils import perspective
import pandas as pd

app = Flask(__name__)
CORS(app)

# モデルとデータの準備
model_path = 'model/unet_model.h5'
model = load_model(model_path)

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({'error': 'ファイルがありません'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'ファイル名がありません'}), 400
    
    try:
        # 画像を読み込み、PNG形式で保存
        image = Image.open(file.stream)
        image.save('sample.png', 'PNG')
        return jsonify({'message': '画像が正常にアップロードされました'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# 画像をBase64エンコードされた文字列に変換する関数
def img_to_base64_str(img_array):
    img = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img)
    buff = BytesIO()
    pil_img.save(buff, format="PNG")
    return base64.b64encode(buff.getvalue()).decode("utf-8")

def preprocess_image(image_path, target_size=(512, 512)):
    img = Image.open(image_path)
    img = img.resize(target_size, Image.ANTIALIAS)
    if img.mode != 'L':
        img = img.convert('L')
    img = np.asarray(img, dtype=np.float32) / 255.0
    img = np.expand_dims(img, axis=-1)
    img = np.expand_dims(img, axis=0)
    return img

# 予測結果を画像として保存する関数
def save_prediction(prediction, save_path='predict.png'):  # 保存パスのデフォルト値を指定
    prediction_image = (prediction.squeeze() * 255).astype(np.uint8)
    img = Image.fromarray(prediction_image)
    img.save(save_path)
    print(f"Prediction saved to {save_path}")

def apply_color_based_on_intensity(input_path, output_path):
    image_gray = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    image_rgb = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2RGB)
    high_intensity_threshold = 200  # Above this, apply yellow
    low_intensity_threshold = 50    # Below this, apply purple
    image_rgb[image_gray > high_intensity_threshold] = [255, 255, 0]  # Yellow
    image_rgb[image_gray < low_intensity_threshold] = [128, 0, 128]  # Purple
    cv2.imwrite(output_path, cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))

@app.route('/predict', methods=['GET'])
def predict():
    # 予測を実行し、結果をBase64文字列として返す
    sample_image_path = 'sample.png'
    preprocessed_img = preprocess_image(sample_image_path)
    predict_img = model.predict(preprocessed_img)
    save_prediction(predict_img)  # この関数は予測結果を'predict.png'に保存します
    # apply_color_based_on_intensity関数を呼び出す際にファイルパスを渡します
    apply_color_based_on_intensity('predict.png', 'modified_predict.png')  # ファイルパスを直接指定
    image = cv2.imread("./modified_predict.png")
    base64_str = img_to_base64_str(image)
    return jsonify({"image_base64": base64_str})

@app.route('/contours', methods=['GET'])
def contours():
    # 輪郭抽出し、結果をBase64文字列として返す
    image = cv2.imread('modified_predict.png')
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    hsv_image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    lower_yellow = np.array([20, 100, 100], dtype=np.uint8)
    upper_yellow = np.array([30, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv_image, lower_yellow, upper_yellow)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contour_only_image = np.zeros_like(image_rgb)
    cv2.drawContours(contour_only_image, contours, -1, (0, 255, 0), 2)
    base64_str = img_to_base64_str(cv2.cvtColor(contour_only_image, cv2.COLOR_RGB2BGR))
    return jsonify({"image_base64": base64_str})

def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

def write_to_csv(data, file_name):
    with open(file_name, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(data)

def save_segmentation_dimensions(contours, csv_file_name='segmentation_dimensions.csv'):
    for i, cnt in enumerate(contours):
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.array(box, dtype="int")

        # 矩形の寸法を計算
        width = dist.euclidean(box[0], box[1])
        height = dist.euclidean(box[0], box[3])

        # CSVに書き込むデータ
        data = [i + 1, width, height] + box.flatten().tolist()
        write_to_csv(data, csv_file_name)

def process_image(original_image, contours, csv_file_name='midpoints.csv'):
    count = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 2000:  # エリアの最小サイズを設定
            count += 1
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = np.array(box, dtype="int")
            box = perspective.order_points(box)
            color = (list(np.random.choice(range(256), size=3)))

            # 中点を計算
            (tl, tr, br, bl) = box
            (tltrX, tltrY) = midpoint(tl, tr)
            (blbrX, blbrY) = midpoint(bl, br)
            (tlblX, tlblY) = midpoint(tl, bl)
            (trbrX, trbrY) = midpoint(tr, br)

            # 中点をCSVに書き込む
            write_to_csv([count, tltrX, tltrY, blbrX, blbrY, tlblX, tlblY, trbrX, trbrY], csv_file_name)
            
            color = (list(np.random.choice(range(256), size=3)))  
            color = [int(color[0]), int(color[1]), int(color[2])]

            # 画像に囲い込みと中点を描画
            cv2.drawContours(original_image, [box.astype("int")], 0, color, 2)
            cv2.circle(original_image, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
            cv2.circle(original_image, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
            cv2.circle(original_image, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
            cv2.circle(original_image, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)

            # 寸法を計算
            dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
            dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))

            # 画像に寸法を表示
            cv2.putText(original_image, "{:.1f}px".format(dA), (int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
            cv2.putText(original_image, "{:.1f}px".format(dB), (int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

    return original_image, count

def sort_and_replace(csv_input_filename, final_output_filename):
    # CSVファイルからデータを読み込む
    with open(csv_input_filename, 'r') as file:
        reader = csv.reader(file)
        rows = list(reader)

    # tltr の X 座標と Y 座標を基に行をソートするために、各行にインデックスを追加
    for i, row in enumerate(rows):
        x_coord = float(row[1])  # tltr の X 座標
        y_coord = float(row[2])  # tltr の Y 座標
        rows[i] = (i, x_coord, y_coord, row)  # インデックス, X座標, Y座標, 元の行

    # Y 座標でグループ化（上の歯と下の歯）
    rows.sort(key=lambda x: x[2])  # Y 座標でソート
    median_y = rows[len(rows) // 2][2]  # Y 座標の中央値

    # 下の歯と上の歯に分ける
    lower_teeth = [row for row in rows if row[2] > median_y]
    upper_teeth = [row for row in rows if row[2] <= median_y]

    # 下の歯を X 座標でソート
    lower_teeth.sort(key=lambda x: x[1])
    # 上の歯を X 座標でソート
    upper_teeth.sort(key=lambda x: x[1])

    # ソートされた行を取得し、一列目の値を行番号に置換
    sorted_rows = [row[3] for row in lower_teeth + upper_teeth]
    for i, row in enumerate(sorted_rows):
        row_number = str(i + 1)  # 行番号はインデックス + 1
        row[0] = row_number

    # 置換後のデータを最終的なファイルに書き出す
    with open(final_output_filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(sorted_rows)

def load_and_prepare_data(file_path):
    """CSVファイルからデータを読み込み、Y座標の中央値を計算する"""
    df = pd.read_csv(file_path)
    y_coords_upper = df.iloc[:, 2]  # 上の歯のY座標（C列）
    y_coords_lower = df.iloc[:, 4]  # 下の歯のY座標（E列）
    y_coords_all = pd.concat([y_coords_upper, y_coords_lower])
    median_y = np.median(y_coords_all)
    return df, median_y

def process_image_and_save_points(df, median_y, image_path, output_file):
    """画像上にドットをプロットし、プロットした点をCSVに保存する。変換したRGB画像を返す"""
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plotted_points = []
    for index, row in df.iterrows():
        if row[2] <= median_y:  # 上の歯の場合
            cv2.circle(image_rgb, (int(row[1]), int(row[2])), 5, (255, 0, 0), -1)
            plotted_points.append((int(row[1]), int(row[2])))
        else:  # 下の歯の場合
            cv2.circle(image_rgb, (int(row[3]), int(row[4])), 5, (255, 0, 0), -1)
            plotted_points.append((int(row[3]), int(row[4])))
    # プロットした座標をCSVファイルに保存
    plotted_df = pd.DataFrame(plotted_points, columns=['X', 'Y'])
    plotted_df.to_csv(output_file, index=False)
    return image_rgb

def read_midpoints_from_csv(csv_file_name):
    midpoints = []
    with open(csv_file_name, mode='r') as csv_file:
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
            if row:  # 空行を無視する
                trbr = (float(row[7]), float(row[8]))  # 右辺中点
                tlbl = (float(row[5]), float(row[6]))  # 左辺中点
                midpoints.append((trbr, tlbl))
    return midpoints

def calculate_adjacent_midpoints(midpoints):
    adjacent_midpoints = []
    for i in range(len(midpoints) - 1):
        midpoint1 = midpoints[i][0]   # i番目の輪郭の右辺中点
        midpoint2 = midpoints[i+1][1]  # i+1番目の輪郭の左辺中点
        adjacent_midpoint = ((midpoint1[0] + midpoint2[0]) / 2, (midpoint1[1] + midpoint2[1]) / 2)
        adjacent_midpoints.append(adjacent_midpoint)
    return adjacent_midpoints

def write_adjacent_midpoints_to_csv(adjacent_midpoints, csv_file_name):
    with open(csv_file_name, 'a', newline='') as f:  # 'w'から'a'に変更してファイルの末尾に追記
        writer = csv.writer(f)
        for midpoint in adjacent_midpoints:
            writer.writerow(midpoint)

def draw_and_return_midpoints_image(image_path, midpoints, output_image_path):
    image = cv2.imread(image_path)
    for midpoint in midpoints:
        cv2.circle(image, (int(midpoint[0]), int(midpoint[1])), 5, (0, 255, 0), -1)
    cv2.imwrite(output_image_path, image)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image_rgb  # この関数は修正された画像のRGB配列を返す

@app.route('/processed_imaged', methods=['GET'])

def get_processed_image():
    # 画像のパス
    original_image_path = './sample.png'
    mask_image_path = 'modified_predict.png'
    
    original_image = cv2.imread(original_image_path)
    mask_image = cv2.imread(mask_image_path)

    # オリジナル画像とマスク画像のサイズを確認
    if original_image.shape[:2] != mask_image.shape[:2]:
    # サイズが異なる場合、マスク画像をリサイズ
     mask_image = cv2.resize(mask_image, (original_image.shape[1], original_image.shape[0]))

    # マスク画像をRGB形式に変換
    mask_image_rgb = cv2.cvtColor(mask_image, cv2.COLOR_BGR2RGB)

    # HSV色空間へ変換
    hsv_image = cv2.cvtColor(mask_image_rgb, cv2.COLOR_RGB2HSV)

    # 黄色の範囲を定義
    lower_yellow = np.array([20, 100, 100], dtype=np.uint8)
    upper_yellow = np.array([30, 255, 255], dtype=np.uint8)

    # 黄色のマスクを作成
    mask = cv2.inRange(hsv_image, lower_yellow, upper_yellow)

    # マスクから輪郭を検出
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    save_segmentation_dimensions(contours, 'segmentation_dimensions.csv')
    processed_image, segment_count = process_image(original_image, contours, 'midpoints.csv')

    retval, buffer = cv2.imencode('.jpg', processed_image)
    img_str = base64.b64encode(buffer).decode()

    return jsonify(image_base64=img_str)


@app.route('/download_csv/midpoints', methods=['GET'])
def get_csv_data():
    file_path = 'midpoints.csv'
    data = []
    try:
        with open(file_path, mode='r', encoding='utf-8') as csv_file:
            csv_reader = csv.DictReader(csv_file)
            for row in csv_reader:
                data.append(row)
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 404
    
@app.route('/histograms', methods=['GET'])
def generate_histograms():
    image_path = './predict.png'
    csv_path = './histograms.csv'

    # 画像を読み込む
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # 二値化
    _, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

    # 輪郭を検出
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # CSVファイルの存在チェック（存在しない場合はファイルを作成する）
    if not os.path.exists(csv_path):
        with open(csv_path, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(['ContourID', 'HistogramData'])  # ヘッダー行の追加

    # CSVファイルへのヒストグラムデータの書き込み
    with open(csv_path, 'a', newline='') as csvfile:  # 'w'ではなく'a'を使用して追記モードで開く
        csv_writer = csv.writer(csvfile)

        for i, contour in enumerate(contours):
            mask = np.zeros_like(image)
            cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
            pixels = image[mask == 255]
            hist, _ = np.histogram(pixels, bins=256, range=(0, 256))
            csv_writer.writerow(['Tooth_' + str(i)] + list(hist))

    # CSVファイルからヒストグラムのデータを読み込み
    histograms = []
    with open(csv_path, 'r') as csvfile:
        csv_reader = csv.reader(csvfile)
        next(csv_reader)  # ヘッダー行をスキップ
        for row in csv_reader:
            histograms.append([int(value) for value in row[1:]])

    return jsonify(histograms)

@app.route('/download_csv/final_plotted_points', methods=['GET'])
def process_data():
    #image_path = 'sample.png'
    arranged_csv = 'final_output.csv'
    csv_file_name = 'midpoints.csv'
    output_csv_file = 'final_plotted_points.csv'
    original_image_path = 'sample.png'
    output_image_path = 'output_image_with_midpoints.png'

    # これらの関数の実装は省略されています
    sort_and_replace(csv_file_name, arranged_csv)
    df, median_y = load_and_prepare_data(csv_file_name)
    image_rgb_1 = process_image_and_save_points(df, median_y, original_image_path, output_csv_file)
    cv2.imwrite('image_rgb_1.png', image_rgb_1)
    midpoints_data = read_midpoints_from_csv(arranged_csv)
    adjacent_midpoints = calculate_adjacent_midpoints(midpoints_data)
    write_adjacent_midpoints_to_csv(adjacent_midpoints, output_csv_file)
    image_rgb_2 = draw_and_return_midpoints_image(original_image_path, adjacent_midpoints, output_image_path)
    cv2.imwrite('image_rgb_2.png', image_rgb_2)
    data = []

    try:
        with open(output_csv_file, mode='r', encoding='utf-8') as csv_file:
            csv_reader = csv.DictReader(csv_file)
            for row in csv_reader:
                data.append(row)
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 404

if __name__ == '__main__':
    app.run(debug=True, port=5000)