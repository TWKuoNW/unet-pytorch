import cv2
from PIL import Image
from unet import Unet
import numpy as np

# 初始化模型
model = Unet(
    model_path = r"logs\Newdata_before_3000\best_epoch_weights.pth",
    num_classes = 2,
    backbone = "vgg",
    input_shape = [512, 512],
    mix_type = 0,
    cuda = False
)

# 影片輸入路徑 & 輸出路徑
video_path = r"video\v2_7m_s.mp4"
output_path = r"video_out/output_with_seagrass_2.mp4"

# 開啟影片讀取
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
width, height = 640, 480  # 你要統一輸出的影片大小

# 建立影片輸出器
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out_video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

frame_index = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 調整大小（必要）
    frame = cv2.resize(frame, (width, height))

    # OpenCV BGR → PIL RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(frame_rgb)

    # 推論取得結果和 mask
    result_pil, mask = model.detect_image(image_pil, return_mask=True)

    # 計算海草比例（類別 0 為海草）
    total_pixels = mask.size
    seagrass_pixels = np.sum(mask == 0)
    ratio = seagrass_pixels / total_pixels * 100
    text = f"Seagrass: {ratio:.2f}%"

    # PIL → NumPy → BGR
    result_np = np.array(result_pil)
    result_bgr = cv2.cvtColor(result_np, cv2.COLOR_RGB2BGR)

    # 標示文字到畫面左上
    cv2.putText(result_bgr, text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0, 255, 0), 2, cv2.LINE_AA)

    # 寫入影片
    out_video.write(result_bgr)

    # 顯示（可選）
    cv2.imshow("Segmentation", result_bgr)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_index += 1

# 清除資源
cap.release()
out_video.release()
cv2.destroyAllWindows()
