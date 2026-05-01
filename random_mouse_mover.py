"""
Random Mouse Mover
------------------
Di chuyển chuột đến vị trí ngẫu nhiên trên màn hình theo chu kỳ ngẫu nhiên.
Hữu ích để giữ máy tính không bị idle (sleep / khóa màn hình).

Cài đặt thư viện trước khi chạy:
    pip install pyautogui

Cách dùng:
    python random_mouse_mover.py

Dừng:
    - Nhấn Ctrl+C trong terminal
    - Hoặc kéo chuột nhanh vào 1 trong 4 góc màn hình (failsafe của pyautogui)
"""

import random
import time
import sys
import pyautogui

# ====== CẤU HÌNH ======
MIN_INTERVAL = 5      # Khoảng thời gian tối thiểu giữa 2 hành động (giây)
MAX_INTERVAL = 30     # Khoảng thời gian tối đa giữa 2 hành động (giây)
MIN_DURATION = 0.5    # Thời gian di chuyển chuột tối thiểu (giây)
MAX_DURATION = 1.5    # Thời gian di chuyển chuột tối đa (giây)
EDGE_MARGIN = 50      # Cách mép màn hình ít nhất bao nhiêu pixel (tránh kích hoạt failsafe)
# Vùng click ngẫu nhiên (theo tỉ lệ màn hình)
# Ngang: trái -> phải (0.0 = trái, 1.0 = phải)
REGION_X_MIN = 0.20
REGION_X_MAX = 0.70
# Dọc: tính từ DƯỚI lên (0.0 = đáy màn hình, 1.0 = đỉnh)
REGION_Y_MIN_FROM_BOTTOM = 0.20
REGION_Y_MAX_FROM_BOTTOM = 0.60
SCROLL_MIN = -5       # Scroll lên (âm) tối thiểu
SCROLL_MAX = 5        # Scroll xuống (dương) tối đa
CLICK_CHANCE = 0.2    # Xác suất click sau khi di chuột (20%)
SCROLL_CHANCE = 0.3   # Xác suất scroll sau khi di chuột (30%)
# ======================

# Bật failsafe: kéo chuột vào góc màn hình để dừng chương trình
pyautogui.FAILSAFE = True


def random_mouse_loop():
    screen_width, screen_height = pyautogui.size()
    print(f"Kích thước màn hình: {screen_width} x {screen_height}")
    print(f"Chu kỳ di chuột: {MIN_INTERVAL}–{MAX_INTERVAL} giây ngẫu nhiên")
    print("Đang chạy... (Ctrl+C để dừng, hoặc kéo chuột vào góc màn hình)\n")

    while True:
        # Vị trí ngẫu nhiên trong vùng cấu hình (giới hạn thêm bởi EDGE_MARGIN để an toàn failsafe)
        x_min = max(EDGE_MARGIN, int(screen_width * REGION_X_MIN))
        x_max = min(screen_width - EDGE_MARGIN, int(screen_width * REGION_X_MAX))
        # Dọc tính từ dưới lên: y_screen = screen_height - frac * screen_height
        y_min = max(EDGE_MARGIN, int(screen_height * (1 - REGION_Y_MAX_FROM_BOTTOM)))
        y_max = min(screen_height - EDGE_MARGIN, int(screen_height * (1 - REGION_Y_MIN_FROM_BOTTOM)))

        x = random.randint(x_min, x_max)
        y = random.randint(y_min, y_max)

        # Thời gian di chuột ngẫu nhiên (chuyển động mượt)
        duration = random.uniform(MIN_DURATION, MAX_DURATION)

        try:
            pyautogui.moveTo(x, y, duration=duration)
            action = f"Di chuột đến ({x}, {y}) trong {duration:.2f}s"

            # Random click
            if random.random() < CLICK_CHANCE:
                pyautogui.click()
                action += " + Click"

            # Random scroll
            if random.random() < SCROLL_CHANCE:
                scroll_amount = random.randint(SCROLL_MIN, SCROLL_MAX)
                if scroll_amount != 0:
                    pyautogui.scroll(scroll_amount)
                    direction = "lên" if scroll_amount < 0 else "xuống"
                    action += f" + Scroll {direction} {abs(scroll_amount)}"

        except pyautogui.FailSafeException:
            print("\n[!] Đã kích hoạt failsafe (chuột chạm góc màn hình). Thoát.")
            sys.exit(0)

        # Đợi 1 khoảng ngẫu nhiên trước hành động tiếp theo
        wait = random.uniform(MIN_INTERVAL, MAX_INTERVAL)
        print(f"-> {action}. Chờ {wait:.1f}s...")
        time.sleep(wait)


if __name__ == "__main__":
    try:
        random_mouse_loop()
    except KeyboardInterrupt:
        print("\nĐã dừng theo yêu cầu người dùng. Bye!")
