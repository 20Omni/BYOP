import streamlit as st
import cv2
import numpy as np

st.set_page_config(page_title="Lane Detection System", layout="wide")

st.title("🚗 Lane Detection and Driver Assistance")
st.write("You can test the system using image, video, or live camera.")

# small variable to keep previous frame info (helps stability)
prev_lane_center = None


def detect_lanes(image):
    global prev_lane_center

    h, w, _ = image.shape

    # converting to HSV for better color filtering
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # white + yellow lane masks
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 50, 255])

    lower_yellow = np.array([15, 80, 120])
    upper_yellow = np.array([35, 255, 255])

    mask = cv2.bitwise_or(
        cv2.inRange(hsv, lower_white, upper_white),
        cv2.inRange(hsv, lower_yellow, upper_yellow)
    )

    # removing lower part (car hood region)
    cut = int(h * 0.72)
    mask[cut:h, :] = 0

    # edge detection
    blur = cv2.GaussianBlur(mask, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    # region of interest (only road area)
    roi_mask = np.zeros_like(edges)

    poly = np.array([[
        (0, cut),
        (w, cut),
        (w, int(h * 0.55)),
        (0, int(h * 0.55))
    ]], np.int32)

    cv2.fillPoly(roi_mask, poly, 255)
    cropped = cv2.bitwise_and(edges, roi_mask)

    # line detection
    lines = cv2.HoughLinesP(
        cropped, 1, np.pi / 180,
        threshold=50,
        minLineLength=40,
        maxLineGap=120
    )

    left, right = [], []

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]

            if x1 == x2:
                continue

            slope = (y2 - y1) / (x2 - x1)

            if abs(slope) < 0.4:
                continue

            if y1 > cut and y2 > cut:
                continue

            if slope < 0 and x1 < w/2:
                left.append((x1, y1, x2, y2))
            elif slope > 0 and x1 > w/2:
                right.append((x1, y1, x2, y2))

    # average lane lines
    def avg_lane(lines):
        if not lines:
            return None

        xs, ys = [], []
        for x1, y1, x2, y2 in lines:
            xs += [x1, x2]
            ys += [y1, y2]

        fit = np.polyfit(ys, xs, 1)

        y1 = h
        y2 = int(h * 0.6)

        x1 = int(fit[0] * y1 + fit[1])
        x2 = int(fit[0] * y2 + fit[1])

        return (x1, y1, x2, y2)

    left_lane = avg_lane(left)
    right_lane = avg_lane(right)

    overlay = image.copy()
    center_lane = None

    # filling lane area
    if left_lane and right_lane:
        pts = np.array([
            [left_lane[0], left_lane[1]],
            [left_lane[2], left_lane[3]],
            [right_lane[2], right_lane[3]],
            [right_lane[0], right_lane[1]]
        ])
        cv2.fillPoly(overlay, [pts], (0, 255, 0))
        center_lane = (left_lane[0] + right_lane[0]) // 2

    if left_lane:
        cv2.line(image, (left_lane[0], left_lane[1]), (left_lane[2], left_lane[3]), (255, 0, 0), 5)

    if right_lane:
        cv2.line(image, (right_lane[0], right_lane[1]), (right_lane[2], right_lane[3]), (0, 0, 255), 5)

    output = cv2.addWeighted(overlay, 0.3, image, 0.7, 0)

    # smoothing (avoids flickering)
    if center_lane is not None:
        if prev_lane_center is not None:
            center_lane = int(0.7 * prev_lane_center + 0.3 * center_lane)
        prev_lane_center = center_lane

    # deciding position
    car_mid = w // 2

    if center_lane is not None:
        diff = center_lane - car_mid

        if abs(diff) < 35:
            msg = "center"
        elif diff < 0:
            msg = "left"
        else:
            msg = "right"
    else:
        if left_lane or right_lane:
            msg = "partial"
        else:
            msg = "detecting"

    return edges, output, msg


# ================= IMAGE =================
st.subheader("🖼️ Try with Image")

img = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if img:
    file_bytes = np.asarray(bytearray(img.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    edges, output, msg = detect_lanes(image)

    c1, c2, c3 = st.columns(3)
    c1.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    c2.image(edges)
    c3.image(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))

    if msg == "center":
        st.success("Looks good. The vehicle is centered.")
    elif msg == "left":
        st.warning("Slight shift towards left.")
    elif msg == "right":
        st.warning("Slight shift towards right.")
    elif msg == "partial":
        st.info("Only one lane is visible.")
    else:
        st.info("Trying to detect lanes...")


# ================= VIDEO =================
st.subheader("🎥 Try with Video")

vid = st.file_uploader("Upload a video", type=["mp4"])

if vid:
    with open("temp.mp4", "wb") as f:
        f.write(vid.read())

    cap = cv2.VideoCapture("temp.mp4")
    frame_box = st.empty()
    text_box = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        _, out, msg = detect_lanes(frame)
        frame = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)

        frame_box.image(frame, use_container_width=True)

        if msg == "center":
            text_box.success("Stable driving detected.")
        elif msg == "left":
            text_box.warning("Vehicle slightly to the left.")
        elif msg == "right":
            text_box.warning("Vehicle slightly to the right.")
        else:
            text_box.info("Processing...")

    cap.release()


# ================= CAMERA =================
st.subheader("📷 Live Camera")

start = st.button("Start Camera")
stop = st.button("Stop Camera")

if "run" not in st.session_state:
    st.session_state.run = False

if start:
    st.session_state.run = True

if stop:
    st.session_state.run = False

if st.session_state.run:
    cap = cv2.VideoCapture(0)
    frame_box = st.empty()
    text_box = st.empty()

    while st.session_state.run and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        _, out, msg = detect_lanes(frame)
        frame = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)

        frame_box.image(frame, use_container_width=True)

        if msg == "center":
            text_box.success("Good alignment.")
        elif msg == "left":
            text_box.warning("Move slightly right.")
        elif msg == "right":
            text_box.warning("Move slightly left.")
        else:
            text_box.info("Analyzing...")

    cap.release()