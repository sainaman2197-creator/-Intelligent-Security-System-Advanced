"""CCTV Sample Video Generator creating realistic synthetic security footage."""

import os
import cv2
import numpy as np
import math

def create_cctv_background(width=1280, height=720):
    """Draws an indoor hallway/lobby security camera view with realistic perspective."""
    frame = np.full((height, width, 3), (35, 38, 42), dtype=np.uint8)

    # Perspective walls and floor
    # Floor: from y=240 to bottom
    floor_pts = np.array([[0, height], [0, 360], [width, 360], [width, height]], dtype=np.int32)
    cv2.fillPoly(frame, [floor_pts], (50, 54, 60))

    # Floor grid tiles for perspective
    for y in range(360, height, 45):
        cv2.line(frame, (0, y), (width, y), (60, 65, 72), 1)
    for x in range(0, width, 90):
        cv2.line(frame, (x, 360), (int(x * 1.5) - 300, height), (60, 65, 72), 1)

    # Back wall & Doors
    cv2.rectangle(frame, (0, 0), (width, 360), (25, 28, 32), -1)
    # Doorway 1
    cv2.rectangle(frame, (120, 180), (280, 360), (15, 17, 20), -1)
    cv2.rectangle(frame, (120, 180), (280, 360), (70, 75, 80), 2)
    # Doorway 2
    cv2.rectangle(frame, (1000, 180), (1160, 360), (15, 17, 20), -1)
    cv2.rectangle(frame, (1000, 180), (1160, 360), (70, 75, 80), 2)

    # Ceiling light fixtures
    for lx in [250, 640, 1030]:
        cv2.rectangle(frame, (lx - 80, 30), (lx + 80, 55), (180, 190, 200), -1)

    return frame

def draw_person(frame, x, y, scale=1.0, walk_phase=0.0, shirt_color=(40, 80, 160), pants_color=(30, 35, 45)):
    """Draws a human figure with head, torso, arms, and animated walking legs."""
    h = int(180 * scale)
    w = int(55 * scale)

    cx = int(x)
    bottom_y = int(y)
    top_y = bottom_y - h

    # Head
    head_r = int(16 * scale)
    head_cy = top_y + head_r + 4
    cv2.circle(frame, (cx, head_cy), head_r, (170, 195, 215), -1) # skin tone
    # Hair
    cv2.ellipse(frame, (cx, head_cy - 4), (head_r, int(head_r * 0.7)), 0, 180, 360, (20, 20, 25), -1)

    # Torso (Shirt)
    torso_top = head_cy + head_r
    torso_bottom = torso_top + int(65 * scale)
    cv2.rectangle(frame, (cx - int(w * 0.45), torso_top), (cx + int(w * 0.45), torso_bottom), shirt_color, -1)

    # Arms
    arm_swing = int(12 * scale * math.sin(walk_phase))
    cv2.line(frame, (cx - int(w * 0.45), torso_top + 10), (cx - int(w * 0.55), torso_bottom + arm_swing), shirt_color, int(8 * scale))
    cv2.line(frame, (cx + int(w * 0.45), torso_top + 10), (cx + int(w * 0.55), torso_bottom - arm_swing), shirt_color, int(8 * scale))

    # Legs (Pants) with walking animation
    leg_swing = int(18 * scale * math.sin(walk_phase))
    left_leg_bottom = (cx - int(w * 0.25) + leg_swing, bottom_y)
    right_leg_bottom = (cx + int(w * 0.25) - leg_swing, bottom_y)

    cv2.line(frame, (cx - int(w * 0.2), torso_bottom), left_leg_bottom, pants_color, int(10 * scale))
    cv2.line(frame, (cx + int(w * 0.2), torso_bottom), right_leg_bottom, pants_color, int(10 * scale))

    # Shoes
    cv2.circle(frame, left_leg_bottom, int(5 * scale), (15, 15, 20), -1)
    cv2.circle(frame, right_leg_bottom, int(5 * scale), (15, 15, 20), -1)

def generate_video(output_path="assets/sample_video.mp4", duration_sec=12, fps=30):
    """Generates a complete CCTV simulation video."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    width, height = 1280, 720
    total_frames = duration_sec * fps

    # OpenCV VideoWriter using mp4v or XVID
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    bg = create_cctv_background(width, height)

    # Target 1 (Intruder): Walks across from center to Restricted Zone (left side) and stays
    # Target 2 (Loiterer): Walks into VIP Lobby (right side) and loiters stationary for 6s

    for i in range(total_frames):
        frame = bg.copy()
        time_sec = i / fps

        # --- Target 1 Simulation (Intruder) ---
        # Starts at frame 0, enters restricted area (x=240, y=550) by frame 120, stays there
        if i < 120:
            p1_x = 640 - (400 * (i / 120))
            p1_y = 440 + (110 * (i / 120))
            p1_phase = i * 0.35
        else:
            # Subtle breathing motion
            p1_x = 240 + math.sin(i * 0.05) * 3
            p1_y = 550
            p1_phase = 0.0

        draw_person(frame, p1_x, p1_y, scale=1.1, walk_phase=p1_phase, shirt_color=(30, 45, 180), pants_color=(30, 30, 35))

        # --- Target 2 Simulation (Loiterer) ---
        # Appears around frame 40, walks to right lobby zone (x=980, y=540) by frame 150, stays stationary until frame 320, then leaves
        if i >= 30:
            rel_i = i - 30
            if rel_i < 100:
                p2_x = 1200 - (220 * (rel_i / 100))
                p2_y = 420 + (120 * (rel_i / 100))
                p2_phase = rel_i * 0.3
            elif rel_i < 270:
                # Stationary loitering in lobby zone
                p2_x = 980 + math.sin(rel_i * 0.04) * 2
                p2_y = 540
                p2_phase = 0.0
            else:
                # Exiting lobby
                exit_i = rel_i - 270
                p2_x = 980 + (250 * (exit_i / 60))
                p2_y = 540 + (100 * (exit_i / 60))
                p2_phase = exit_i * 0.35

            draw_person(frame, p2_x, p2_y, scale=1.05, walk_phase=p2_phase, shirt_color=(180, 40, 40), pants_color=(40, 45, 50))

        # CCTV timestamp watermark
        ts_str = f"2026-09-21  14:15:{int(time_sec):02d}  CAM-01 NORTH WING"
        cv2.putText(frame, ts_str, (30, height - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 2, cv2.LINE_AA)
        cv2.putText(frame, "REC ●", (width - 120, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 50, 255), 2, cv2.LINE_AA)

        out.write(frame)

    out.release()
    print(f"Successfully generated synthetic CCTV footage: {output_path} ({total_frames} frames, {duration_sec}s)")

if __name__ == "__main__":
    generate_video()
