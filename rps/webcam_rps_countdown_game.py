import random
import time
from pathlib import Path
import cv2
from ultralytics import YOLO

MODEL_PATH = "rps_yolo11_cls/yolo11x_rps/weights/best.pt"
ROI_SIZE = 280
TOTAL_ROUNDS = 3
REQUIRED_WINS = 2
ICON_SIZE = 24
ICON_SPACING = 8
ICON_VERTICAL_OFFSET = 2
HISTORY_ICON_START_X = 150
LABEL_ICON_GAP = 8
ASSETS_DIR = Path(__file__).resolve().parent / "assets"
ICON_FILENAMES = {
    "rock": "rock.png",
    "paper": "paper.png",
    "scissors": "scissors.png",
}
WIN_RULES = {
    "rock": "scissors",
    "paper": "rock",
    "scissors": "paper",
}
KNOWN_CHOICES = list(WIN_RULES.keys())


def normalize_choice(label):
    return label.strip().lower()


def determine_winner(player, opponent):
    if player == opponent:
        return "draw"
    if WIN_RULES.get(player) == opponent:
        return "player"
    if WIN_RULES.get(opponent) == player:
        return "opponent"
    return "unknown"


def fresh_match_state():
    return [], [], 0, 0, False, None


def load_choice_icons():
    icons = {}
    for choice, filename in ICON_FILENAMES.items():
        path = ASSETS_DIR / filename
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"Warning: could not load icon at {path}")
            continue
        icons[choice] = cv2.resize(img, (ICON_SIZE, ICON_SIZE), interpolation=cv2.INTER_AREA)
    return icons


def overlay_icon(frame, icon, x, y):
    h, w = icon.shape[:2]
    if y < 0 or x < 0 or y + h > frame.shape[0] or x + w > frame.shape[1]:
        return
    icon_location = frame[y:y + h, x:x + w]
    if icon.shape[2] == 4:
        alpha = icon[:, :, 3:] / 255.0
        color = icon[:, :, :3]
        icon_location[:] = (alpha * color + (1 - alpha) * icon_location).astype(icon_location.dtype)
    else:
        icon_location[:] = icon


def draw_history_row(frame, prefix, history, base_y, icons):
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.7
    thickness = 2
    label = f"{prefix}:"
    cv2.putText(frame, label, (10, base_y),
                font, scale, (255, 255, 255), thickness)
    label_width = cv2.getTextSize(label, font, scale, thickness)[0][0]
    x = max(10 + label_width + LABEL_ICON_GAP, HISTORY_ICON_START_X)
    icon_y = max(0, base_y - ICON_SIZE + ICON_VERTICAL_OFFSET)
    for idx in range(TOTAL_ROUNDS):
        if idx < len(history):
            choice = history[idx]
            icon = icons.get(choice)
            if icon is not None:
                overlay_icon(frame, icon, x, icon_y)
            else:
                pass
        x += ICON_SIZE + ICON_SPACING

def main():
    model = YOLO(MODEL_PATH)

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    last_preds = None
    player_history, opponent_history, player_wins, opponent_wins, match_over, winner_text = (
        fresh_match_state()
    )
    choice_icons = load_choice_icons()

    print("Controls:")
    print("  SPACE = 3-second countdown + snap prediction")
    print("  ESC   = quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        x1 = (w - ROI_SIZE) // 2
        y1 = (h - ROI_SIZE) // 2
        x2 = x1 + ROI_SIZE
        y2 = y1 + ROI_SIZE

        if not winner_text:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, "SPACE: countdown+snap | ESC: quit", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        if last_preds is not None:
            preds_text = ", ".join(f"{label} ({conf:.2f})" for label, conf in last_preds)
            cv2.putText(frame, f"Last: {preds_text}", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        draw_history_row(frame, "Player", player_history, h - 60, choice_icons)
        draw_history_row(frame, "Opponent", opponent_history, h - 20, choice_icons)

        if winner_text:
            color = (0, 255, 0) if winner_text == "YOU WIN" else (0, 0, 255) if winner_text == "YOU LOSE" else (0, 255, 255)
            text_size = cv2.getTextSize(winner_text, cv2.FONT_HERSHEY_SIMPLEX, 2.0, 4)[0]
            text_x = (w - text_size[0]) // 2
            text_y = (h + text_size[1]) // 2
            cv2.putText(frame, winner_text, (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 2.0, color, 4)

        round_counter = len(player_history)
        if not match_over:
            round_counter = min(len(player_history) + 1, TOTAL_ROUNDS)
        round_text = f"Round {max(1, round_counter)}/{TOTAL_ROUNDS}"
        round_font = cv2.FONT_HERSHEY_SIMPLEX
        round_scale = 0.7
        round_thickness = 2
        text_size = cv2.getTextSize(round_text, round_font, round_scale, round_thickness)[0]
        cv2.putText(frame, round_text, (w - text_size[0] - 10, h - 20),
                    round_font, round_scale, (0, 255, 255), round_thickness)

        cv2.imshow("RPS Snapshot (YOLO11-CLS)", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break

        if key == 32:  # SPACE
            if match_over:
                player_history, opponent_history, player_wins, opponent_wins, match_over, winner_text = (
                    fresh_match_state()
                )
                last_preds = None

            # Countdown overlay
            for t in [3, 2, 1]:
                ret2, f2 = cap.read()
                if not ret2:
                    break
                cv2.rectangle(f2, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(f2, f"Capturing in {t}...", (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
                cv2.imshow("RPS Snapshot (YOLO11-CLS)", f2)
                cv2.waitKey(1)
                time.sleep(1)

            # Snapshot frame
            ret3, snap = cap.read()
            if not ret3:
                continue

            roi = snap[y1:y2, x1:x2]

            roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            results = model(roi_rgb, verbose=False)[0]
            probs = results.probs

            if probs is not None:
                scores = probs.data.tolist()
                ordered = sorted(
                    ((results.names[idx], float(score)) for idx, score in enumerate(scores)),
                    key=lambda item: item[1],
                    reverse=True,
                )
                last_preds = ordered[:3]
                top_label = ordered[0][0]
            else:
                last_preds = [("unknown", 0.0)]
                top_label = "unknown"

            player_choice = normalize_choice(top_label)
            if player_choice not in KNOWN_CHOICES:
                continue

            opponent_choice = random.choice(KNOWN_CHOICES)
            player_history.append(player_choice)
            opponent_history.append(opponent_choice)

            result = determine_winner(player_choice, opponent_choice)
            if result == "player":
                player_wins += 1
            elif result == "opponent":
                opponent_wins += 1

            if player_wins >= REQUIRED_WINS:
                match_over = True
                winner_text = "YOU WIN"
            elif opponent_wins >= REQUIRED_WINS:
                match_over = True
                winner_text = "YOU LOSE"
            elif len(player_history) >= TOTAL_ROUNDS:
                match_over = True
                if player_wins > opponent_wins:
                    winner_text = "YOU WIN"
                elif opponent_wins > player_wins:
                    winner_text = "YOU LOSE"
                else:
                    winner_text = "DRAW"

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
