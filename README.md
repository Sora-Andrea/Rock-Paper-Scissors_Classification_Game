# Rock-Paper-Scissors Gesture Recognition Game

- YOLOv11 classification model trained on:
  - A dataset of ~700 samples per class
  - PNG images at 300x200 px
  - Samples separated into class subfolders
  - Preprocessing and augmentation for better results
  - Test and validation splits of about 10% each
- Prediction logic:
  - Uses a live webcam video with a fixed ROI box; after a brief countdown, the model predicts the user's hand gesture and shows a confidence score.
  - The opponent's choice is random to determine the winner of the round.

The game keeps track of the score and displays results.

Refer to `rps/Notes.txt` for more info.
# Rock-Paper-Scissors_Classification_Game
