import numpy as np
import pandas as pd


#### PLUTCHIK EMOTIONS

PLUTCHIK_EMOTIONS_POLAR = {
    "anticipation": (2 / 3, 135),
    "joy": (2 / 3, 90),
    "love": (1, 67.5),
    "surprise": (2 / 3, 315),
    "anger": (2 / 3, 180),
    "fear": (2 / 3, 0),
    "disgust": (2 / 3, 225),
    "pessimism": (1, 270),
    "sadness": (2 / 3, 270),
    "optimism": (1, 112.5),
    "trust": (2 / 3, 45),
    "admiration": (1 / 3, 45),
}

# cosine similarity between emotions
PLUTCHIK_EMOTION_SIMILARITY = {
    k: {
        k2: np.cos(np.deg2rad(v[1] - v2[1]))
        for k2, v2 in PLUTCHIK_EMOTIONS_POLAR.items()
    }
    for k, v in PLUTCHIK_EMOTIONS_POLAR.items()
}
PLUTCHIK_EMOTION_SIMILARITY = (
    pd.DataFrame(PLUTCHIK_EMOTION_SIMILARITY) + 1
) / 2


### Function to convert polar to Cartesian coordinates
def polar_to_cartesian(r, theta_deg, normalize=False):
    theta_rad = np.deg2rad(theta_deg)
    if normalize:
        r = 1
    x = r * np.cos(theta_rad)
    y = r * np.sin(theta_rad)
    return np.array([x, y])


PLUTCHIK_EMOTION_CARTESIAN = {
    k: polar_to_cartesian(*v) for k, v in PLUTCHIK_EMOTIONS_POLAR.items()
}

### END PLUTCHIK EMOTIONS
