# ------------------------------------------------------------------------------------------
# Imports
# ------------------------------------------------------------------------------------------

import calendar
import glob
import joblib
import pandas as pd
import rps_utils as utils
import time
from pathlib import Path
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# ------------------------------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------------------------------

IMG_DIR = Path("./data/train")

# ------------------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------------------

# Process images
for label, gesture in enumerate(["rock", "paper", "scissors"]):
    utils.process_images(Path(IMG_DIR / gesture), label)

# Read csv files
csv_files = glob.glob(str(IMG_DIR) + "/**/*.csv", recursive=True)
df = pd.concat(map(lambda file: pd.read_csv(file, header=None, sep=';'), csv_files))

# Get features and labels
labels = df[df.columns[0]].astype(int)
features = df.drop(columns=df.columns[0])

# Split samples into train and test set
train_features, test_features, train_labels, test_labels = train_test_split(features,
                                                                            labels,
                                                                            test_size=0.3)

# Train random forest classifier
clf = RandomForestClassifier(n_estimators=100, random_state=420)
clf.fit(train_features, train_labels)
predicted_labels = clf.predict(test_features)
print("Accuracy on test set:", metrics.accuracy_score(test_labels, predicted_labels))

# Save the trained model
timestamp = calendar.timegm(time.gmtime())
joblib.dump(clf, f"model_{timestamp}.pkl")
