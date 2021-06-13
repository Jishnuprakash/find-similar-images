# ==== Initialisation ====
import pandas as pd
import py.similarity as sm

# ==== Load image database ====
db = pd.read_pickle('data/db.pkl')
db.head()

# ==== Load test image ====
test_image = "data/test/test-01.jpg"
sm.show_images([test_image], add_titles=['test Image'])

# Extract it's features ====
feature = sm.feature_extraction(test_image)
feature.shape

# ==== Calculate cosine similarity of the test image with images in db ====
db['similarity'] = db.feature.apply(lambda x: sm.similarity(feature, x))
db.describe()

# ==== Find top matches ====
threshold = 90  # defining a threshold value
top_n = 3
output = db.loc[db.similarity >= threshold]
output = output.sort_values(by=['similarity'], ascending=False).reset_index(drop=True)  # Sort values
output = output.head(top_n)  # Select top results

# Show Results
n_cols = min(3, output.shape[0])
sm.show_images(output['file'], cols=n_cols, add_titles=output['similarity'].astype(str) + " %")
