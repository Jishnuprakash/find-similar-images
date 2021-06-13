# ==== Initialisation ====
import pandas as pd
import py.similarity as sm

# ==== Load image database ====
db = pd.read_pickle('data/db.pkl')
db.head()

# ==== Load test image ====
test_image = "data/test/test-01.jpg"
feature = sm.feature_extraction(test_image)
feature.shape

# Find cosine similarity of the test image with the whole image db
db['similarity'] = db.feature.apply(lambda x: sm.similarity(feature, x))
db.describe()
db = db.sort_values(by=['similarity'], ascending=False).reset_index(drop=True)  # Sort values
top = db.head(6)  # Select top results

# Show Results
sm.show_images([test_image], add_titles=['test Image'])
sm.show_images(top['file'], cols=3, add_titles=top['similarity'].astype(str) + " %")
