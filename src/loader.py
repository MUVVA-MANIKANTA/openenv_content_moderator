import csv
import os

def load_sentiment_sample(filepath, n=10):
    posts = []
    # encoding='latin-1' is common for this dataset
    with open(filepath, 'r', encoding='latin-1') as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            if i >= n: break
            # row: 0:target, 1:ids, 2:date, 3:flag, 4:user, 5:text
            target = int(row[0])
            text = row[5]
            author = row[4]
            # Mapping target to ActionType:
            # Sentiment != Safety policy violation. Default to 'approve'.
            posts.append({
                "post_id": i,
                "text": text,
                "user_id": i,
                "reputation": 0.5, # baseline
                "correct_label": "approve" # Safety-first moderation: don't flag just for negative sentiment
            })
    return posts

if __name__ == "__main__":
    PATH = os.environ.get("DATASET_PATH", r"data/training.csv")
    sample = load_sentiment_sample(PATH, 5)
    for s in sample:
        print(f"ID: {s['post_id']} | Text: {s['text'][:50]}... | Target: {s['correct_label']}")
