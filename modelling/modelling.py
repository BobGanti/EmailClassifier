from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.multioutput import ClassifierChain
import numpy as np

seed = 0


def data_labels(data):
    # Split data into features and targets, including 'y1'
    X = data['combined_text']
    y = data[['y2', 'y3', 'y4']]
    y1 = data['y1']

    # Split the data into training and test sets, including 'y1'
    X_train, X_test, y_train, y_test, y1_train, y1_test = train_test_split(X, y, y1, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test, y1_train, y1_test

# RandomForest Classifier within a ClassifierChain
def train(X_train_tfidf, y_train, X_test_tfidf):
    base_rf = RandomForestClassifier(n_estimators=1000, random_state=42)
    chain_clf = ClassifierChain(base_rf, order='random', random_state=42)
    chain_clf.fit(X_train_tfidf, y_train)
    return chain_clf.predict(X_test_tfidf)


# This function prints the classification report for each class,
# conditional on the correct predictions of the previous classes,
# further grouped by the 'y1' categories.
def print_conditional_classification_reports(y_true, y_pred, y1_test, label_encoders):
    # Loop over classes
    for class_idx, class_label in enumerate(['y2', 'y3', 'y4']):
        print(f"\n\nClassification Report for {class_label}:")
        print("-" * 50)

        # Loop over categories within each class
        for category in np.unique(y1_test):
            category_name = label_encoders['y1'].inverse_transform([category])[0]
            category_mask = y1_test == category
            y_test_category = y_true[category_mask]
            y_pred_category = y_pred[category_mask]

            # For y2, there's no condition
            if class_label == 'y2':
                print(f"\nCategory: {category_name}")
                print(classification_report(
                    y_test_category['y2'],
                    y_pred_category[:, 0],
                    target_names=label_encoders['y2'].classes_,
                    zero_division=0
                ))
            else:
                # Need to consider the correctness of previous predictions for y3 and y4. Therefore we filter based on the correct predictions of previous classes
                mask = np.ones(len(y_test_category), dtype=bool)
                for i in range(class_idx):
                    mask &= (y_test_category.iloc[:, i] == y_pred_category[:, i])

                y_test_filtered = y_test_category[mask]
                y_pred_filtered = y_pred_category[mask, class_idx]

                if mask.any():
                    unique_labels = np.unique(y_test_filtered.iloc[:, class_idx])
                    unique_class_names = label_encoders[class_label].inverse_transform(unique_labels)
                    
                    print(f"\nCategory: {category_name}")
                    print(classification_report(
                        y_test_filtered.iloc[:, class_idx],
                        y_pred_filtered,
                        labels=unique_labels,
                        target_names=unique_class_names,
                        zero_division=0               
                  ))