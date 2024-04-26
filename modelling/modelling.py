from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np
from Config import Config 
from model.model_classifier import RandomForest

seed = 0


def split_features_targets(data):
    # Split data into features and targets, including 'y1'
    X = data[Config.COMBINE_TEXT]
    y = data[Config.TYPE_COLS]
    y1 = data[Config.GROUPED]

    # Split the data into training and test sets, including 'y1'
    X_train, X_test, y_train, y_test, y1_train, y1_test = train_test_split(X, y, y1, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test, y1_test

# RandomForest Classifier within a ClassifierChain
def train(X_train_tfidf, y_train):
    base_model, chain_model = RandomForest.build_chain_model()
    chain_model.fit(X_train_tfidf, y_train)
    return base_model, chain_model

def make_predictions(chain_model, X_test):
    return chain_model.predict(X_test)


# This functtion prints the classification report for each class (y2, y3, y4),
# conditional on the correct predictions of the previous classes,
# further grouped by the 'y1' categories.
def Display_classification_reports(model_name, y_true, y_pred, y1_test, label_encoders):
    accuracies = 0
    # Loop over classes
    targets = Config.TYPE_COLS
    for class_idx, class_label in enumerate(targets):
        print(f"\n\n Classification Report for Type: {class_label}\n"
        f"Classifier: {model_name} {class_idx+1}\n"
        f"{'='*45}"
        )
        # Loop over categories within each class
        for group in np.unique(y1_test):
            group_name = label_encoders[Config.GROUPED].inverse_transform([group])[0]
            group_mask = y1_test == group
            y_test_group = y_true[group_mask]
            y_pred_group = y_pred[group_mask]

            # For y2, there's no condition
            if class_label == Config.CLASS_COL_2:
                print(f"\nGroup: {group_name}")
                print("-"*25)
                print(classification_report(
                    y_test_group[Config.CLASS_COL_2],
                    y_pred_group[:, 0],
                    target_names=label_encoders[Config.CLASS_COL_2].classes_,
                    zero_division=0
                ))
            else:
                # Needs to consider the correctness of previous predictions for y3 and y4. Therefore need to filter based on the correct predictions of previous classes
                mask = np.ones(len(y_test_group), dtype=bool)
                for i in range(class_idx):
                    mask &= (y_test_group.iloc[:, i] == y_pred_group[:, i])

                y_test_filtered = y_test_group[mask]
                y_pred_filtered = y_pred_group[mask, class_idx]

                if mask.any():
                    unique_labels = np.unique(y_test_filtered.iloc[:, class_idx])
                    unique_class_names = label_encoders[class_label].inverse_transform(unique_labels)
                    
                    print(f"\nGroup: {group_name}")
                    print('-'*25)
                    print(classification_report(
                        y_test_filtered.iloc[:, class_idx],
                        y_pred_filtered,
                        labels=unique_labels,
                        target_names=unique_class_names,
                        zero_division=0               
                  ))
   