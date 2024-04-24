import pandas as pd
from sklearn.preprocessing import LabelEncoder
import modelling.modelling
import preprocess
import embeddings
from modelling import modelling

def main():
    dataset = preprocess.load_data()
    data = preprocess.preprocess_data(dataset)
    data = pd.read_csv('data_cleaned/cleaned.csv')

    data = preprocess.prepare_for_training(data)

    # Encoding target variables, including 'y1'
    label_encoders = {col: LabelEncoder() for col in ['y1', 'y2', 'y3', 'y4']}
    for col in label_encoders:
        data[col] = label_encoders[col].fit_transform(data[col])

    X_train, X_test, y_train, y_test, y1_train, y1_test = modelling.data_labels(data)

    # Applying TF-IDF vectorization
    X_train_tfidf, X_test_tfidf = embeddings.get_tfidf_embd(X_train, X_test)

    ypred = modelling.train(X_train_tfidf, y_train, X_test_tfidf)

    # Display Reports
    modelling.print_conditional_classification_reports(y_test, ypred, y1_test, label_encoders)

if __name__ == "__main__":
    main()
