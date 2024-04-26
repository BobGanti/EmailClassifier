import pandas as pd
from modelling import modelling
import preprocess
import embeddings


def main():
    dataset = preprocess.load_data()  # Loads and concartenates two datasets 
    data = preprocess.preprocess_data(dataset)  # Cleans and save data locally
    data = pd.read_csv('data_cleaned/cleaned.csv') # Reads data from location

    data = preprocess.prepare_for_training(data)

    # Encoding group and target variables
    label_encoders = preprocess.get_encodings(data)

    X_train, X_test, y_train, y_test, y1_test = modelling.split_features_targets(data)

    # Applying TF-IDF vectorization
    X_train_tfidf, X_test_tfidf = embeddings.get_embeddings(X_train, X_test)

    base_model, chain_model = modelling.train(X_train_tfidf, y_train)

    ypred = modelling.make_predictions(chain_model, X_test_tfidf)

    # Display Reports
    classifier_names = f"{type(base_model).__name__} with {type(chain_model).__name__}"
    modelling.Display_classification_reports(classifier_names, y_test, ypred, y1_test, label_encoders)

if __name__ == "__main__":
    main()
