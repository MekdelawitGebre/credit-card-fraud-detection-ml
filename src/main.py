from preprocessing.preprocessing import Preprocessor

if __name__ == "__main__":
    preprocessor = Preprocessor(raw_path="data/raw/creditcard.csv")
    preprocessor.run()
