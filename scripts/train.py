from src.data.load_data import load_data
from src.data.preprocess import preprocess_data
from src.models.train_model import train_model

def main():
    data = load_data('data/processed/train.csv')
    processed_data = preprocess_data(data)
    model, accuracy = train_model(processed_data, 'target')
    print(f"Model trained with accuracy: {accuracy}")

if __name__ == '__main__':
    main()
