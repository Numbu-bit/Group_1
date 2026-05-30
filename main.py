import cleandata as cs
from createmodel import create_model

def main():
    data = cs.clean_data()
    create_model(data)
    
    print(data.head())
    print(data.tail())
    # print(data.describe())


if __name__ == "__main__":
    main()