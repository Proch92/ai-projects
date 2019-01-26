from fnn import Fnn
from dataset import Dataset

def main():
	dataset = Dataset('data/mintemp.csv')
	
	model = Fnn(window_size=10)
	model.train(dataset, epochs=400)

if __name__ == '__main__':
	main()