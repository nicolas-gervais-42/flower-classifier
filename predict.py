from utils import load_model, show_prediction
import sys
import argparse

def predict(image_path, model_path, top_k, class_names_path):
	model = load_model(model_path)
	show_prediction(image_path, model, top_k, class_names_path)

def main():
	parser = argparse.ArgumentParser(description='This is flower classifier')
	parser.add_argument('image_path', type=str, help='path to flower image to classify')
	parser.add_argument('model_path', type=str, help='path to model to use')
	parser.add_argument('--top_k', type=int, default=5,  help='top k probabilites predicted by the model, default = 5')
	parser.add_argument('--class_names_path', type=str, default='./label_map.json', help='path to custom label mapping json')
	args = parser.parse_args()
	predict(args.image_path, args.model_path, args.top_k, args.class_names_path)

main()