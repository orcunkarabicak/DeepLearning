# DeepLearning
model folde explanation:  

CNN_final.ipynb: main notebook to run the model, including hyper parameter settings.

Data.ipynb: download data from kaggle to the remote server (don’t need to submit)

Dataset.py (preprocessing the data): ImageDataset Class;
			value function: calculate weights, mean and 				variance;
			dataset function: data splitting, 							transformation, normalisation.
			
FoF.py: wrong stuff (I already deleted but I dunno why it is still here).

Model.py: class CNN: cnn architecture;
		 train_and_validation function;
		 predict function: for the test results.

Plot.py: plot_metrics function for plotting loss, accuracy, and other metrics.

Pre-processing and Model Exploration.ipynb: Shows steps taken in the project until the final model.
