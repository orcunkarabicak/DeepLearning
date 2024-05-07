import os
import numpy
from matplotlib import pyplot

def plot_metrics(path, epoch_size, train_metrics, validation_metrics):
    """Plots the metrics."""
    epochs = numpy.arange(1, epoch_size + 1)
    # LOSS
    figure, axes = pyplot.subplots(nrows=1, ncols=1, figsize=(8, 8))
    
    train_losses = train_metrics['losses']
    validation_losses = validation_metrics['losses']
    
    axes.plot(epochs, train_losses, label=r'$\mathrm{Train}$')
    axes.plot(epochs, validation_losses, label=r'$\mathrm{Validation}$')
    
    axes.legend()
    axes.set_xlabel(r'$\mathrm{Epochs}$')
    axes.set_ylabel(r'$\mathrm{Losses}$')
    
    axes.set_ylim(0.25, 0.50)
    axes.set_xlim(1, epoch_size + 1)
    figure.savefig(os.path.join(path, 'LOSS.pdf'), bbox_inches='tight')
    
    # SCORE
    figure, axes = pyplot.subplots(nrows=1, ncols=1, figsize=(8, 8))
    
    train_scores = train_metrics['scores']
    validation_scores = validation_metrics['scores']
    
    axes.plot(epochs, train_scores, label=r'$\mathrm{Train}$')
    axes.plot(epochs, validation_scores, label=r'$\mathrm{Validation}$')
    
    axes.legend()
    axes.set_xlabel(r'$\mathrm{Epochs}$')
    axes.set_ylabel(r'$\mathrm{Scores}$')
    
    axes.set_ylim(0.50, 1.00)
    axes.set_xlim(1, epoch_size + 1)
    figure.savefig(os.path.join(path, 'SCORE.pdf'), bbox_inches='tight')
    
    # RECALL
    figure, axes = pyplot.subplots(nrows=1, ncols=1, figsize=(8, 8))
    
    train_recalls = train_metrics['recalls']
    validation_recalls = validation_metrics['recalls']
    
    axes.plot(epochs, train_recalls, label=r'$\mathrm{Train}$')
    axes.plot(epochs, validation_recalls, label=r'$\mathrm{Validation}$')
    
    axes.legend()
    axes.set_xlabel(r'$\mathrm{Epochs}$')
    axes.set_ylabel(r'$\mathrm{Recalls}$')
    
    axes.set_ylim(0.50, 1.00)
    axes.set_xlim(1, epoch_size + 1)
    figure.savefig(os.path.join(path, 'RECALL.pdf'), bbox_inches='tight')
    
    # PRECISION
    figure, axes = pyplot.subplots(nrows=1, ncols=1, figsize=(8, 8))
    
    train_precisions = train_metrics['precisions']
    validation_precisions = validation_metrics['precisions']
    
    axes.plot(epochs, train_precisions, label=r'$\mathrm{Train}$')
    axes.plot(epochs, validation_precisions, label=r'$\mathrm{Validation}$')
    
    axes.legend()
    axes.set_xlabel(r'$\mathrm{Epochs}$')
    axes.set_ylabel(r'$\mathrm{Precisions}$')
    
    axes.set_ylim(0.50, 1.00)
    axes.set_xlim(1, epoch_size + 1)
    figure.savefig(os.path.join(path, 'PRECISION.pdf'), bbox_inches='tight')
    
    # ACCURACY
    figure, axes = pyplot.subplots(nrows=1, ncols=1, figsize=(8, 8))
    
    train_accuracies = train_metrics['accuracies']
    validation_accuracies = validation_metrics['accuracies']
    
    axes.plot(epochs, train_accuracies, label=r'$\mathrm{Train}$')
    axes.plot(epochs, validation_accuracies, label=r'$\mathrm{Validation}$')
    
    axes.legend()
    axes.set_xlabel(r'$\mathrm{Epochs}$')
    axes.set_ylabel(r'$\mathrm{Accuracies}$')
    
    axes.set_ylim(0.50, 1.00)
    axes.set_xlim(1, epoch_size + 1)
    figure.savefig(os.path.join(path, 'ACCURACY.pdf'), bbox_inches='tight')