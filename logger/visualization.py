import os
import matplotlib.pyplot as plt

class Visualizer:
    def save_loss_plot(self, train_losses, val_losses, epoch, log_dir, final=False):
        plot_dir = os.path.join(log_dir, 'loss_plots')
        os.makedirs(plot_dir, exist_ok=True)
        plt.figure()
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Val Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        if final:
            plt.savefig(os.path.join(plot_dir, 'final_loss_plot.png'))
        else:
            plt.savefig(os.path.join(plot_dir, f'loss_plot_{epoch}.png'))
        plt.close()
        
    def save_metric_plot(self, train_metric, val_metric, epoch, log_dir, final=False) : 
        plot_dir = os.path.join(log_dir, 'metric_plots')
        os.makedirs(plot_dir, exist_ok=True)
        plt.figure()
        plt.plot(train_metric, label='Train Metric')
        plt.plot(val_metric, label='Val Metric')
        plt.xlabel('Epochs')
        plt.ylabel('Metrics')
        plt.legend()
        if final:
            plt.savefig(os.path.join(plot_dir, 'final_metric_plot.png'))
        else:
            plt.savefig(os.path.join(plot_dir, f'metric_plot_{epoch}.png'))