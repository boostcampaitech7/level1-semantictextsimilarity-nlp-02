import os
import matplotlib.pyplot as plt

class Visualizer:
    def save_progress_plot(self, train, val, epoch, log_dir, type, final = False) :
        if type == "loss" :
            plot_dir = os.path.join(log_dir, 'loss_plots')
        elif type == "metric" :
            plot_dir = os.path.join(log_dir, 'metric_plots')
        
        os.makedirs(plot_dir, exist_ok=True)
        plt.figure()
        plt.plot(train, label = "train")
        plt.plot(val, label = "val")
        plt.xlabel('Epochs')
        plt.ylabel(type)
        plt.legend()
        
        if final :
            plt.savefig(os.path.join(plot_dir, f'final_{type}_plot.png'))
        else :
            plt.savefig(os.path.join(plot_dir, f'{type}_plot_{epoch}.png'))
        plt.close()
