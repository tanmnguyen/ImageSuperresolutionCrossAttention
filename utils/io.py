import os 
import matplotlib.pyplot as plt


def logging(content, log_file):
    with open(log_file, 'a') as f:
        f.write(content)
        f.write('\n')

        print(content)

def plot_learning_curve(train_history, valid_history, result_dir):
    # generator loss 
    train_gen_loss = [x['gen_loss'] for x in train_history]
    valid_gen_loss = [x['gen_loss'] for x in valid_history]
     
    # discriminator loss
    train_disc_loss = [x['disc_loss'] for x in train_history]
    valid_disc_loss = [x['disc_loss'] for x in valid_history]

    # Plot generator loss
    fig, axes = plt.subplots(2, 1)
    axes[0].plot(train_gen_loss, label='train_gen_loss')
    axes[0].plot(valid_gen_loss, label='valid_gen_loss')
    axes[0].legend()
    axes[0].set_title('Generator Loss')

    # Plot discriminator loss
    axes[1].plot(train_disc_loss, label='train_disc_loss')
    axes[1].plot(valid_disc_loss, label='valid_disc_loss')
    axes[1].legend()
    axes[1].set_title('Discriminator Loss')

    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, 'loss.png')) 