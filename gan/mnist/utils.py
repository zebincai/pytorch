import os
import numpy as np
import matplotlib.pyplot as pyplot
import matplotlib.gridspec as gridspec


def show_img(tensor, count, out_path):
    images = tensor.detach().numpy()[0:16, :]

    img_bs, img_len = images.shape
    images = 255 * (0.5 * images + 0.5)
    images = images.astype(np.uint8)
    grid_length = int(np.ceil(np.sqrt(img_bs)))
    pyplot.figure(figsize=(4, 4))
    width = int(np.sqrt(img_len))
    gs = gridspec.GridSpec(grid_length,grid_length,wspace=0,hspace=0)

    print("Starting ... ...")
    for i, img in enumerate(images):
        ax = pyplot.subplot(gs[i])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect("equal")
        pyplot.imshow(img.reshape([width, width]), cmap=pyplot.cm.gray)
    pyplot.axis("off")
    pyplot.tight_layout()
    print("showing ..... ")
    pyplot.tight_layout()
    save_image_name = '{}.png'.format(count)
    save_file = os.path.join(out_path, save_image_name)
    pyplot.savefig(save_file, bbox_inches='tight')



























