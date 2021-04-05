import os
import numpy as np
import matplotlib.pyplot as pyplot
import matplotlib.gridspec as gridspec


def show_img(tensor, count, out_path):
    images = tensor.to('cpu')
    images = images.detach().numpy()
    images = images[[6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84, 90, 96]]

    img_bs, img_len, img_height, img_width = images.shape
    images = 255 * (0.5 * images + 0.5)
    images = images.astype(np.uint8)
    grid_length = int(np.ceil(np.sqrt(img_bs)))
    pyplot.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(grid_length,grid_length,wspace=0,hspace=0)

    print("Starting ... ...")
    for i, img in enumerate(images):
        ax = pyplot.subplot(gs[i])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect("equal")
        pyplot.imshow(img.reshape([img_height, img_width]), cmap=pyplot.cm.gray)
    pyplot.axis("off")
    pyplot.tight_layout()
    print("showing ..... ")
    pyplot.tight_layout()
    save_image_name = '{}.png'.format(count)
    save_file = os.path.join(out_path, save_image_name)
    pyplot.savefig(save_file, bbox_inches='tight')



























