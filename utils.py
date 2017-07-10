import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from PIL import ImageDraw, Image
import numpy as np

def plot(samples, preds, save, path, rgba=(255,0,0,255), point_thickness=0.3):
        fig = plt.figure(figsize=(2, 2))
        gs = gridspec.GridSpec(2, 2)
        gs.update(wspace=0.05, hspace=0.05)

        i = 0
        #rescaling pred
        preds = preds * 64

        for sample, pred in zip(samples, preds):
            ax = plt.subplot(gs[i])
            plt.axis('off')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_aspect('equal')

            img = Image.fromarray(sample.reshape(64,64)*255)
            draw = ImageDraw.Draw(img)

            for x, y in pred.reshape(-1,2):
                draw.ellipse([x, y, x+point_thickness, y+point_thickness], fill=0)

            sample = np.array(img.getdata(), np.uint8).reshape(64, 64)
            print(sample)
            plt.imshow(sample, cmap='Greys_r')

            i += 1

        if save == True:
            fig.savefig(path)

        return fig
