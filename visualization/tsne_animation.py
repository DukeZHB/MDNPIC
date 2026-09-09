"""
Assemble the per-timestep t-SNE images produced by trainer.generate_tsne_plots
into an animated GIF that visualizes the reverse diffusion (classification)
procedure.
"""
import argparse
import os
import re

import imageio


def create_gif_from_pngs(directory, output_filename='tsne_animation.gif'):
    """Collect all 'tsne_t_*.png' files in ``directory`` and save them as a GIF."""
    png_files = sorted(
        [f for f in os.listdir(directory) if f.endswith('.png') and 'tsne_t_' in f],
        key=lambda x: int(re.search(r'tsne_t_(\d+).png', x).group(1)),
        reverse=True)

    frames = [imageio.v2.imread(os.path.join(directory, f)) for f in png_files]

    with imageio.get_writer(os.path.join(directory, output_filename),
                            mode='I', duration=0.5, loop=0) as writer:
        for frame in frames:
            writer.append_data(frame)
    print(f"Saved animation to {os.path.join(directory, output_filename)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', type=str, required=True,
                        help='Directory containing tsne_t_*.png files')
    parser.add_argument('--output', type=str, default='tsne_animation.gif')
    args = parser.parse_args()
    create_gif_from_pngs(args.directory, args.output)
