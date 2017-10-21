from moviepy.editor import ImageSequenceClip
import argparse


def main():
    parser = argparse.ArgumentParser(description='Create driving video.')
    parser.add_argument(
        'image_folder',
        type=str,
        nargs='?',
        default='video',
        help='Path to image folder. The video will be created from these images.'
    )
    parser.add_argument(
        '--fps',
        type=int,
        nargs='?',
        default=25,
        help='FPS (Frames per second) setting for the video.')
    args = parser.parse_args()

    exp_image_folder = args.image_folder
    exp_fps = args.fps

    #exp_image_folder = './move2'
    #exp_fps = 25

    video_file = exp_image_folder + '.mp4'
    print("Creating video {}, FPS={}".format(video_file, exp_fps))
    clip = ImageSequenceClip(exp_image_folder, fps=exp_fps)
    clip.write_videofile(video_file)


if __name__ == '__main__':
    main()
