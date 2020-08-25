import os
import shutil


def move_fig():
    for dir_name in os.listdir("./Flickr"):
        dir_path = os.path.join("./Flickr", dir_name)
        for fig_name in os.listdir(dir_path):
            fig_path = os.path.join(dir_path, fig_name)
            dest_path = os.path.join("./images", fig_name)
            if os.path.exists(dest_path) == False:
                shutil.move(fig_path, "./images")
                print(fig_path + ": done!")


def copy_test_img():
    f = open('test_img_bak.txt', 'r')
    for line in f.readlines():
        fig_path = line.strip()
        fig_name = fig_path.split('/')[1]
        shutil.copy(fig_path, 'test_images/' + fig_name)
        # print('debug')


if __name__ == "__main__":
    copy_test_img()
