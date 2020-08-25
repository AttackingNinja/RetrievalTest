import os

f = open('database_img.txt', 'w', encoding='utf-8')
for figname in os.listdir('images'):
    figpath = 'images/' + figname
    f.write(figpath + '\n')
