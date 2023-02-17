import matplotlib.pyplot as plt
import filecmp

def visdiff(file1, file2):
    with open(file1) as f1, open(file2) as f2:
        plt.subplot(1,2,1)
        plt.title(file1)
        plt.imshow(plt.imread(f1))
        plt.axis('off')
        
        plt.subplot(1,2,2)
        plt.title(file2)
        plt.imshow(plt.imread(f2))
        plt.axis('off')
        
        plt.show()
        plt.pause(0.001)

cfgFN0 = 'file1.png'
cfgFN = 'file2.png'

if filecmp.cmp(cfgFN0, cfgFN, shallow=False):
    print("Files are the same.")
else:
    visdiff(cfgFN0, cfgFN)
    plt.draw()
    plt.pause(0.001)