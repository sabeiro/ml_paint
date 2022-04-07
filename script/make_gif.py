import numpy as np

def makegif(directory):
    filenames = np.sort(os.listdir(directory))
    filenames = [ fnm for fnm in filenames if ".png" in fnm]
    
    with imageio.get_writer(directory + '/image.gif', mode='I') as writer:
        for filename in filenames:
            image = imageio.imread(directory + filename)
            writer.append_data(image)
