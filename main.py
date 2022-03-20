from image_alignment import *
from hdr_debevec import *
from hdr_robertson import *
from tone_mapping import *
from fractions import Fraction
import cv2

def read_imgs_and_times(path, filename):
    '''
    Input :
        path : the folder path to the .txt file
        filename : filename of .txt which stores the information of images
            [image1 filename] [shutter time]
            [image2 filename] [shutter time]
            ....
    Output:
        imgs : p x h x w x c => p images
        lnT  : p x 1 =>array of log shutter times
    '''

    with open(os.path.join(path, filename)) as f:
        content = f.readlines()

    imgs, shuttertimes = [], []

    for line in content:
        info = line.split()
        img = cv2.imread(os.path.join(path, info[0]))
        # imgs.append(cv2.resize(img, (img.shape[1] // 5, img.shape[0] // 5)))
        imgs.append(img)
        shuttertimes.append(float(Fraction(info[1])))

    return imgs, np.array(shuttertimes)

if __name__ == '__main__':

    ## add argument
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type = str, default = './data/', help = 'Path to the directory that contains series of images.')
    parser.add_argument('--result_path', type = str, default = './result/', help = 'Path to the directory that stores all of results.')
    parser.add_argument('--series_of_images', type = str, default = 'desk', help = 'The folder of a series of images that contains images and shutter time file.')
    parser.add_argument('--shutter_time_filename', type = str, default = 'shutter_times.txt', help = 'The name of the file where shutter time information is stored.')
    parser.add_argument('--HDR_method', type = int, default = 0, help = "0: Paul Debevec's method, 1: Robertson's method")
    parser.add_argument('--points_num', type = int, default = 70, help = 'The number of points selected per image.')
    parser.add_argument('--set_lambda', type = int, default = 50, help = 'The constant that determines the amount of smoothness.')
    args = parser.parse_args()

    ## variables
    method = 'Debevec' if args.HDR_method == 0 else 'Robertson'
    path = os.path.join(args.data_path, args.series_of_images, "")
    save_path = os.path.join(args.result_path, args.series_of_images, method, "")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    filename = args.shutter_time_filename
    n = args.points_num # select n points per image
    l = args.set_lambda

    ## read images and get the shutter time of images
    print('Read images...')
    imgs, shuttertimes = read_imgs_and_times(path, filename)
    lnT = np.log(shuttertimes).astype('float32')

    ## image alignment
    print(np.array(imgs).shape)
    print('Image alignment...')
    imgs = image_alignment(imgs)
    print(np.array(imgs).shape)

    ## construct HDR radiance map by using Paul Debevec's method
    if args.HDR_method == 0:
        ## select sample points
        print('Select sample points...')
        Z_BGR = select_sample_points(imgs, n)
        print("Construct HDR radiance map by using Paul Debevec's method...")
        radiances = get_hdr_by_Paul_Debevec(imgs, Z_BGR, lnT, l, save_path)
    ## construct HDR radiance map by using Robertson's method
    else:
        print("Construct HDR radiance map by using Robertson's method...")
        radiances = get_hdr_by_Robertson(imgs, shuttertimes, 5, save_path = save_path)

    ## tone mapping
    print('Tone mapping...')
    ldrDrago = cv2.createTonemapDrago(1.0, 0.7).process(radiances) * 255 * 3
    cv2.imwrite(save_path + "tonemapping_Drago.png", ldrDrago)

    ldrGlobal = global_operator(radiances)
    cv2.imwrite(save_path + "tonemapping_Global.png", ldrGlobal)

    ldrLocal = local_operator(radiances)
    cv2.imwrite(save_path + "tonemapping_Local.png", ldrLocal)

    print('Finish !')
