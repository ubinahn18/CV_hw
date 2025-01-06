import utils.utils as utils
from utils.video_utils import create_video_from_intermediate_results
import cv2

import torch
from torch.optim import Adam, LBFGS
from torch.autograd import Variable
import numpy as np
import os
import argparse


def build_loss(neural_net, optimizing_img, target_representations, content_feature_maps_index, style_feature_maps_indices, config):
    target_content_representation = target_representations[0]
    target_style_representation_list = target_representations[1]

    current_set_of_feature_maps = neural_net(optimizing_img)

    current_content_representation = current_set_of_feature_maps[content_feature_maps_index].squeeze(axis=0)
    content_loss = torch.nn.MSELoss(reduction='mean')(target_content_representation, current_content_representation)

    style_loss = 0.0
    current_style_representation = [utils.gram_matrix(x) for cnt, x in enumerate(current_set_of_feature_maps) if cnt in style_feature_maps_indices]

    for i in range(len(target_style_representation_list)):
      target_style_representation = target_style_representation_list[i]
      for gram_gt, gram_hat in zip(target_style_representation, current_style_representation):
          style_loss += torch.nn.MSELoss(reduction='sum')(gram_gt[0], gram_hat[0])
    style_loss /= len(target_style_representation)
    style_loss /= len(target_style_representation_list)

    tv_loss = utils.total_variation(optimizing_img)

    total_loss = config['content_weight'] * content_loss + config['style_weight'] * style_loss + config['tv_weight'] * tv_loss

    return total_loss, content_loss, style_loss, tv_loss


def make_tuning_step(neural_net, optimizer, target_representation_tensor, content_feature_maps_index, style_feature_maps_indices, config):
    # Builds function that performs a step in the tuning loop
    def tuning_step(optimizing_img):
        total_loss, content_loss, style_loss, tv_loss = build_loss(neural_net, optimizing_img, target_representations, content_feature_maps_index, style_feature_maps_indices, config)
        # Computes gradients
        total_loss.backward()
        # Updates parameters and zeroes gradients
        optimizer.step()
        optimizer.zero_grad()
        return total_loss, content_loss, style_loss, tv_loss

    # Returns the function that will be called inside the tuning loop
    return tuning_step

def augment_brightness_camera_images(image):
    image1 = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    random_bright = .25+np.random.uniform()
    #print(random_bright)
    image1[:,:,2] = image1[:,:,2]*random_bright
    image1 = cv2.cvtColor(image1,cv2.COLOR_HSV2RGB)
    return image1

def transform_image(img,ang_range,shear_range,trans_range,brightness=0):
    '''
    This function transforms images to generate new images.
    The function takes in following arguments,
    1- Image
    2- ang_range: Range of angles for rotation
    3- shear_range: Range of values to apply affine transform to
    4- trans_range: Range of values to apply translations over.

    A Random uniform distribution is used to generate different parameters for transformation

    '''
    # Rotation

    ang_rot = np.random.uniform(ang_range)-ang_range/2
    rows,cols,ch = img.shape    
    Rot_M = cv2.getRotationMatrix2D((cols/2,rows/2),ang_rot,1)

    # Translation
    tr_x = trans_range*np.random.uniform()-trans_range/2
    tr_y = trans_range*np.random.uniform()-trans_range/2
    Trans_M = np.float32([[1,0,tr_x],[0,1,tr_y]])

    # Shear
    pts1 = np.float32([[5,5],[20,5],[5,20]])

    pt1 = 5+shear_range*np.random.uniform()-shear_range/2
    pt2 = 20+shear_range*np.random.uniform()-shear_range/2

    # Brightness
    pts2 = np.float32([[pt1,5],[pt2,pt1],[5,pt2]])

    shear_M = cv2.getAffineTransform(pts1,pts2)

    img = cv2.warpAffine(img,Rot_M,(cols,rows))
    img = cv2.warpAffine(img,Trans_M,(cols,rows))
    img = cv2.warpAffine(img,shear_M,(cols,rows))

    if brightness == 1:
      img = augment_brightness_camera_images(img)

    # Assuming img is already loaded and is a floating-point image (CV_32FC1)
    img = cv2.convertScaleAbs(img) 
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cnt = contours[0]
    x, y, w, h = cv2.boundingRect(cnt)
    
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    img = img[y:y+h,x:x+w]

    return img

# added
def augment_img(img, aug_num):
    ang_range = 360
    shear_range = 10
    trans_range = 0
    brightness = 0
    aug_img_list = []
    for i in range(aug_num):
      aug_img_list.append(transform_image(img,ang_range,shear_range,trans_range,brightness))
    return aug_img_list


def neural_style_transfer(config):

    # style image num
    aug_num = 10

    content_img_path = os.path.join(config['content_images_dir'], config['content_img_name'])
    style_img_path = os.path.join(config['style_images_dir'], config['style_img_name'])

    out_dir_name = 'combined_' + os.path.split(content_img_path)[1].split('.')[0] + '_' + os.path.split(style_img_path)[1].split('.')[0]
    dump_path = os.path.join(config['output_img_dir'], out_dir_name)
    os.makedirs(dump_path, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    content_img = utils.prepare_img(content_img_path, config['height'], device)
    # style_img = utils.prepare_img(style_img_path, config['height'], device)

    # modified
    style_img_orig = utils.load_image(style_img_path)
    aug_img_list = augment_img(style_img_orig, aug_num)

    if config['init_method'] == 'random':
        # white_noise_img = np.random.uniform(-90., 90., content_img.shape).astype(np.float32)
        gaussian_noise_img = np.random.normal(loc=0, scale=90., size=content_img.shape).astype(np.float32)
        init_img = torch.from_numpy(gaussian_noise_img).float().to(device)
    elif config['init_method'] == 'content':
        init_img = content_img
    else:
        # init image has same dimension as content image - this is a hard constraint
        # feature maps need to be of same size for content image and init image
        style_img_resized = utils.prepare_img(style_img_path, np.asarray(content_img.shape[2:]), device)
        init_img = style_img_resized

    # we are tuning optimizing_img's pixels! (that's why requires_grad=True)
    optimizing_img = Variable(init_img, requires_grad=True)

    neural_net, content_feature_maps_index_name, style_feature_maps_indices_names = utils.prepare_model(config['model'], device)
    print(f'Using {config["model"]} in the optimization procedure.')

    content_img_set_of_feature_maps = neural_net(content_img)
    # style_img_set_of_feature_maps = neural_net(style_img)
    aug_style_features = []
  # list where each element is a style feature vector for one augmented style image
    for i in range(aug_num):
        aug_img = aug_img_list[i]
        aug_img = utils.prepare_style_image(aug_img)
        aug_style_features.append(neural_net(aug_img))

    target_content_representation = content_img_set_of_feature_maps[content_feature_maps_index_name[0]].squeeze(axis=0)
  
    target_style_representation_list = []
    for i in range(augnum):
        style_img_set_of_feature_maps = aug_style_features[i]
        target_style_representation = [utils.gram_matrix(x) for cnt, x in enumerate(style_img_set_of_feature_maps) if cnt in style_feature_maps_indices_names[0]]
        target_style_representation_list.append(target_style_representation)
    # list holding gram matrix for each augmented style image
    target_representations = [target_content_representation, target_style_representation_list]

    # magic numbers in general are a big no no - some things in this code are left like this by design to avoid clutter
    num_of_iterations = {
        "lbfgs": 1000,
        "adam": 3000,
    }

    #
    # Start of optimization procedure
    #
    if config['optimizer'] == 'adam':
        optimizer = Adam((optimizing_img,), lr=1e1)
        tuning_step = make_tuning_step(neural_net, optimizer, target_representations, content_feature_maps_index_name[0], style_feature_maps_indices_names[0], config)
        for cnt in range(num_of_iterations[config['optimizer']]):
            total_loss, content_loss, style_loss, tv_loss = tuning_step(optimizing_img)
            with torch.no_grad():
                print(f'Adam | iteration: {cnt:03}, total loss={total_loss.item():12.4f}, content_loss={config["content_weight"] * content_loss.item():12.4f}, style loss={config["style_weight"] * style_loss.item():12.4f}, tv loss={config["tv_weight"] * tv_loss.item():12.4f}')
                utils.save_and_maybe_display(optimizing_img, dump_path, config, cnt, num_of_iterations[config['optimizer']], should_display=False)
    elif config['optimizer'] == 'lbfgs':
        # line_search_fn does not seem to have significant impact on result
        optimizer = LBFGS((optimizing_img,), max_iter=num_of_iterations['lbfgs'], line_search_fn='strong_wolfe')
        cnt = 0

        def closure():
            nonlocal cnt
            if torch.is_grad_enabled():
                optimizer.zero_grad()
            total_loss, content_loss, style_loss, tv_loss = build_loss(neural_net, optimizing_img, target_representations, content_feature_maps_index_name[0], style_feature_maps_indices_names[0], config)
            if total_loss.requires_grad:
                total_loss.backward()
            with torch.no_grad():
                print(f'L-BFGS | iteration: {cnt:03}, total loss={total_loss.item():12.4f}, content_loss={config["content_weight"] * content_loss.item():12.4f}, style loss={config["style_weight"] * style_loss.item():12.4f}, tv loss={config["tv_weight"] * tv_loss.item():12.4f}')
                utils.save_and_maybe_display(optimizing_img, dump_path, config, cnt, num_of_iterations[config['optimizer']], should_display=False)

            cnt += 1
            return total_loss

        optimizer.step(closure)

    return dump_path



if __name__ == "__main__":
    #
    # fixed args - don't change these unless you have a good reason
    #
    default_resource_dir = os.path.join(os.path.dirname(__file__), 'data')
    content_images_dir = os.path.join(default_resource_dir, 'content-images')
    style_images_dir = os.path.join(default_resource_dir, 'style-images')
    output_img_dir = os.path.join(default_resource_dir, 'output-images')
    img_format = (4, '.jpg')  # saves images in the format: %04d.jpg

    #
    # modifiable args - feel free to play with these (only small subset is exposed by design to avoid cluttering)
    # sorted so that the ones on the top are more likely to be changed than the ones on the bottom
    #
    parser = argparse.ArgumentParser()
    parser.add_argument("--content_img_name", type=str, help="content image name", default='figures.jpg')
    parser.add_argument("--style_img_name", type=str, help="style image name", default='vg_starry_night.jpg')
    parser.add_argument("--height", type=int, help="height of content and style images", default=400)

    parser.add_argument("--content_weight", type=float, help="weight factor for content loss", default=1e5)
    parser.add_argument("--style_weight", type=float, help="weight factor for style loss", default=3e4)
    parser.add_argument("--tv_weight", type=float, help="weight factor for total variation loss", default=1e0)

    parser.add_argument("--optimizer", type=str, choices=['lbfgs', 'adam'], default='lbfgs')
    parser.add_argument("--model", type=str, choices=['vgg16', 'vgg19'], default='vgg19')
    parser.add_argument("--init_method", type=str, choices=['random', 'content', 'style'], default='content')
    parser.add_argument("--saving_freq", type=int, help="saving frequency for intermediate images (-1 means only final)", default=-1)
    args = parser.parse_args()

    # improvement on NST
    # edge detection and keep all above threshold edges?
    # :: add LoG loss?
    # data augmentation to generate more style imgs, take avg
    # 여기부터 추가임
    

    # some values of weights that worked for figures.jpg, vg_starry_night.jpg (starting point for finding good images)
    # once you understand what each one does it gets really easy -> also see README.md

    # lbfgs, content init -> (cw, sw, tv) = (1e5, 3e4, 1e0)
    # lbfgs, style   init -> (cw, sw, tv) = (1e5, 1e1, 1e-1)
    # lbfgs, random  init -> (cw, sw, tv) = (1e5, 1e3, 1e0)

    # adam, content init -> (cw, sw, tv, lr) = (1e5, 1e5, 1e-1, 1e1)
    # adam, style   init -> (cw, sw, tv, lr) = (1e5, 1e2, 1e-1, 1e1)
    # adam, random  init -> (cw, sw, tv, lr) = (1e5, 1e2, 1e-1, 1e1)

    # just wrapping settings into a dictionary
    optimization_config = dict()
    for arg in vars(args):
        optimization_config[arg] = getattr(args, arg)
    optimization_config['content_images_dir'] = content_images_dir
    optimization_config['style_images_dir'] = style_images_dir
    optimization_config['output_img_dir'] = output_img_dir
    optimization_config['img_format'] = img_format

    # original NST (Neural Style Transfer) algorithm (Gatys et al.)
    results_path = neural_style_transfer(optimization_config)

    # uncomment this if you want to create a video from images dumped during the optimization procedure
    # create_video_from_intermediate_results(results_path, img_format)
