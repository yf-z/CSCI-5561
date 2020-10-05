import cv2
import numpy as np
import matplotlib.pyplot as plt

def get_differential_filter():
    # gaussian blur without normalization
    # sigma = 1
    # gaussian_filter = np.zeros((3,3))
    # for i in range(3):
    #     for j in range(3):
    #         gaussian_filter[i,j] = (float)(1/2*np.pi*sigma)
    #         gaussian_filter[i,j] = gaussian_filter[i,j]*np.exp(-(pow(i-1,2)+pow(j-1,2))/(2.0*sigma))

    # sum_gaussian = sum(gaussian_filter.ravel())
    
    # # gaussian_filter = np.divide(gaussian_filter, sum_gaussian) # normalize

    # filter_x = np.asarray([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    # filter_x = np.multiply(filter_x, gaussian_filter)

    # filter_y = np.asarray([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
    # filter_y = np.multiply(filter_y, gaussian_filter)

    # original filter
    # filter_x = np.asarray([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    # filter_y = np.asarray([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])

    # sobel filter
    G = np.asarray([1, 2, 1], float).reshape(1,3)
    filter_all = np.asarray([-1, 0, 1], float).reshape(1,3)

    filter_x = np.matmul(np.transpose(G), filter_all)
    filter_y = np.matmul(np.transpose(filter_all), G)

    return filter_x, filter_y


def filter_image(im, filter):
    # To do
    m,n = im.shape
    a, b = filter.shape
    im_filtered = np.zeros((m, n))

    for i in range(m):
        for j in range(n):
            start_i = i-1
            start_j = j-1
            for k in range(a):
              for l in range(b):
                if (start_i+k >= 0 and start_i+k < m and start_j+l >= 0 and start_j+l < n):
                  im_filtered[i, j] = im_filtered[i, j]+im[start_i+k, start_j+l]*filter[k, l]
    
    return im_filtered

def get_gradient(im_dx, im_dy):
    grad_mag = np.sqrt(np.add(np.square(im_dx), np.square(im_dy)))
    grad_angle = np.arctan2(im_dy, im_dx)
    grad_angle = np.where(grad_angle < 0, grad_angle+np.pi, grad_angle)
    return grad_mag, grad_angle

def get_index(angle):
    theta = np.asarray([15, 45, 75, 105, 135, 165], dtype='f')
    theta = theta/180*np.pi

    if angle < theta[0] or angle >= theta[5]:
        return 0
    elif angle < theta[1]:
        return 1
    elif angle < theta[2]:
        return 2
    elif angle < theta[3]:
        return 3
    elif angle < theta[4]:
        return 4
    elif angle < theta[5]:
        return 5


def build_histogram(grad_mag, grad_angle, cell_size):
    # To do
    m,n = grad_mag.shape
    m = m//cell_size
    n = n//cell_size

    ori_histo = np.zeros((m, n, 6))

    for u in range(grad_mag.shape[0]):
        for v in range(grad_mag.shape[1]):
            i = u//cell_size
            j = v//cell_size
            if (i < m and j < n):
                k = get_index(grad_angle[u,v])
                ori_histo[i,j,k] += grad_mag[u, v]

    return ori_histo


def get_block_descriptor(ori_histo, block_size):
    # To do
    stride = 1
    e = 0.001*0.001
    m,n = ori_histo.shape[:-1]
    m = m-(block_size-1)
    n = n-(block_size-1)
    ori_histo_normalized = np.zeros((m, n, 6*block_size*block_size), dtype='f')

    for i in range(m):
        for j in range(n):
            cur_v = ori_histo[i:i+block_size, j:j+block_size, :]
            cur_v = cur_v.ravel()
            sum_square = sum(np.square(cur_v))
            ori_histo_normalized[i,j,:] = np.divide(cur_v, np.sqrt(sum_square)+e)

    return ori_histo_normalized


def extract_hog(im):
    # convert grey-scale image to double format
    im = im.astype('float') / 255.0
    # To do
    filter_x, filter_y = get_differential_filter()
    vertical_edge = filter_image(im, filter_x)
    horizontal_edge = filter_image(im, filter_y)
    magnitude_img, angle_img = get_gradient(vertical_edge, horizontal_edge)
    ori_histo = build_histogram(magnitude_img, angle_img, 8)
    ori_histo_normalized = get_block_descriptor(ori_histo, 2)

    hog = ori_histo_normalized.ravel()

    # visualize to verify
    visualize_hog(im, hog, 8, 2)

    return hog

def get_hog(im):
    # convert grey-scale image to double format
    im = im.astype('float') / 255.0

    filter_x, filter_y = get_differential_filter()
    vertical_edge = filter_image(im, filter_x)
    horizontal_edge = filter_image(im, filter_y)
    magnitude_img, angle_img = get_gradient(vertical_edge, horizontal_edge)
    ori_histo = build_histogram(magnitude_img, angle_img, 8)
    ori_histo_normalized = get_block_descriptor(ori_histo, 2)

    hog = ori_histo_normalized.ravel()

    return hog


# visualize histogram of each block
def visualize_hog(im, hog, cell_size, block_size):
    num_bins = 6
    max_len = 7  # control sum of segment lengths for visualized histogram bin of each block
    im_h, im_w = im.shape
    num_cell_h, num_cell_w = int(im_h / cell_size), int(im_w / cell_size)
    num_blocks_h, num_blocks_w = num_cell_h - block_size + 1, num_cell_w - block_size + 1
    histo_normalized = hog.reshape((num_blocks_h, num_blocks_w, block_size**2, num_bins))
    histo_normalized_vis = np.sum(histo_normalized**2, axis=2) * max_len  # num_blocks_h x num_blocks_w x num_bins
    angles = np.arange(0, np.pi, np.pi/num_bins)
    mesh_x, mesh_y = np.meshgrid(np.r_[cell_size: cell_size*num_cell_w: cell_size], np.r_[cell_size: cell_size*num_cell_h: cell_size])
    mesh_u = histo_normalized_vis * np.sin(angles).reshape((1, 1, num_bins))  # expand to same dims as histo_normalized
    mesh_v = histo_normalized_vis * -np.cos(angles).reshape((1, 1, num_bins))  # expand to same dims as histo_normalized

    plt.figure(1)
    plt.imshow(im, cmap='gray', vmin=0, vmax=1)
    for i in range(num_bins):
        plt.quiver(mesh_x - 0.5 * mesh_u[:, :, i], mesh_y - 0.5 * mesh_v[:, :, i], mesh_u[:, :, i], mesh_v[:, :, i],
                   color='white', headaxislength=0, headlength=0, scale_units='xy', scale=1, width=0.002, angles='xy')
    plt.show()


def IoU(bounding_box1, bounding_box2, m, n):
    diff_x = np.abs(bounding_box1[0]-bounding_box2[0])
    diff_y = np.abs(bounding_box1[1]-bounding_box2[1])
    if (diff_x >= m and diff_y >= n):
        return 0
    elif (diff_x < m and diff_y < n):
        diff_area = m*diff_y+n*diff_x-diff_x*diff_y
        IoU = (m*n-diff_area)/(m*n+diff_area)
        return IoU
    else:
        return 0

def remove_duplicates(bounding_boxes, m, n):
    bounding_boxes = bounding_boxes[np.argsort(-bounding_boxes[:,-1])]
    bounding_box1 = bounding_boxes[0]
    idx = 0

    while (idx < bounding_boxes.shape[0]):
        bounding_box1 = bounding_boxes[idx]
        new_bounding_boxes = np.array(bounding_boxes[:idx+1,:]).reshape(idx+1,3)

        for i in range(idx+1, bounding_boxes.shape[0]):
            if IoU(bounding_box1, bounding_boxes[i], m, n) < 0.5:
                new_bounding_boxes = np.append(new_bounding_boxes, bounding_boxes[i].reshape(1,3), axis = 0)

        bounding_boxes = new_bounding_boxes
        idx += 1
    
    return bounding_boxes

def face_recognition(I_target, I_template):
    m,n = I_template.shape
    template_hog = get_hog(I_template)
    temp_mean = np.mean(template_hog)
    template_hog -= np.mean(template_hog)
    x,y = I_target.shape
    x -= m
    y -= n
    temp_bounding_boxes = np.array([]).reshape(0,3)
    threshold = 0.45

    for i in range(0, x):
        for j in range(0, y):
            cur_target = I_target[i:i+m, j:j+n]
            cur_target_hog = get_hog(cur_target)
            cur_target_hog = cur_target_hog-np.mean(cur_target_hog)
            cur_target_mean = np.mean(cur_target_hog)
            cur_target_hog -= cur_target_mean
            # zero mean
            cur_s = np.dot(template_hog, cur_target_hog)/(np.linalg.norm(template_hog)*np.linalg.norm(cur_target_hog))
            if cur_s >= threshold:
                temp_bounding_boxes = np.append(temp_bounding_boxes, np.array([j, i, cur_s]).reshape(1,3), axis = 0)

    bounding_boxes = temp_bounding_boxes
    bounding_boxes = remove_duplicates(bounding_boxes, m, n)
    return  bounding_boxes


def visualize_face_detection(I_target,bounding_boxes,box_size):

    hh,ww,cc=I_target.shape

    fimg=I_target.copy()
    for ii in range(bounding_boxes.shape[0]):

        x1 = bounding_boxes[ii,0]
        x2 = bounding_boxes[ii, 0] + box_size 
        y1 = bounding_boxes[ii, 1]
        y2 = bounding_boxes[ii, 1] + box_size

        if x1<0:
            x1=0
        if x1>ww-1:
            x1=ww-1
        if x2<0:
            x2=0
        if x2>ww-1:
            x2=ww-1
        if y1<0:
            y1=0
        if y1>hh-1:
            y1=hh-1
        if y2<0:
            y2=0
        if y2>hh-1:
            y2=hh-1
        fimg = cv2.rectangle(fimg, (int(x1),int(y1)), (int(x2),int(y2)), (255, 0, 0), 1)
        cv2.putText(fimg, "%.2f"%bounding_boxes[ii,2], (int(x1)+1, int(y1)+2), cv2.FONT_HERSHEY_SIMPLEX , 0.5, (0, 255, 0), 2, cv2.LINE_AA)


    plt.figure(3)
    plt.imshow(fimg, vmin=0, vmax=1)
    plt.show()




if __name__=='__main__':

    im = cv2.imread('cameraman.tif', 0)
    hog = extract_hog(im)

    I_target= cv2.imread('target.png', 0)
    #MxN image

    I_template = cv2.imread('template.png', 0)
    #mxn  face template

    bounding_boxes=face_recognition(I_target, I_template)

    I_target_c= cv2.imread('target.png')
    # MxN image (just for visualization)
    visualize_face_detection(I_target_c, bounding_boxes, I_template.shape[0])
    # this is visualization code.