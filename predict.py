'''
This file is designed for prediction of bounding boxes for a single image.

Predictions could be made in two ways: command line style or service style. Command line style denotes that one can 
run this script from the command line and configure all options right in the command line. Service style allows 
to call :func:`initialize` function once and call :func:`hot_predict` function as many times as it needed to. 

'''

import tensorflow as tf
import os, json, subprocess, random
from optparse import OptionParser
from os import path

from scipy.misc import imread, imresize, imsave
import numpy as np
from PIL import Image, ImageDraw

from train import build_forward
from utils.annolist import AnnotationLib as al
from utils.train_utils import add_rectangles, rescale_boxes
from utils.data_utils import Rotate90

if __package__ is None:
    import sys
    sys.path.append(path.abspath(path.join(path.dirname(__file__), path.pardir, 'detect-widgets/additional')))

from geometry import iou


def initialize(weights_path, hypes_path, options=None):
    '''Initialize prediction process.

    All long running operations like TensorFlow session start and weights loading are made here.

    Args:
        weights_path (string): The path to the model weights file. 
        hypes_path (string): The path to the hyperparameters file. 
        options (dict): The options dictionary with parameters for the initialization process.

    Returns (dict):
        The dict object which contains `sess` - TensorFlow session, `pred_boxes` - predicted boxes Tensor, 
          `pred_confidences` - predicted confidences Tensor, `x_in` - input image Tensor, 
          `hypes` - hyperparametets dictionary.
    '''

    H = prepare_options(hypes_path, options)

    tf.reset_default_graph()
    x_in = tf.placeholder(tf.float32, name='x_in', shape=[H['image_height'], H['image_width'], 3])
    if H['use_rezoom']:
        pred_boxes, pred_logits, pred_confidences, pred_confs_deltas, pred_boxes_deltas \
            = build_forward(H, tf.expand_dims(x_in, 0), 'test', reuse=None)
        grid_area = H['grid_height'] * H['grid_width']
        pred_confidences = tf.reshape(
            tf.nn.softmax(tf.reshape(pred_confs_deltas, [grid_area * H['rnn_len'], H['num_classes']])),
            [grid_area, H['rnn_len'], H['num_classes']])
        if H['reregress']:
            pred_boxes = pred_boxes + pred_boxes_deltas
    else:
        pred_boxes, pred_logits, pred_confidences = build_forward(H, tf.expand_dims(x_in, 0), 'test', reuse=None)

    saver = tf.train.Saver()
    sess = tf.Session()
    sess.run(tf.initialize_all_variables())
    saver.restore(sess, weights_path)
    return {'sess': sess, 'pred_boxes': pred_boxes, 'pred_confidences': pred_confidences, 'x_in': x_in, 'hypes': H,
            'sliding_predict': options['sliding_predict'], 'classID': options['classID']}


def hot_predict(image_path, parameters, to_json=True, verbose=False):
    '''Makes predictions when all long running preparation operations are made.

    Args:
        image_path (string): The path to the source image.
        parameters (dict): The parameters produced by :func:`initialize`.

    Returns (Annotation):
        The annotation for the source image.
    '''

    H = parameters['hypes']
    # The default options for prediction of bounding boxes.
    options = H['evaluate']
    if 'pred_options' in parameters:
        # The new options for prediction of bounding boxes
        for key, val in parameters['pred_options'].items():
            options[key] = val

    # predict
    if 'sliding_predict' in parameters and 'sliding_window' in parameters['sliding_predict'] and parameters['sliding_predict']['sliding_window']:
        if verbose:
            print('sliding')
        return sliding_predict(image_path, parameters, H, options)
    else:
        if verbose:
            print('regular')
        return regular_predict(image_path, parameters, to_json, H, options)


def calculate_medium_box(boxes):
    x1, y1, x2, y2, conf_sum = 0, 0, 0, 0, 0
    new_box = {}
    for box in boxes:
        x1 = x1 + box['x1'] * box['score']
        x2 = x2 + box['x2'] * box['score']
        y1 = y1 + box['y1'] * box['score']
        y2 = y2 + box['y2'] * box['score']
        conf_sum = conf_sum + box['score']
    new_box['x1'] = x1 / conf_sum
    new_box['x2'] = x2 / conf_sum
    new_box['y1'] = y1 / conf_sum
    new_box['y2'] = y2 / conf_sum
    new_box['classID'] = boxes[0]['classID']
    new_box['score'] = conf_sum / len(boxes)
    return new_box


def non_maximum_suppression(boxes):
    conf = [box['score'] for box in boxes]
    ind = np.argmax(conf)
    if isinstance(ind, int):
        return boxes[ind]
    else:
        random.seed()
        num = random.randint(0, len(ind))
        return boxes[num]


def combine_boxes(boxes, iou_min, nms, verbose=False):
    neighbours, result = [], []
    for i, box in enumerate(boxes):
        cur_set = set()
        cur_set.add(i)
        for j, neigh_box in enumerate(boxes):
            if verbose:
                print(i, j, iou(box, neigh_box))
            if i != j and iou(box, neigh_box) > iou_min:
                cur_set.add(j)
        if not len(cur_set):
            result.append(box)

        if len(cur_set):
            for group in neighbours:
                if len(cur_set.intersection(group)):
                    neighbours.remove(group)
                    cur_set = cur_set.union(group)
            neighbours.append(cur_set)

    for group in neighbours:
        cur_boxes = [boxes[i] for i in group]
        if nms:
            medium_box = non_maximum_suppression(cur_boxes)
        else:
            medium_box = calculate_medium_box(cur_boxes)
        result.append(medium_box)

    return result


def process_result_boxes(pred_anno_rects, margin, parameters):
    pred_boxes = []
    for rect in pred_anno_rects:
        pred_boxes.append(to_box(rect, parameters))

    for box in pred_boxes:
        if np.isnan(box['x1']) or np.isnan(box['x2']) or np.isnan(box['y1']) or np.isnan(box['y2']):
            del box
        box['y1'] += margin
        box['y2'] += margin

    return pred_boxes


def to_box(anno_rect, parameters):
    box = {}
    box['x1'] = anno_rect.x1
    box['x2'] = anno_rect.x2
    box['y1'] = anno_rect.y1
    box['y2'] = anno_rect.y2
    box['score'] = anno_rect.score
    if 'classID' in parameters:
        box['classID'] = parameters['classID']
    else:
        box['classID'] = anno_rect.classID
    return box


def regular_predict(image_path, parameters, to_json, H, options):
    orig_img = imread(image_path)[:, :, :3]
    img = Rotate90.do(orig_img)[0] if 'rotate90' in H['data'] and H['data']['rotate90'] else orig_img
    img = imresize(img, (H['image_height'], H['image_width']), interp='cubic')
    np_pred_boxes, np_pred_confidences = parameters['sess']. \
        run([parameters['pred_boxes'], parameters['pred_confidences']], feed_dict={parameters['x_in']: img})

    image_info = {'path': image_path, 'original': orig_img, 'transformed': img}
    pred_anno = postprocess_regular(image_info, np_pred_boxes, np_pred_confidences, H, options)
    result = [r.writeJSON() for r in pred_anno] if to_json else pred_anno
    return result


def sliding_predict(image_path, parameters, H, options):
    orig_img = imread(image_path)[:, :, :3]
    height, width, _ = orig_img.shape
    if 'verbose' in parameters and parameters['verbose']:
        print(width, height)

    assert (parameters['sliding_predict']['step'] > parameters['sliding_predict']['overlap'])

    result = []
    reached_end = False
    for idx, i in enumerate(range(0, height, parameters['sliding_predict']['step'] - parameters['sliding_predict']['overlap'])):
        top, bottom = i, min(height, i + parameters['sliding_predict']['step'])
        if 'verbose' in parameters and parameters['verbose']:
            print(0, top, width, bottom)
        if (height <= i + parameters['sliding_predict']['step']):
            reached_end = True

        img = orig_img[top:bottom, 0:width]
        img = Rotate90.do(img)[0] if 'rotate90' in H['data'] and H['data']['rotate90'] else img
        img = imresize(img, (H['image_height'], H['image_width']), interp='cubic')

        np_pred_boxes, np_pred_confidences = parameters['sess']. \
            run([parameters['pred_boxes'], parameters['pred_confidences']], feed_dict={parameters['x_in']: img})
        image_info = {'path': image_path, 'original': orig_img, 'transformed': img}

        slice_height = bottom - top
        np_pred_boxes = postprocess_single_slice(image_info, parameters, np_pred_boxes, np_pred_confidences, H, options, top, slice_height)

        result.extend(np_pred_boxes)
        if reached_end:
            break
    result = combine_boxes(result, parameters['sliding_predict']['iou_min'], parameters['sliding_predict']['nms'])

    return result


def postprocess_single_slice(image_info, parameters, np_pred_boxes, np_pred_confidences, H, options, margin, slice_height):
    pred_anno = al.Annotation()
    pred_anno.imageName = image_info['path']
    pred_anno.imagePath = os.path.abspath(image_info['path'])
    _, rects = add_rectangles(H, [image_info['transformed']], np_pred_confidences, np_pred_boxes, use_stitching=True,
                              rnn_len=H['rnn_len'], min_conf=options['min_conf'], tau=options['tau'],
                              show_suppressed=False)

    h, w = image_info['original'].shape[:2]
    if 'rotate90' in H['data'] and H['data']['rotate90']:
        # original image height is a width for rotated one
        rects = Rotate90.invert(slice_height, rects)

    rects = [r for r in rects if r.x1 < r.x2 and r.y1 < r.y2 and r.score > options['min_conf']]
    pred_anno.rects = rects
    pred_anno = rescale_boxes((slice_height, H['image_width']), pred_anno, h, w)
    rects = process_result_boxes(pred_anno.rects, margin, parameters)
    return rects


def postprocess_regular(image_info, np_pred_boxes, np_pred_confidences, H, options):
    pred_anno = al.Annotation()
    pred_anno.imageName = image_info['path']
    pred_anno.imagePath = os.path.abspath(image_info['path'])
    _, rects = add_rectangles(H, [image_info['transformed']], np_pred_confidences, np_pred_boxes, use_stitching=True,
                              rnn_len=H['rnn_len'], min_conf=options['min_conf'], tau=options['tau'],
                              show_suppressed=False)

    h, w = image_info['original'].shape[:2]
    if 'rotate90' in H['data'] and H['data']['rotate90']:
        # original image height is a width for rotated one
        rects = Rotate90.invert(h, rects)

    rects = [r for r in rects if r.x1 < r.x2 and r.y1 < r.y2 and r.score > options['min_conf']]
    pred_anno.rects = rects
    pred_anno = rescale_boxes((H['image_height'], H['image_width']), pred_anno, h, w)
    return pred_anno


def prepare_options(hypes_path='hypes.json', options=None):
    '''Sets parameters of the prediction process. If evaluate options provided partially, it'll merge them. 
    The priority is given to options argument to overwrite the same obtained from the hyperparameters file.

    Args:
        hypes_path (string): The path to model hyperparameters file.
        options (dict): The command line options to set before start predictions.

    Returns (dict):
        The model hyperparameters dictionary.
    '''

    with open(hypes_path, 'r') as f:
        H = json.load(f)


    # set default options values if they were not provided
    if options is None:
        if 'evaluate' in H:
            options = H['evaluate']
        else:
            print ('Evaluate parameters were not found! You can provide them through hyperparameters json file '
                   'or hot_predict options parameter.')
            return None
    else:
        if 'evaluate' not in H:
            H['evaluate'] = {}
        # merge options argument into evaluate options from hyperparameters file
        for key, val in options.items():
            H['evaluate'][key] = val

    os.environ['CUDA_VISIBLE_DEVICES'] = str(H['evaluate']['gpu'])
    return H


def save_results(image_path, anno):
    '''Saves results of the prediction.

    Args:
        image_path (string): The path to source image to predict bounding boxes.
        anno (Annotation): The predicted annotations for source image.

    Returns:
        Nothing.
    '''

    # draw
    new_img = Image.open(image_path)
    d = ImageDraw.Draw(new_img)
    rects = anno['rects'] if type(anno) is dict else anno.rects
    for r in rects:
        d.rectangle([r.left(), r.top(), r.right(), r.bottom()], outline=(255, 0, 0))

    # save
    fpath = os.path.join(os.path.dirname(image_path), 'result.png')
    new_img.save(fpath)
    subprocess.call(['chmod', '777', fpath])

    fpath = os.path.join(os.path.dirname(image_path), 'result.json')
    if type(anno) is dict:
        with open(fpath, 'w') as f:
            json.dump(anno, f)
    else:
        al.saveJSON(fpath, anno)
    subprocess.call(['chmod', '777', fpath])


def save_results_boxes(src_path, dst_path, rects, classes):
    '''Saves results of the prediction.

    Args:
        src_path (string): The path to source image to predict bounding boxes.
        dst_path (string): The path to source image to predict bounding boxes.
        rects (list): The collection of boxes to draw on screenshot.
        classes (list): The collection of classes corresponding their ids 

    Returns: 
        Nothing.
    '''

    # draw
    new_img = Image.open(src_path)
    draw = ImageDraw.Draw(new_img)
    for r in rects:
        draw.text(((r['x1'] + r['x2']) / 2, (r['y1'] + r['y2']) / 2),
                  text = classes[r['classID']],fill='purple')
        draw.rectangle([r['x1'], r['y1'], r['x2'], r['y2']], outline=(255, 0, 0))
    # save
    new_img.save(dst_path)
    subprocess.call(['chmod', '644', dst_path])


def main():

    parser = OptionParser(usage='usage: %prog [options] <config>')
    parser.add_option('--gpu', action='store', type='int', default=0)
    parser.add_option('--tau', action='store', type='float', default=0.25)
    parser.add_option('--min_conf', action='store', type='float', default=0.2)

    (options, args) = parser.parse_args()
    if len(args) < 1:
        print ('Provide path configuration json file')
        return

    config = json.load(open(args[0], 'r'))
    image_filename = '75.jpg'

    init_params = initialize(config['weights'], config['hypes'], config)
    pred_anno = hot_predict(image_filename, init_params)
    classes = ['background', 'banner', 'float_banner', 'logo', 'sitename', 'menu', 'navigation', 'button', 'file',
               'social', 'socialGroups', 'goods', 'form', 'search', 'header', 'text', 'image', 'video', 'map', 'table',
               'slider', 'gallery']

    save_results_boxes(image_filename, 'predictions_sliced.png', pred_anno, classes)


if __name__ == '__main__':
    main()
