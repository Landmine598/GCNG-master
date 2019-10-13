import torch

def convert_caffe_to_torch(caffemodel_path, prototxt_path='files/vgg16_20M.prototxt'):
    import caffe
    # caffe中,通过caffe.Net()函数读取模型文件,加载已经训练好的模型
    caffe_model = caffe.Net(prototxt_path, caffemodel_path, caffe.TEST)

    dict = {}
    for caffe_name in list(caffe_model.params.keys()):
        dict[caffe_name + '.weight'] = torch.from_numpy(caffe_model.params[caffe_name][0].data)
        dict[caffe_name + '.bias'] = torch.from_numpy(caffe_model.params[caffe_name][1].data)

    return dict
