import numpy as np


NYU14_name_list = ['Unknown', 'Bed', 'Books', 'Ceiling', 'Chair',
                  'Floor', 'Furniture', 'Objects', 'Picture',
                  'Sofa', 'Table', 'TV', 'Wall', 'Window'
                 ]

def formatString(means,name):
    def numpy_to_string(x):
        string = ''
        for n in x:
            string += '%5.3f\t' % n
        return string
    return '{}{}'.format(numpy_to_string(means[name]), '%5.3f' % means[name].mean())

class IoU_base():
    """
    Given two [n,class] tensors, calculating per class IoU and average IoU
    """
    def __init__(self):
        self.epsilon = 1e-6
    def __call__(self, pred, gt):
        class_num = pred.shape[-1]
        output = np.ones([])
        
        intersections = np.zeros([class_num])
        unions = np.zeros([class_num])
        output = np.zeros([class_num])
        
        for class_n in range(class_num):
            pred_c = pred[:, class_n].flatten().astype(int)
            gt_c   = gt[:, class_n].flatten().astype(int)
            intersection = (pred_c & gt_c).sum() # if one of them is 0: 0
            union = (pred_c | gt_c).sum() # if both 0: 0        
            iou = (intersection+self.epsilon)/(union+self.epsilon)
            intersections[class_n] = intersection
            unions[class_n] = union
            output[class_n] = iou
           
        return output, intersections, unions#, self.valid
    
class PerClassAccRecall_Base():
    """
    Given two [n,class] tensors, calculating per class accuracy
    """
    def __init__(self):
        self.epsilon = 1e-6
    def __call__(self, pred, gt):
        class_num = pred.shape[-1]
        accs = np.zeros([class_num])
        recalls = np.zeros([class_num])
        # valid = np.zeros([class_num])
        intersections = np.zeros([class_num])
        gt_sums = np.zeros([class_num])
        pred_sums = np.zeros([class_num])

        for class_n in range(class_num):
            pred_c = pred[:, class_n].flatten().astype(int)
            gt_c   = gt[:, class_n].flatten().astype(int)         
            intersection = (pred_c & gt_c).sum() # if one of them is 0: 0
            gt_sum = gt_c.sum()
            pred_sum = pred_c.sum()
            acc = (intersection+self.epsilon)/(gt_sum+self.epsilon) #
            recall = (intersection+self.epsilon)/(pred_sum+self.epsilon)

            accs[class_n] = acc
            recalls[class_n] = recall
            # valid[class_n] = valid 
            intersections[class_n] += intersection
            gt_sums[class_n] += gt_sum
            pred_sums[class_n] += pred_sum
            
        return accs, recalls, intersections, \
            gt_sums, pred_sums
        
        
def onehot(x, class_num):
    tmp = np.clip(x.flatten(),0, class_num-1)
    out = np.zeros((tmp.size,class_num))
    out[np.arange(tmp.size), tmp] = 1
    return out
class IoU():
    def __init__(self, class_num):
        self.iou = IoU_base()
        self.class_num=class_num

    def __call__(self, pred_, gt_):
        """
        Parameters
        ----------
        pred_ : TYPE
            Shape=[batch, class, ...]
        gt_ : TYPE
            Shape=[batch, ...]
        class_num : TYPE
            Number of classes
        mask_ : TYPE, optional
            Shape=[batch, ...]. Larger than 0 are the region to be masked

        Returns
        -------
            IoU, Intersections, Unions, Valid

        """
        volume = onehot(pred_,self.class_num)
        gt = onehot(gt_,self.class_num)
        
        return self.iou(volume,gt)
    
class PerClassAccRecall():
    def __init__(self,class_num):
        self.acc = PerClassAccRecall_Base()
        self.class_num=class_num

    def __call__(self, pred_, gt_):
        """
        Parameters
        ----------
        pred_ : TYPE
            Shape=[batch, class, ...]
        gt_ : TYPE
            Shape=[batch, ...]
        class_num : TYPE
            Number of classes
        mask_ : TYPE, optional
            Shape=[batch, ...]. Larger than 0 are the region to be masked

        Returns
        -------
            Accuracy, Recall, Valid

        """
        
        volume = onehot(pred_, self.class_num)
        gt = onehot(gt_, self.class_num)
        return self.acc(volume,gt)
        
def test_basic():
    batch=1
    class_num=3
    
    volume_pred = np.random.randint(0, class_num, size=(batch,15))
    full_gt = np.random.randint(0, class_num,size=(batch,15))
   
    print('volume_pred', volume_pred)
    print('gt', full_gt)
    
    # Semantic 
    print("semantic: ")
    iou_ = IoU(class_num)
    acc_ = PerClassAccRecall(class_num)
    ious = iou_(volume_pred, full_gt)
    print('ious',ious)
    accs = acc_(volume_pred, full_gt)
    print('accs',accs)
    
    # Completion
    print("completion: ")
    iou_ = IoU(2)
    acc_ = PerClassAccRecall(2)
    ious = iou_(volume_pred, full_gt)
    print('ious',ious)
    accs = acc_(volume_pred, full_gt)
    print('accs',accs)
    
if __name__ == "__main__":
    test_basic()
    # test()
    
    # import numpy as np
    # a = np.load('/media/sc/SSD1TB/TrainingData/SceneNet_train/gt/scenenet_rgbd_train_1/00000000.npy')
    # gt = torch.from_numpy(a)
    # gt = gt.long()
    # class_num = 14
    # gt = torch.nn.functional.one_hot(gt,class_num).view(-1,class_num)
    # for i in range(class_num):
    #     print(gt[:,i].sum().item())
        
    