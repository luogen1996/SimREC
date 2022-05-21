import torch
from layers.conv_layer import aspp_decoder
from layers.losses import *

class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss
def smooth_L1(y_true, y_pred,sigma=3.0):
    sigma_squared = sigma ** 2

    # compute smooth L1 loss
    # f(x) = 0.5 * (sigma * x)^2          if |x| < 1 / sigma / sigma
    #        |x| - 0.5 / sigma / sigma    otherwise
    regression_diff = y_true - y_pred
    regression_diff = torch.abs(regression_diff)
    regression_loss = torch.where(
        regression_diff<(1.0 / sigma_squared),
        0.5 * sigma_squared * torch.pow(regression_diff, 2),
        regression_diff - 0.5 / sigma_squared
    )
    return regression_loss.sum()


class REShead(nn.Module):
    """
    detection layer corresponding to yolo_layer.c of darknet
    """
    def __init__(self, __C, layer_no, in_ch, ignore_thre=0.5):
        """
        Args:
            config_model (dict) : model configuration.
                ANCHORS (list of tuples) :
                ANCH_MASK:  (list of int list): index indicating the anchors to be
                    used in YOLO layers. One of the mask group is picked from the list.
                N_CLASSES (int): number of classes
            layer_no (int): YOLO layer number - one from (0, 1, 2).
            in_ch (int): number of input channels.
            ignore_thre (float): threshold of IoU above which objectness training is ignored.
        """

        super(REShead, self).__init__()
        # strides = [32, 16, 8] # fixed
        self.anchors = __C.ANCHORS
        self.anch_mask = __C.ANCH_MASK[layer_no]
        self.n_anchors = len(self.anch_mask)
        self.n_classes = __C.N_CLASSES
        self.ignore_thre = ignore_thre
        self.l2_loss = smooth_L1#nn.SmoothL1Loss()#nn.MSELoss()
        self.kld = nn.KLDivLoss(reduction='sum')
        h,w=__C.INPUT_SHAPE
        self.stride = 32
        self.sconv=nn.Sequential(aspp_decoder(in_ch,__C.HIDDEN_SIZE//2,1),
                                 nn.UpsamplingBilinear2d(scale_factor=8)
                                 )

    def forward(self, xin,yin, x_label=None,y_label=None):
        """
        In this
        Args:
            xin (torch.Tensor): input feature map whose size is :math:`(N, C, H, W)`, \
                where N, C, H, W denote batchsize, channel width, height, width respectively.
            x_label (torch.Tensor): label data whose size is :math:`(N, K, 5)`. \
                N and K denote batchsize and number of x_label.
                Each label consists of [class, xc, yc, w, h]:
                    class (float): class index.
                    xc, yc (float) : center of bbox whose values range from 0 to 1.
                    w, h (float) : size of bbox whose values range from 0 to 1.
        Returns:
            loss (torch.Tensor): total loss - the target of backprop.
            loss_xy (torch.Tensor): x, y loss - calculated by binary cross entropy (BCE) \
                with boxsize-dependent weights.
            loss_wh (torch.Tensor): w, h loss - calculated by l2 without size averaging and \
                with boxsize-dependent weights.
            loss_obj (torch.Tensor): objectness loss - calculated by BCE.
            loss_cls (torch.Tensor): classification loss - calculated by BCE for each class.
            loss_l2 (torch.Tensor): total l2 loss - only for logging.
        """

        batchsize = xin.shape[0]
        fsize = xin.shape[2]


        mask=self.sconv(yin)

        # print(output.size(),mask.size())

        n_ch = 5 + self.n_classes
        dtype = torch.cuda.FloatTensor if xin.is_cuda else torch.FloatTensor
        devices=xin.device


        if not self.training:  # not training
            mask=(mask.sigmoid()>0.35).float()
            box=torch.zeros(batchsize,5,device=devices)
            return box,mask.squeeze(1)



        loss_seg=nn.BCEWithLogitsLoss(reduction='sum')(mask,y_label)/640./batchsize

        loss=loss_seg.sum()
        return loss,torch.zeros_like(loss),loss_seg
