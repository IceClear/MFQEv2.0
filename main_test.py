import glob, os
import numpy as np
import tensorflow as tf
from skimage.measure import compare_psnr, compare_ssim
import net_MFCNN
from utils import *
import subprocess

### Settings
dir_CmpVideo = "/media/iceclear/yuhang/RA_Rec_nof/"
dir_RawVideo = "/media/iceclear/yuhang/YUV_All/"
dir_model = "Models"

file_object = open("record_test.txt", 'w')

os.environ['CUDA_VISIBLE_DEVICES'] = '1'



"""
If there exist frames with different QPs in a video, you can record the QP of each frame in a list. And they will be enhanced by models with corresponding QP.
ApprQP: approximate QP. Because we have only 5 QP models (22,27,32,37,42), so we should record the nearest QP for each QP in a video.
For example, if the QPs for 4 frames are: [21,28,25,33], then we should record: [22,27,27,32], and save it as "ApprQP_BasketballPass_416x240_500.npy".
"""
# dir_ApprQP = "Data"
# opt_QPLabel = True
"""
If all frames in a video are with the same QP:
"""
QP_video = 32 # for the test video in this demo(BasketballPass), all frames are with QP37. Record the QP_video here.
dir_PQFLabel = "Data/QP"+str(QP_video)+"/"
opt_QPLabel = False # no need for uploading QP label.



QP_list = [QP_video]
net1_list = [37,42] # network1 for QP37 and 42, network2 for other QPs. See net_MFCNN for details.

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # only show error and warning
config = tf.ConfigProto(allow_soft_placement = True) # if GPU is not usable, then turn to CPU automatically

BATCH_SIZE = 1
CHANNEL = 1

### List all cmp test videos
# CmpVideo_path_list = glob.glob(os.path.join(dir_CmpVideo, "*.yuv"))
# num_CmpVideo = len(CmpVideo_path_list)


def y_import(video_path, height_frame, width_frame, nfs, startfrm):
    """Import Y channel from a yuv video.

    startfrm: start from 0
    return: (nfs * height * width), dtype=uint8."""

    fp = open(video_path,'rb')

    # target at startfrm
    blk_size = int(height_frame * width_frame * 3 / 2)
    fp.seek(blk_size * startfrm, 0)

    d0 = height_frame // 2
    d1 = width_frame // 2

    Yt = np.zeros((height_frame, width_frame), dtype=np.uint8) # 0-255

    for ite_frame in range(nfs):

        for m in range(height_frame):
            for n in range(width_frame):
                Yt[m,n] = ord(fp.read(1))
        for m in range(d0):
            for n in range(d1):
                fp.read(1)
        for m in range(d0):
            for n in range(d1):
                fp.read(1)

        if ite_frame == 0:
            Y = Yt[np.newaxis, :, :]
        else:
            Y = np.vstack((Y, Yt[np.newaxis, :, :]))

    fp.close()
    return Y


def return_PQFIndices(PQF_label, QP, ApprQP_label):
    """Find all PQFs and their pre/sub PQFs pertain to this QP."""

    PQF_indices = [i for i in range(len(PQF_label)) if PQF_label[i] == 1]

    ApprQPLabel_PQF = [ApprQP_label[i] for i in range(len(ApprQP_label)) if i in PQF_indices]

    PQF_order_part = [o for o in range(len(ApprQPLabel_PQF)) if ApprQPLabel_PQF[o] == QP]
    PQFIndex_list_part = [PQF_indices[o] for o in range(len(PQF_indices)) if o in PQF_order_part]

    if len(PQFIndex_list_part) == 0:
        return [],[],[]

    num_PQF = len(PQFIndex_list_part)

    CmpPQFIndex_list_part = PQFIndex_list_part.copy()
    PrePQFIndex_list_part = PQFIndex_list_part[0: (num_PQF - 1)]
    SubPQFIndex_list_part = PQFIndex_list_part[1: num_PQF]

    PrePQFIndex_list_part = [PQFIndex_list_part[0]] + PrePQFIndex_list_part
    SubPQFIndex_list_part.append(PQFIndex_list_part[-1])

    return PrePQFIndex_list_part, CmpPQFIndex_list_part, SubPQFIndex_list_part


def return_NPIndices(PQF_label, QP, ApprQP_label):
    """Find all non-PQFs and their pre/sub PQFs pertain to this QP."""

    PQFIndex_list = [i for i in range(len(PQF_label)) if PQF_label[i] == 1]

    # Find unqualified non-PQFs and their sub PQFs. Pre PQFs are themselves.
    NonPQFIndex_list = [i for i in range(len(PQF_label)) if (PQF_label[i] == 0) and (i < PQFIndex_list[0])]
    PrePQFIndex_list = NonPQFIndex_list.copy()
    SubPQFIndex_list = [PQFIndex_list[0]] * len(NonPQFIndex_list)

    # Find qualified non-PQFs and their pre/sub PQFs.
    NonPQFIndex_list_good = [i for i in range(len(PQF_label)) if (PQF_label[i] == 0) and (i > PQFIndex_list[0]) and (i < PQFIndex_list[-1])]
    NonPQFIndex_list += NonPQFIndex_list_good
    num_NonPQF = len(NonPQFIndex_list_good)
    for ite_NonPQF in range(num_NonPQF):

        index_NonPQF = NonPQFIndex_list_good[ite_NonPQF]

        for ite_PQF in range(len(PQFIndex_list) - 1):

            if (PQFIndex_list[ite_PQF] < index_NonPQF) and (PQFIndex_list[ite_PQF + 1] > index_NonPQF):

                PrePQFIndex_list.append(PQFIndex_list[ite_PQF])
                SubPQFIndex_list.append(PQFIndex_list[ite_PQF + 1])
                break

    # Find unqualified non-PQFs and their sub PQFs. Sub PQFs are themselves.
    NonPQFIndex_list_bad = [i for i in range(len(PQF_label)) if (PQF_label[i] == 0) and (i > PQFIndex_list[-1])]
    NonPQFIndex_list += NonPQFIndex_list_bad
    PrePQFIndex_list += [PQFIndex_list[-1]] * len(NonPQFIndex_list_bad)
    SubPQFIndex_list += NonPQFIndex_list_bad

    # Find non-PQFs pertain to this QP
    ApprQPLabel_nonPQF = [ApprQP_label[i] for i in range(len(ApprQP_label)) if i in NonPQFIndex_list]

    NonPQF_order_part = [o for o in range(len(ApprQPLabel_nonPQF)) if ApprQPLabel_nonPQF[o] == QP]
    NonPQFIndex_list_part = [NonPQFIndex_list[o] for o in range(len(NonPQFIndex_list)) if o in NonPQF_order_part]

    if len(NonPQFIndex_list_part) == 0:
        return [],[],[]

    PrePQFIndex_list_part = [PrePQFIndex_list[o] for o in range(len(PrePQFIndex_list)) if o in NonPQF_order_part]
    SubPQFIndex_list_part = [SubPQFIndex_list[o] for o in range(len(SubPQFIndex_list)) if o in NonPQF_order_part]

    return PrePQFIndex_list_part, NonPQFIndex_list_part, SubPQFIndex_list_part


def isplane(frame):
    """Detect black frames or other plane frames."""

    tmp_array = np.squeeze(frame).reshape([-1])

    if all(tmp_array[1:] == tmp_array[:-1]): # all values in this frame are equal
        return True
    else:
        return False


def func_enhance(ir_model_pre, QP, PreIndex_list, CmpIndex_list, SubIndex_list, CmpVideo_name, is_PQF=False):
    """Enhance PQFs or non-PQFs, record dpsnr, dssim and enhanced frames."""

    global enhanced_list, sum_dpsnr, sum_dssim

    tf.reset_default_graph()

    ### Defind enhancement process
    if width <= 1920:
        x1 = tf.placeholder(tf.float32, [BATCH_SIZE, height, width, CHANNEL])  # previous
        x2 = tf.placeholder(tf.float32, [BATCH_SIZE, height, width, CHANNEL])  # current
        x3 = tf.placeholder(tf.float32, [BATCH_SIZE, height, width, CHANNEL])  # subsequent

    else: # 2k
        x1 = tf.placeholder(tf.float32, [BATCH_SIZE, int(height / 2), int(width / 2), CHANNEL])  # previous
        x2 = tf.placeholder(tf.float32, [BATCH_SIZE, int(height / 2), int(width / 2), CHANNEL])  # current
        x3 = tf.placeholder(tf.float32, [BATCH_SIZE, int(height / 2), int(width / 2), CHANNEL])  # subsequent

    if QP in net1_list:
        is_training = tf.placeholder_with_default(False, shape=())

    x1to2 = net_MFCNN.warp_img(BATCH_SIZE, x2, x1, False)
    x3to2 = net_MFCNN.warp_img(BATCH_SIZE, x2, x3, True)

    if QP in net1_list:
        x2_enhanced = net_MFCNN.network(x1to2, x2, x3to2, is_training)
    else:
        x2_enhanced = net_MFCNN.network2(x1to2, x2, x3to2)

    saver = tf.train.Saver()

    with tf.Session(config = config) as sess:

        # Restore model
        model_path = os.path.join(dir_model_pre, "model_step2.ckpt-" + str(QP))
        saver.restore(sess, model_path)

        nfs = len(CmpIndex_list)

        sum_dpsnr_part = 0.0
        sum_dssim_part = 0.0

        for ite_frame in range(nfs):

            if width <= 1920:
                # Load frames
                pre_frame = y_import(CmpVideo_path, height, width, 1, PreIndex_list[ite_frame])[:,:,:,np.newaxis] / 255.0
                cmp_frame = y_import(CmpVideo_path, height, width, 1, CmpIndex_list[ite_frame])[:,:,:,np.newaxis] / 255.0
                sub_frame = y_import(CmpVideo_path, height, width, 1, SubIndex_list[ite_frame])[:,:,:,np.newaxis] / 255.0

                # if cmp frame is plane?
                if isplane(cmp_frame):
                    continue

                # if PQF frames are plane?
                if isplane(pre_frame):
                     pre_frame = np.copy(cmp_frame)
                if isplane(sub_frame):
                     sub_frame = np.copy(cmp_frame)

                # Enhance
                if QP in net1_list:
                    enhanced_frame = sess.run(x2_enhanced, feed_dict={x1:pre_frame, x2:cmp_frame, x3:sub_frame, is_training:False})
                else:
                    enhanced_frame = sess.run(x2_enhanced, feed_dict={x1:pre_frame, x2:cmp_frame, x3:sub_frame})

                # Record for output video
                enhanced_list[CmpIndex_list[ite_frame]] = np.squeeze(enhanced_frame)

                # Evaluate and accumulate dpsnr
                raw_frame = np.squeeze(y_import(RawVideo_path, height, width, 1, CmpIndex_list[ite_frame])) / 255.0
                cmp_frame = np.squeeze(cmp_frame)
                enhanced_frame = np.squeeze(enhanced_frame)

                raw_frame = np.float32(raw_frame)
                cmp_frame = np.float32(cmp_frame)

                psnr_ori = compare_psnr(cmp_frame, raw_frame, data_range=1.0)
                psnr_aft = compare_psnr(enhanced_frame, raw_frame, data_range=1.0)

                ssim_ori = compare_ssim(cmp_frame, raw_frame, data_range=1.0)
                ssim_aft = compare_ssim(enhanced_frame, raw_frame, data_range=1.0)

                sum_dpsnr_part += psnr_aft - psnr_ori
                sum_dssim_part += ssim_aft - ssim_ori

            else:

                pre_frame = y_import(CmpVideo_path, height, width, 1, PreIndex_list[ite_frame])[:,:,:,np.newaxis] / 255.0
                cmp_frame = y_import(CmpVideo_path, height, width, 1, CmpIndex_list[ite_frame])[:,:,:,np.newaxis] / 255.0
                sub_frame = y_import(CmpVideo_path, height, width, 1, SubIndex_list[ite_frame])[:,:,:,np.newaxis] / 255.0
                enhance_part_list=[]

                # if cmp frame is plane?
                if isplane(cmp_frame):
                    continue

                # if PQF frames are plane?
                if isplane(pre_frame):
                     pre_frame = np.copy(cmp_frame)
                if isplane(sub_frame):
                     sub_frame = np.copy(cmp_frame)

                for height_start in [0, int(height / 2)]:

                    for width_start in [0, int(width / 2)]:

                        x1_feed = pre_frame[:, height_start: (height_start + int(height / 2)), width_start: (width_start + int(width / 2)), :]
                        x2_feed = cmp_frame[:, height_start: (height_start + int(height / 2)), width_start: (width_start + int(width / 2)), :]
                        x3_feed = sub_frame[:, height_start: (height_start + int(height / 2)), width_start: (width_start + int(width / 2)), :]

                        # Enhance
                        if QP in net1_list:
                            enhanced_frame = sess.run(x2_enhanced, feed_dict={x1:x1_feed, x2:x2_feed, x3:x3_feed, is_training:False})
                        else:
                            enhanced_frame = sess.run(x2_enhanced, feed_dict={x1:x1_feed, x2:x2_feed, x3:x3_feed})

                        enhance_part_list.append(np.squeeze(enhanced_frame))

                enhanced_frame_h1 = np.hstack((enhance_part_list[0],enhance_part_list[1]))
                enhanced_frame_h2 = np.hstack((enhance_part_list[2],enhance_part_list[3]))
                enhanced_frame_sum = np.vstack((enhanced_frame_h1,enhanced_frame_h2))

                # Record for output video
                enhanced_list[CmpIndex_list[ite_frame]] = np.squeeze(enhanced_frame_sum)

                # Evaluate and accumulate dpsnr
                raw_frame = np.squeeze(y_import(RawVideo_path, height, width, 1, CmpIndex_list[ite_frame])) / 255.0
                cmp_frame = np.squeeze(cmp_frame)
                enhanced_frame = np.squeeze(enhanced_frame)

                raw_frame = np.float32(raw_frame)
                cmp_frame = np.float32(cmp_frame)

                psnr_ori = compare_psnr(cmp_frame, raw_frame, data_range=1.0)
                psnr_aft = compare_psnr(enhanced_frame_sum, raw_frame, data_range=1.0)

                ssim_ori = compare_ssim(cmp_frame, raw_frame, data_range=1.0)
                ssim_aft = compare_ssim(enhanced_frame_sum, raw_frame, data_range=1.0)

                sum_dpsnr_part += psnr_aft - psnr_ori
                sum_dssim_part += ssim_aft - ssim_ori

            print("%d | %d at QP = %d" % (ite_frame + 1, nfs, QP), end="\r")
        print("              ", end="\r")

        sum_dpsnr += sum_dpsnr_part
        sum_dssim += sum_dssim_part

        average_dpsnr = sum_dpsnr_part / nfs
        average_dssim = sum_dssim_part / nfs
        if is_PQF:
            print(CmpVideo_name +" -PQF dPSNR: %.3f - dSSIM: %.3f - nfs: %4d" % (average_dpsnr, average_dssim, nfs), flush=True)
            file_object.write(CmpVideo_name + " -PQF dPSNR: %.3f - dSSIM: %.3f - nfs: %4d\n" % (average_dpsnr, average_dssim, nfs))
        else:
            print(CmpVideo_name +"  -NoPQF  dPSNR: %.3f - dSSIM: %.3f - nfs: %4d" % (average_dpsnr, average_dssim, nfs), flush=True)
            file_object.write(CmpVideo_name + " -NoPQF dPSNR: %.3f - dSSIM: %.3f - nfs: %4d\n" % (average_dpsnr, average_dssim, nfs))

        file_object.flush()

### Enhancement video by video
f = open('./testInfo.txt','r')
for c in f.readlines():
    c_array = c.split()
    ### Extract info from cmp video path
    CmpVideo_name = c_array[0]
    print('>>>>>>>>>>>>>> '+CmpVideo_name+' is starting')
    nfs = int(c_array[3])
    width = int(c_array[1])
    height = int(c_array[2])
    CmpVideo_path = "/media/iceclear/yuhang/RA_Rec_nof/"+'rec_nof_RA_'+CmpVideo_name+'_qp'+str(QP_video)+'_nf'+str(nfs)+".yuv"

    RawVideo_name = c_array[0]
    RawVideo_path = "/media/iceclear/yuhang/YUV_All/"+RawVideo_name+'.yuv'

    dir_saveframe = "/media/iceclear/IceKing2/compare_results/QP_"+str(QP_video)+"/MFQE2.0/"
    createpath(dir_saveframe+CmpVideo_name)



    # Load PQF label and ApprQP label
    PQF_label = list(np.load(os.path.join(dir_PQFLabel, "PQFLabel_" + CmpVideo_name + "0.npy")))
    if opt_QPLabel:
        ApprQP_label = list(np.load(os.path.join(dir_ApprQP, "ApprQP_" + CmpVideo_name + ".npy")))
    else:
        ApprQP_label = [QP_video] * nfs

    # Initialize enhanced_list
    enhanced_list = np.zeros((nfs, height, width), dtype=np.float32)

    # Record dpsnr and dssim
    sum_dpsnr = 0.0
    sum_dssim = 0.0

    ### PQF enhancement
    print("enhancing PQF...")
    for QP in QP_list:

        # Find all PQFs and their pre/sub PQFs pertain to this QP
        PrePQFIndex_list_part, CmpPQFIndex_list_part, SubPQFIndex_list_part = return_PQFIndices(PQF_label, QP, ApprQP_label)
        if len(PrePQFIndex_list_part) == 0:
            continue

        # Enhance PQF
        dir_model_pre = dir_model + "/PQF_enhancement" + "/model_QP" + str(QP)
        func_enhance(dir_model_pre, QP, PrePQFIndex_list_part, CmpPQFIndex_list_part, SubPQFIndex_list_part, CmpVideo_name,True)

    ### Non-PQF enhancement
    print("enhancing non-PQFs...")
    for QP in QP_list:

        # Find pre-PQFs, non-PQFs and sub-PQFs pertain to this QP
        PrePQFIndex_list_part, NonPQFIndex_list_part, SubPQFIndex_list_part = return_NPIndices(PQF_label, QP, ApprQP_label)
        if len(PrePQFIndex_list_part) == 0:
            continue

        # Enhance non-PQF
        dir_model_pre = dir_model + "/NP_enhancement" + "/model_QP" + str(QP)
        func_enhance(dir_model_pre, QP, PrePQFIndex_list_part, NonPQFIndex_list_part, SubPQFIndex_list_part, CmpVideo_name,False)

    ### Output and record result
    average_dpsnr = sum_dpsnr / nfs
    average_dssim = sum_dssim / nfs
    print("dPSNR: %.3f - dSSIM: %.3f - nfs: %4d - %s" % (average_dpsnr, average_dssim, nfs, CmpVideo_name), flush=True)
    file_object.write("dPSNR: %.3f - dSSIM: %.3f - nfs: %4d - %s\n" % (average_dpsnr, average_dssim, nfs, CmpVideo_name))
    file_object.flush()

    [y_lq,u_lq,v_lq] = read_YUV420(CmpVideo_path,height,width,nfs+1)
    index = 0
    print(len(enhanced_list))
    for item in enhanced_list:
        saveimg(dir_saveframe,CmpVideo_name,width,height,item, u_lq, v_lq, index)
        index+=1
    ### Output bmp
    # Here we have enhanced_list that records all enhanced frames. If you want to output enhanced images, code here.
    pass
