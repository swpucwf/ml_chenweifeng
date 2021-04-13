import cv2 as cv
import numpy as np
# 相关函数
# cv.VideoCapture()	初始化摄像头，0开启第一个摄像头，1开启第2个摄像头，返回摄像头对象，一般会自动打开摄像头
# cap.read()	读取摄像头帧，返回值1表示是否成功读取帧，返回值2表示该帧
# cv.cvtColor(frame,mode)	转换图片的色彩空间
# cap.release()	关闭摄像头
# cap.isOpened()	检查摄像头是否打开
# cap.open()	打开摄像头
# cap.get(propld)	获得该帧的大小
# cap.set(propld,value)	设置该帧的大小

video = cv.VideoCapture(0,cv.CAP_DSHOW)
# 设置编码格式
# MP4
# fourcc = cv.VideoWriter_fourcc(*"mp4v")
# avi
# fourcc_2 = cv.VideoWriter_fourcc(*'XVID')
# out_video = cv.VideoWriter('output.mp4',fourcc, 20.0, (640,480))
# out_video_2 = cv.VideoWriter('ori.avi',fourcc, 20.0, (640,480))
# 背景减法器 基于自适应混合高斯背景建模的背景减除法
# history：用于训练背景的帧数，默认为500帧，如果不手动设置learningRate，history就被用于计算当前的learningRate，此时history越大，learningRate越小，背景更新越慢；
# varThreshold：方差阈值，用于判断当前像素是前景还是背景。一般默认16，如果光照变化明显，如阳光下的水面，建议设为25,36，具体去试一下也不是很麻烦，值越大，灵敏度越低；
# detectShadows:是否检测影子，设为true为检测，false为不检测，检测影子会增加程序时间复杂度，如无特殊要求，建议设为false

backsub = cv.createBackgroundSubtractorKNN(history=500,dist2Threshold=16,detectShadows=True)
while True:
    # ret 读取状态,frame image data
    ret,frame = video.read()
    # 获取掩码
    if ret:
        mask = backsub.apply(frame)
        # print(frame.shape)
        # print(mask.shape)
        # 扩充维度
        # mask = np.expand_dims(mask,axis=2).repeat(3,axis=2)
        # out_video.write(mask)
        # out_video_2.write(frame)
        # 膨胀一下
        mask = cv.dilate(mask,kernel=None,dst=3)
        # 任务最大轮廓提取
        cnts,_ = cv.findContours(mask,cv.RETR_CCOMP,cv.CHAIN_APPROX_SIMPLE)
        cv.imshow("frame",mask)
    if cv.waitKey(30) & 0xFF ==ord('q'):
        break
#     释放资源
video.release()
# out_video.release()
# out_video_2.release()
cv.destroyAllWindows()