- 深蹲检测
    - 视频的读取
    ```python
  
  
        相关函数api
        # cv.VideoCapture()	初始化摄像头，0开启第一个摄像头，1开启第2个摄像头，返回摄像头对象，一般会自动打开摄像头
        #　cap.read()	读取摄像头帧，返回值1表示是否成功读取帧，返回值2表示该帧
        # cv.cvtColor(frame,mode)	转换图片的色彩空间
        #　cap.release()	关闭摄像头
        #　cap.isOpened()	检查摄像头是否打开
        # cap.open()	打开摄像头
        # cap.get(propld)	获得该帧的大小
        # cap.set(propld,value)	设置该帧的大小
  
 ```