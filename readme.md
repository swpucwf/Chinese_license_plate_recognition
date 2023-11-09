## 1.CCPD2019/2020

### 1.1 数据分布组成

CCPD数据集主要采集于安徽某停车场一段时间内的数据,所有图片尺寸固定为720×1160(w×h),大约包含25w+的各种场景图片，如下图所示：

|      类别      |                      描述                       |      图片数      |
| :------------: | :---------------------------------------------: | :--------------: |
|   CCPD-Base    |                  通用车牌图片                   | $200 \mathrm{k}$ |
|    CCPD-FN     |       车牌离摄像头拍摄位置相对较近或较远        | $20 \mathrm{~K}$ |
|    CCPD-DB     |        车牌区域亮度较亮、较暗或者不均匀         | $20 \mathrm{k}$  |
|  CCPD-Rotate   | 车牌水平倾斜 20 到 50 度，竖直倾斜- 10 到 10 度 | $10 \mathrm{~K}$ |
|   CCPD-Tilt    | 车牌水平倾斜 15 到 45 度，竖直倾斜 15 到 45 度  | $10 \mathrm{~K}$ |
|  CCPD-Weather  |            车牌在雨雪雾天气拍摄得到             | $10 \mathrm{k}$  |
| CCPD-Challenge |      在车牌检测识别任务中较有挑战性的图片       | $10 \mathrm{~K}$ |
|   CCPD-Blur    |      由于摄像机镜头抖动导致的模楜车牌图片       |  $5 \mathrm{k}$  |
|    CCPD-NP     |             没有安装车牌的新车图片              |  $5 \mathrm{k}$  |

### 1.2 文件名字解析

以**01-90_89-279&506_467&562-464&564_281&560_284&498_467&502-0_0_32_8_26_26_30-127-34.jpg**为例

```markdown
1、01:车牌占整个界面比例；（一般没用，可忽略）
2、90_89: 车牌的水平角度和垂直角度
3、279&506_467&562: 车牌标注框左上角和右下角的坐标
4、464&564_281&560_284&498_467&502：车牌四个顶点的坐标，顺序为右下、左下、左上、右上
5、0_0_32_8_26_26_30: 这个代表着和省份 (第一位)、地市 (第二位)、车牌号 (剩余部分) 的映射关系
6、127: 亮度，值越大亮度越高（仅供参考）
7、34：模糊度，值越小越模糊（仅供参考)
```

![image-20231107230109360](https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/images/image-20231107230109360.png)

![image-20231107230144545](https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/images/image-20231107230144545.png)

![image-20231107230207842](https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/images/image-20231107230207842.png)

#### 1.2.1 处理ccpd数据集

详情见ccpd_process.py

```python
import os
def update_txt(file_root = r"I:/CCPD2019/ccpd",save_img_path=r"G:/data\images",save_txt_path="G:/data/labels"):
    print(file_root, "start!!!!!")
    file_list = []
    count = 0
    allFilePath(file_root, file_list)
    # print(file_list)
    # exit()
    for img_path in file_list:
        count += 1
        text_path = img_path.replace(".jpg", ".txt")
        # 读取图片
        img = cv2.imread(img_path)
        rect, landmarks, landmarks_sort = get_rect_and_landmarks(img_path)
        # annotation=x1x2y1y2_yolo(rect,landmarks,img)
        annotation = xywh2yolo(rect, landmarks_sort, img)
        str_label = "0 "
        for i in range(len(annotation[0])):
            str_label = str_label + " " + str(annotation[0][i])
        str_label = str_label.replace('[', '').replace(']', '')
        str_label = str_label.replace(',', '') + '\n'
        shutil.move(img_path,os.path.join(os.path.join(save_img_path,os.path.basename(img_path))))
        text_path_save = os.path.join(save_txt_path,os.path.basename(text_path))
        with open(text_path_save, "w") as f:
            f.write(str_label)
        print(text_path,"finished!")
    print(os.getpid(),"end!!!")


```

#### 1.2.2 处理车牌为yolo格式

```python
    # 1. 处理ccpd文件夹
    import multiprocessing
    pool = multiprocessing.Pool(processes=14)  # 这里使用4个进程
    imgs_path = r"G:\CCPD2019.tar\CCPD2019"
    files = []
    for dir in os.listdir(imgs_path):
        if dir =="ccpd_np":
            # 过滤掉无车牌
            continue
        else:
            dir_path = os.path.join(imgs_path, dir)
            if not os.path.isfile(dir_path):
                files.append(dir_path)
    # 使用进程池执行任务
    results = pool.map(update_txt,files)
    # 关闭进程池，防止新任务被提交
    pool.close()
    # 等待所有任务完成
    pool.join()
```

#### 1.2.3 清除异常的数据集

- 清除只有jpg却没有对应txt的文件

```python
import os 
def delete_non_jpg_images(image_folder):
    for filename in os.listdir(image_folder):
        if not filename.endswith(".jpg"):
            file_path = os.path.join(image_folder, filename)
            os.remove(file_path)
            print("删除完毕")
            
    # 2. 清理异常文件夹
    # 调用删除非jpg图像的函数
image_folder = r"G:\data\images"
    # 删除文件
delete_non_jpg_images(image_folder)

```
- [X] 1.单行蓝牌
- [X] 2.单行黄牌
- [X] 3.新能源车牌
- [X] 4.白色警用车牌
- [X] 5.教练车牌
- [X] 6.武警车牌
- [X] 7.双层黄牌
- [X] 8.双层白牌
- [X] 9.使馆车牌
- [X] 10.港澳粤Z牌
- [X] 11.双层绿牌
- [X] 12.民航车牌