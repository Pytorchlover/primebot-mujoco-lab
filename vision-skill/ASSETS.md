# 图片来源与使用方式

- `assets/washer.jpg`：Adrian Pingstone（Arpingstone），[Washer.600pix.jpg](https://commons.wikimedia.org/wiki/File:Washer.600pix.jpg)，作者释入公有领域。实验生成缩放、亮度降低、灰条遮挡版本，并绘制模型预测框。
- `assets/laundry.jpg`：Fanti Salms，[Front Load Washing Machine.jpg](https://commons.wikimedia.org/wiki/File:Front_Load_Washing_Machine.jpg)，[CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/)。实验纠正 EXIF 方向、缩放、降低亮度、添加灰条并绘制预测框。含该照片的衍生图按 CC BY-SA 3.0 提供。
- `assets/no_washer.png`：本项目此前使用启元 Q1 开源模型实际渲染的 MuJoCo 画面，用作“没有洗衣机”的输入；不是实拍。
- `assets/blank.jpg`：程序生成的纯灰故障输入。

两张实拍照片及其变换只用于接口调试，不构成泛化能力评测集。模型没有用这些图片训练或微调。
