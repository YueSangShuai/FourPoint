## 1.Pytorch->TensorRT

 ```shell
 python export.py --weights "torch's path" --onnx2trt  --fp16_trt 
 ```


## 2.TensorRT inference
```shell
python torch2trt/main.py --trt_path "trt's path"
```
Image preprocessing -> TensorRT inference -> visualization 



```shell
python torch2trt/speed.py --torch_path "torch's path" --trt_path "trt's path"
```




