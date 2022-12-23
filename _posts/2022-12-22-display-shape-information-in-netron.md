---
title: "Display shape information in Netron"
excerpt_separator: "<!--more-->"
categories:
  - Blog
tags:
  - python
  - onnx
  - netron
---
## Display shape information in Netron

[Netron](https://github.com/lutzroeder/netron) is an amazing viewer for neural network and machine learning models. It is extremely versatile in the shear number of types of models it can handle. However, one feature that I miss is shape information for the modules. One could sense a theme with me and shapes from my previous post [Get all Input & Output Sizes of for Layers of a PyTorch Model](https://sshiv.github.io/blog/Print-the-input-output-sizes-of-a-PyTorch-model/).

While it is possible for supporting this feature by modifying the viewer code. There is a bit easier path for ONNX models, by embedding the size information directly into the model.

```python
import onnx

original_model = onnx.load('mobilenetv2-7.onnx')
shape_inferred_model = \
              onnx.shape_inference.infer_shapes(original_model)
onnx.save(shape_inferred_model, 'mobilenetv2-7_shape_inferred.onnx')

```
Here is how the output looks after shape data is added to the model.

![merged.png]({{site.baseurl}}/assets/images/20221222-netron-merged.png)

Note this does increase the size of the onnx file slightly.

