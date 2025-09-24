import math
import torch


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def int8_quantize(x, scale, zero_point):
    q = torch.round(x / scale + zero_point)
    return q

def dequantize(q, scale, zero_point):
    x = scale * (q - zero_point)
    return x


if __name__ == "__main__":
    x = torch.tensor([-10.0, -5.0, 0.0, 5.0, 10.0])
    y = torch.sigmoid(x)
    print(y)
    for i in x:
        print(sigmoid(i.item()))
    
    x_min = x.min()
    x_max = x.max()
    q_min = -128
    q_max = 127

    scale = (x_max - x_min) / (q_max - q_min)
    print(f"scale: {scale}")

    zero_point = torch.round(q_min - x_min / scale) 
    print(f"zero_point: {zero_point}")

    for i in x:
        int8_tensor = int8_quantize(i.item(), scale, zero_point)
        deint8_tensor = dequantize(int8_tensor, scale, zero_point)
        print(f"original: {i.item()}, int8: {int8_tensor.item()}, dequantized: {deint8_tensor.item()}")
