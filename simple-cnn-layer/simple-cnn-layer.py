import numpy as np

def conv2d(x, w, b=None, stride=1, padding=0):

    x = np.array(x)
    w = np.array(w)

    N, C_in, H, W = x.shape
    C_out, _, K_h, K_w = w.shape

    # Padding
    if padding > 0:
        x = np.pad(x, ((0,0),(0,0),(padding,padding),(padding,padding)), mode='constant')

    H_p = x.shape[2]
    W_p = x.shape[3]

    out_h = (H_p - K_h)//stride + 1
    out_w = (W_p - K_w)//stride + 1

    output = np.zeros((N, C_out, out_h, out_w))

    for n in range(N):
        for oc in range(C_out):
            for i in range(out_h):
                for j in range(out_w):

                    region = x[n, :, i*stride:i*stride+K_h, j*stride:j*stride+K_w]
                    output[n, oc, i, j] = np.sum(region * w[oc])

                    if b is not None:
                        output[n, oc, i, j] += b[oc]

    return output