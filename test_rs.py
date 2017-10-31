from PIL import Image
import numpy as np
from model import build_model

input_size=28
output_size= 4*input_size
model = build_model((input_size,input_size,3))
model.load_weights("my_model.h5")
model.summary()

input_path = "/home/reza/Git/DeepSuperResolution/dataset/DIV2K_valid_LR_bicubic/X4/0805x4.png"

lr_pil_image = Image.open(input_path)
print lr_pil_image.size
test_img = np.array(lr_pil_image)


test_img = test_img / 256.0
H, W, C = test_img.shape

bicubic_4x = lr_pil_image.resize((W*4,H*4),Image.BICUBIC)
bicubic_4x.save("bicubic.png")

res=np.zeros((H * 4, W * 4, 3))

for i in range(0,H - input_size,input_size):
    for j in range(0, W - input_size, input_size):
        res [i * 4 : (i + input_size) * 4, j * 4 : (j + input_size) * 4, : ] = \
        model.predict(np.expand_dims(test_img[i: i + input_size, j: j + input_size, :],axis = 0),batch_size = 1)
    print i, "/",H

res = (res * 128 ) + 128

hr_pil_img=Image.fromarray(res.astype('uint8'))
hr_pil_img.save("dnn_result.png")
print "Done!"