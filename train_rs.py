from dataset.div2k import read_dataset
from model import build_model
from keras import losses
from keras.callbacks import ReduceLROnPlateau, CSVLogger, EarlyStopping
import numpy as np

input_size=96
output_size= 4*input_size

hr_train, hr_test , lr_train,lr_test= read_dataset("/home/reza/Git/DeepSuperResolution/dataset","DIV2K_train_HR", "DIV2K_valid_HR",
                                 "DIV2K_train_LR_bicubic/X4","DIV2K_valid_LR_bicubic/X4",patch_size=input_size)

print "Reading dataset finished!"
print "hr train size:",hr_train.shape, ", hr test size:", hr_test.shape, ", lr train size:", lr_train.shape,\
      "lr test size:", lr_test.shape




model = build_model((input_size,input_size,3))



model.compile(loss="mean_squared_error", optimizer='adam', metrics=['accuracy'])


lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-6)
early_stopper = EarlyStopping(min_delta=0.001, patience=10)
csv_logger = CSVLogger('sr_mse.csv')


model.fit(lr_train,hr_train,batch_size=16, nb_epoch=10, validation_data=(lr_test,hr_test),
          callbacks=[lr_reducer, early_stopper, csv_logger])

