from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import pickle
import sys
import statistics
import datetime



from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
import tensorflow as tf
if tf.test.gpu_device_name():
   print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
else:
   print("Please install GPU version of TF")

def split_meaurement_data(path, chop_num, test_nmb):
    with open(path, 'rb') as handle:
        data_o = pickle.load(handle)


    test_data_original = data_o[:chop_num,:]


    

    total_list = list(range(0,chop_num))

    test_list=[]

    for i in range(len(test_nmb)):

        test_list.append([(test_nmb[i]-1)*3,(test_nmb[i]-1)*3+1, (test_nmb[i]-1)*3+2 ])

    test_list_total=[]

    for i in range(len(test_list)):
        test_list_total = test_list_total+test_list[i]


    X_test_never = test_data_original[test_list_total,:-1]
    Y_test_never = test_data_original[test_list_total,-1] -1

    # X_test_never = padding_data(X_test_never)
    # # X_test_measurement = data_meaurement[:,:-1]
    # Y_test_never  = Y_test_never -1

    algo_list =  [item for item in total_list if item not in test_list_total]

    X_test_original = test_data_original[algo_list,:-1]
    Y_test_original = test_data_original[algo_list,-1]-1

  


    return(X_test_never, Y_test_never, X_test_original, Y_test_original , data_o )

sim_data_path = '../data_7th/data_simulation_python_20240908_5n7_out_3s.pickle'

sim_data_path = 'D:\\PHD\\CODE\\Paper\\paper4\\winnie\\data_simulation_winnie_20250403_1n6.pickle'

# sim_data_path = '../data_try/data_simulation_python_20250403_1n7_out.pickle'

testing_data_path = '../data_7th/third_paper_7th_original_data.pickle'

loop_numb =1
chop_num =33

# 6,7,8,6_2nd,7_2nd, 8_2nd, starting at 1

# 1-5, 3 phase, 6-8 6 phase
test_numb=[1, 6,9]

X_test_never, Y_test_never, X_test_original, Y_test_original , data_o= split_meaurement_data(testing_data_path, chop_num, test_numb)

def split_sim_data(path,  test_size=0.33):
    with open(path, 'rb') as handle:
        data_o = pickle.load(handle)
    # if alloption == True:
    #     data_sim = data_o[:np.intc(np.shape(data_o)[0]/3*2),:]
    # else:
    #     row_process = np.random.randint(0, np.shape(data_o)[0]/3*2, size=random_size )
        # data_sim = data_o[row_process, :]


    X_train, X_test, y_train, y_test = train_test_split(
        data_o[:,:-1], data_o[:,-1]-1, test_size=test_size,random_state=42)

    return(X_train, X_test, y_train, y_test, data_o)



X_train, X_test, y_train, y_test, data_sim = split_sim_data(sim_data_path, test_size=0.33)

# pre_score_test = []
# Ypred_loop = []
# pre_score_measurement = []



# change label to 0, 1


# sys.exit()

# ========================= Model ==========================

def data_reshape(X_train):
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    return(X_train) 

X_train = data_reshape(X_train)
X_test = data_reshape(X_test)

X_test_never = data_reshape(X_test_never)
X_test_original = data_reshape(X_test_original)





# change label to 0






def make_model_first(input_shape):
    input_layer = keras.layers.Input(input_shape)

    conv1 = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(input_layer)
    conv1 = keras.layers.BatchNormalization()(conv1)
    conv1 = keras.layers.ReLU()(conv1)

    conv2 = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(conv1)
    conv2 = keras.layers.BatchNormalization()(conv2)
    conv2 = keras.layers.ReLU()(conv2)

    conv3 = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(conv2)
    conv3 = keras.layers.BatchNormalization()(conv3)
    conv3 = keras.layers.ReLU()(conv3)

    x = keras.layers.GlobalAveragePooling1D(data_format = 'channels_first')(conv3)
    x = keras.layers.Dense(128, activation="relu")(x)
    gap = keras.layers.Dropout(0.3)(x)
    output_layer = keras.layers.Dense(2, activation="softmax")(gap)

    return keras.models.Model(inputs=input_layer, outputs=output_layer)


model = make_model_first(input_shape=X_train.shape[1:])
keras.utils.plot_model(model, show_shapes=True)

model.summary()


# sys.exit()

epochs = 500
batch_size = 128
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

callbacks = [
    # keras.callbacks.ModelCheckpoint(
    #     "adaption/cov/best_model_4084_500_20240530_2.h5", save_best_only=True, monitor="val_loss"
    # ),
    # keras.callbacks.ReduceLROnPlateau(
    #     monitor="val_loss", factor=0.5, patience=20, min_lr=0.0001
    # ),
    keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, verbose=1,restore_best_weights=True),
    keras.callbacks.TensorBoard(log_dir=log_dir),
]

# tf.keras.callbacks.TensorBoard(log_dir="./logs")
# log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


model.compile(
    optimizer="adam", 
    loss="sparse_categorical_crossentropy",
    metrics=["sparse_categorical_accuracy"],
)

pre_score = []

pre_score_test = []
never_score = []



pred_labels_never = []

pred_labels_original = []

device = '/device:GPU:0'
with tf.device(device):

    for i in range(loop_numb): 
        print('This is {} round'.format(i))

        history = model.fit(
            X_train,
            y_train,
            batch_size=batch_size,
            epochs=epochs,
            callbacks=callbacks,
            validation_split=0.2,
            verbose=1,
        )



        # model = keras.models.load_model("best_model.h5")

        test_loss, test_score = model.evaluate(X_test, y_test)
        test_never_loss, test_score_never = model.evaluate(X_test_never, Y_test_never)
        test_og_loss, test_score_original= model.evaluate(X_test_original, Y_test_original)

        pre_score_test.append(test_score)


        pre_score.append(test_score_original)

        never_score.append(test_score_never)

        pred_labels_never.append( (model.predict(X_test_never)>0.5).astype("int32"))

        pred_labels_original.append( np.where(model.predict(X_test_original) > 0.5, 1,0))

        

mean_value = statistics.mean(pre_score)
print('real: {}'.format(mean_value) )

mean_value_never = statistics.mean(never_score)
print(never_score)
print('Never seen: {}'.format(mean_value_never) )


mean_value_test = statistics.mean(pre_score_test)

print('synthetic: {}'.format(mean_value_test) )


metric = "sparse_categorical_accuracy"
plt.figure()
plt.plot(history.history[metric])
plt.plot(history.history["val_" + metric])
plt.title("model " + metric)
plt.ylabel(metric, fontsize="large")
plt.xlabel("epoch", fontsize="large")
plt.legend(["train", "val"], loc="best")
plt.show()
plt.close()