import tensorflow as tf
import numpy as np
import scipy.io as sio

### Adult dataset prepare data
def find_means_for_continuous_types(X):
    means = []
    for col in range(len(X[0])):
        summ = 0
        count = 0.000000000000000000001
        for value in X[:, col]:
            if isFloat(value): 
                summ += float(value)
                count +=1
        means.append(summ/count)
    return means

def isFloat(string):
    # credits: http://stackoverflow.com/questions/2356925/how-to-check-whether-string-might-be-type-cast-to-float-in-python
    try:
        float(string)
        return True
    except ValueError:
        return False

def prepare_data(raw_data, means):
    inputs = (
        ("age", ("continuous",)), 
        ("workclass", ("Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov", "Local-gov", "State-gov", "Without-pay", "Never-worked")), 
        ("fnlwgt", ("continuous",)), 
        ("education", ("Bachelors", "Some-college", "11th", "HS-grad", "Prof-school", "Assoc-acdm", "Assoc-voc", "9th", "7th-8th", "12th", "Masters", "1st-4th", "10th", "Doctorate", "5th-6th", "Preschool")), 
        ("education-num", ("continuous",)), 
        ("marital-status", ("Married-civ-spouse", "Divorced", "Never-married", "Separated", "Widowed", "Married-spouse-absent", "Married-AF-spouse")), 
        ("occupation", ("Tech-support", "Craft-repair", "Other-service", "Sales", "Exec-managerial", "Prof-specialty", "Handlers-cleaners", "Machine-op-inspct", "Adm-clerical", "Farming-fishing", "Transport-moving", "Priv-house-serv", "Protective-serv", "Armed-Forces")), 
        ("relationship", ("Wife", "Own-child", "Husband", "Not-in-family", "Other-relative", "Unmarried")), 
        ("race", ("White", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other", "Black")), 
        ("sex", ("Female", "Male")), 
        ("capital-gain", ("continuous",)), 
        ("capital-loss", ("continuous",)), 
        ("hours-per-week", ("continuous",)), 
        ("native-country", ("United-States", "Cambodia", "England", "Puerto-Rico", "Canada", "Germany", "Outlying-US(Guam-USVI-etc)", "India", "Japan", "Greece", "South", "China", "Cuba", "Iran", "Honduras", "Philippines", "Italy", "Poland", "Jamaica", "Vietnam", "Mexico", "Portugal", "Ireland", "France", "Dominican-Republic", "Laos", "Ecuador", "Taiwan", "Haiti", "Columbia", "Hungary", "Guatemala", "Nicaragua", "Scotland", "Thailand", "Yugoslavia", "El-Salvador", "Trinadad&Tobago", "Peru", "Hong", "Holand-Netherlands"))
    )
    input_shape = []
    for i in inputs:
        count = len(i[1])
        input_shape.append(count)
    input_dim = sum(input_shape)
    
    X = raw_data[:, :-1]
    y = raw_data[:, -1:]
    
    # X:
    def flatten_persons_inputs_for_model(person_inputs):
        # global means
        float_inputs = []

        for i in range(len(input_shape)):
            features_of_this_type = input_shape[i]
            is_feature_continuous = features_of_this_type == 1

            if is_feature_continuous:
                mean = means[i]
                if isFloat(person_inputs[i]):
                    scale_factor = 1/(2*mean)  # we prefer inputs mainly scaled from -1 to 1. 
                    float_inputs.append(float(person_inputs[i])*scale_factor)
                else:
                    float_inputs.append(1/2)
            else:
                for j in range(features_of_this_type):
                    feature_name = inputs[i][1][j]

                    if feature_name == person_inputs[i]:
                        float_inputs.append(1.)
                    else:
                        float_inputs.append(0)
        return float_inputs
    
    new_X = []
    for person in range(len(X)):
        formatted_X = flatten_persons_inputs_for_model(X[person])
        new_X.append(formatted_X)
    new_X = np.array(new_X)
    
    # y:
    new_y = []
    for i in range(len(y)):
        if y[i] == ">50k":
            new_y.append((1, 0))
        else:  # y[i] == "<=50k":
            new_y.append((0, 1))
    new_y = np.array(new_y)
    
    return (new_X, new_y)

def get_data(net_info):
    ### Load Dataset
    if net_info['dataset'] == 'cifar-10':
        cifar = tf.keras.datasets.cifar10
        (x_train, y_train), (x_test, y_test) = cifar.load_data()

    elif net_info['dataset'] == 'mnist':
        mnist = tf.keras.datasets.mnist
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train = x_train[..., tf.newaxis]
        x_test = x_test[..., tf.newaxis]

    elif net_info['dataset'] == 'svhn':
        train_loaded = sio.loadmat('data/train_32x32.mat')
        test_loaded = sio.loadmat('data/test_32x32.mat')

        x_train, y_train = train_loaded["X"].astype(np.float32), train_loaded["y"].astype(np.int32)
        ### data is in the shape of (32,32,3,73257)
        x_train = x_train.transpose(3, 0, 1, 2)
        y_train[y_train == 10] = 0
        ### label is in the shape of (73257,1)
        y_train = y_train.reshape(len(y_train))

        x_test, y_test = test_loaded["X"].astype(np.float32), test_loaded["y"].astype(np.int32)
        x_test = x_test.transpose(3, 0, 1, 2)
        y_test[y_test == 10] = 0
        y_test = y_test.reshape(len(y_test))

    elif net_info['dataset'] == 'adult':
        training_data = np.genfromtxt('data/adult.data.txt', delimiter=', ', dtype=str, autostrip=True)
        test_data = np.genfromtxt('data/adult.test.txt', delimiter=', ', dtype=str, autostrip=True)

        means = find_means_for_continuous_types(np.concatenate((training_data, test_data), 0))

        x_train, y_train = prepare_data(training_data, means)
        x_test, y_test = prepare_data(test_data, means)

    ### Preprocessing
    if (net_info['dataset'] == 'mnist') or (net_info['dataset'] == 'svhn') or (net_info['dataset'] == 'cifar-10'):
        x_train, x_test = x_train.astype('float32') / 255.0, x_test.astype('float32') / 255.0
        mean = np.mean(x_train, axis=0)
        std = np.std(x_train, axis=0)
        std[std==0] = 0.00000001
        x_train -= mean
        x_test -= mean
        x_train /= std
        x_test /= std
        y_train = tf.keras.utils.to_categorical(y_train, 10)
        y_test = tf.keras.utils.to_categorical(y_test, 10)

    return x_train, y_train, x_test, y_test

if __name__ == '__main__':
    net_info = {'dataset': 'adult'}
    x_train, y_train, x_test, y_test = get_data(net_info)
