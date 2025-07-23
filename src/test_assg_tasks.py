import numpy as np
import pandas as pd
import sklearn
import random
#import unittest
from twisted.trial import unittest
from assg_tasks import basic_sigmoid
from assg_tasks import sigmoid
from assg_tasks import sigmoid_grad
from assg_tasks import standard_scalar
from assg_tasks import softmax
from assg_tasks import one_hot
from assg_tasks import rmse
from assg_tasks import mae


class test_basic_sigmoid(unittest.TestCase):

    def setUp(self):
        pass

    def test_input_3(self):
        s = basic_sigmoid(3)
        self.assertAlmostEqual(s, 0.9525741268224334)

    def test_input_0(self):
        s = basic_sigmoid(0)
        self.assertAlmostEqual(s, 0.5)
        
    def test_input_neg5(self):
        s = basic_sigmoid(-5)
        self.assertAlmostEqual(s, 0.0066928509242848554)

    def test_using_math_library(self):
        # if using math library should not be vectorized, expect an exception
        # if we pass a numpy array
        x = np.array([1, 2, 3])
        with self.assertRaises(TypeError):
            basic_sigmoid(x)

class test_sigmoid(unittest.TestCase):

    def setUp(self):
        pass

    def test_input_scalar(self):
        s = sigmoid(3)
        self.assertAlmostEqual(s, 0.9525741268224334)

    def test_input_vector(self):
        # test a 1-d vector
        x = np.array([-5, 0, 3])
        s = sigmoid(x)
        self.assertTrue(np.allclose(s, np.array([0.00669285, 0.5, 0.95257413])))

    def test_input_matrix(self):
        # test a 3-d tensor
        x = np.linspace(-5, 5, 27).reshape((3,3,3))
        s = sigmoid(x)
        expected_s = np.array(
            [[[0.00669285, 0.00980136, 0.01433278],
              [0.02091496, 0.03042661, 0.04406926],
              [0.06342879, 0.09048789, 0.12751884]],
            
             [[0.17675903, 0.23978727, 0.31664553],
              [0.40501421, 0.5,        0.59498579],
              [0.68335447, 0.76021273, 0.82324097]],
            
             [[0.87248116, 0.90951211, 0.93657121],
              [0.95593074, 0.96957339, 0.97908504],
              [0.98566722, 0.99019864, 0.99330715]]]
        )
        self.assertTrue(np.allclose(s, expected_s))

    def test_input_list(self):
        # a regular list still does not work for a vectorized function
        x = [-5, 0, 3]
        with self.assertRaises(TypeError):
            s = sigmoid(x)
        

class test_sigmoid_grad(unittest.TestCase):

    def setUp(self):
        pass

    def test_input_scalar(self):
        s = sigmoid_grad(3)
        self.assertAlmostEqual(s, 0.045176659730912)

    def test_input_vector(self):
        # test a 1-d vector
        x = np.array([-5, 0, 3])
        s = sigmoid_grad(x)
        self.assertTrue(np.allclose(s, np.array([0.00664806, 0.25, 0.04517666])))

    def test_input_matrix(self):
        # test a 3-d tensor
        x = np.linspace(-5, 5, 27).reshape((3,3,3))
        s = sigmoid_grad(x)
        expected_s = np.array(
        [[[0.00664806, 0.00970529, 0.01412736],
          [0.02047752, 0.02950084, 0.04212716],
          [0.05940558, 0.08229983, 0.11125779]],
        
         [[0.14551528, 0.18228933, 0.21638114],
          [0.2409777,  0.25,       0.2409777 ],
          [0.21638114, 0.18228933, 0.14551528]],
        
         [[0.11125779, 0.08229983, 0.05940558],
          [0.04212716, 0.02950084, 0.02047752],
          [0.01412736, 0.00970529, 0.00664806]]]
        )
        self.assertTrue(np.allclose(s, expected_s))

    def test_input_list(self):
        # a regular list still does not work for a vectorized function
        x = [-5, 0, 3]
        with self.assertRaises(TypeError):
            s = sigmoid_grad(x)


class test_standard_scalar(unittest.TestCase):

    def setUp(self):
        pass

    def test_matrix(self):
        x = np.array(
            [[ 0,  3,   5,  3],
             [ 1,  6,   4,  8],
             [-1, 10, -20, 22]]
        )
        expected_x_scaled = np.array(
            [[ 0.        , -1.16247639,  0.74993067, -0.99483201],
             [ 1.22474487, -0.11624764,  0.66340021, -0.373062  ],
             [-1.22474487,  1.27872403, -1.41333087,  1.36789401]]
        )
        x_scaled, x_mu, x_sigma = standard_scalar(x)
        self.assertTrue(np.allclose(x_scaled, expected_x_scaled))
        self.assertTrue(x_scaled.shape == (3, 4))
        self.assertFalse(np.allclose(x, x_scaled))

        expected_x_mu = np.array([ 0.        ,  6.33333333, -3.66666667, 11.        ])
        self.assertTrue(np.allclose(x_mu, expected_x_mu))

        expected_x_sigma = np.array([ 0.81649658,  2.86744176, 11.55662388,  8.04155872])
        self.assertTrue(np.allclose(x_sigma, expected_x_sigma))
        

    def test_random_matrix(self):
        np.random.seed(1)
        x = np.random.random((20,10))
        x_scaled, x_mu, x_sigma = standard_scalar(x)

        expected_x_scaled = np.array(
    [[ 0.06700099,  0.53830867, -1.6014901 , -0.60718175, -0.84424448,
        -1.30479394, -0.89056906, -0.69249776, -0.2892714 , -0.13552687],
       [ 0.0741006 ,  0.43147262, -0.92839635,  1.54133768, -1.22198481,
         0.58953993, -0.14946092,  0.10313851, -1.13074344, -1.30440105],
       [ 1.3209784 ,  1.29286285, -0.56944027,  0.84805133,  1.46467605,
         1.3239685 , -1.21523366, -1.83672087, -1.03410636,  1.02858051],
       [-0.97440626, -0.37230677,  1.55344369,  0.25416178,  0.8807887 ,
        -0.57351791,  0.71402366,  1.13323981, -1.53148607,  0.58946374],
       [ 1.93572949,  0.62303849, -0.67807777,  1.20984164, -0.98199456,
        -0.1397601 ,  1.42642336, -0.88642051, -0.64699682, -1.53793529],
       [-1.23250675,  0.41204411, -0.90475888, -0.74444699,  0.24692791,
        -1.43250534,  0.35353829, -1.43476181,  0.34266189,  0.41660725],
       [-0.96137509, -0.39376722,  0.68550353, -0.18983003, -1.15057525,
         0.14859572,  0.64119005, -0.06037477,  1.50876423,  0.02824647],
       [ 1.65645513, -1.23549513, -1.1430873 ,  1.27742573, -0.05020649,
        -1.06554639,  1.48708972, -0.68426573,  0.87274604,  0.50662618],
       [ 1.59078344,  0.24416395,  0.87175511, -0.43342335, -0.45446734,
         1.32816117, -0.11486206,  1.61934555,  0.58598511,  0.14880179],
       [-0.92081507,  1.23573251, -0.11984575,  0.42291465, -0.01710599,
        -0.8306988 ,  1.40969236,  0.15909669, -1.58208965,  0.13318957],
       [-0.2283448 , -0.04986459,  1.31644717, -0.40218572,  1.56640189,
         0.43518486, -1.43727611,  1.48718264,  0.67609713,  1.43744637],
       [-0.73260032, -1.23652668,  1.47012444,  0.86482629, -1.09979551,
         0.86804172,  0.93013931,  1.46324328,  0.74380021, -1.55768762],
       [-1.23082973, -1.57410726, -1.50862461, -0.81659702,  1.41290111,
         0.15821161,  0.28522956,  1.16088438, -1.18395852, -1.02623657],
       [ 0.6184221 ,  1.29692317,  0.24617983, -1.66574265,  1.22494507,
        -0.84397815,  1.10087887, -0.53458714,  1.24273877,  0.57909402],
       [ 0.52195603, -1.23859774, -1.40449658, -1.28253579, -1.16766852,
        -1.25513435, -0.76403041,  0.67915558,  0.24554866, -1.94094244],
       [-1.06058985,  1.28986442,  0.26946937, -0.97674329, -0.51016924,
         0.82991054, -0.86115733,  0.18776495,  1.59221293,  0.92115421],
       [-0.51199154, -0.15117211,  0.4402818 ,  1.35798655, -0.81248718,
        -1.54648847, -1.26341907, -0.16693292,  0.39853648, -0.03248837],
       [-0.25867931,  1.35480853,  0.30782748, -0.31684191,  0.43481997,
         0.83485363,  0.658634  , -0.99354083, -1.37379143, -0.71438886],
       [ 0.76207441, -1.01424721,  0.87772757, -1.48704594, -0.48488698,
         1.02955341, -0.86755721,  0.40466645,  0.13052123,  1.18867332],
       [-0.43536188, -1.45313461,  0.81945764,  1.14602879,  1.56412567,
         1.44640236, -1.44327334, -1.1076155 ,  0.43283102,  1.27172365]]            
        )
        x_scaled, x_mu, x_sigma = standard_scalar(x)
        self.assertTrue(np.allclose(x_scaled, expected_x_scaled))
        self.assertTrue(x_scaled.shape == (20, 10))
        self.assertFalse(np.allclose(x, x_scaled))

        expected_x_mu = np.array(
            [0.39651941, 0.54344298, 0.48629492, 0.4650521 , 0.4135424 ,
             0.4905467 , 0.46390008, 0.53106185, 0.48490297, 0.57832147])
        self.assertTrue(np.allclose(x_mu, expected_x_mu))

        expected_x_sigma = np.array(
            [0.30600437, 0.32858752, 0.30358011, 0.26799146, 0.31600623,
             0.3051885 , 0.31175557, 0.26787253, 0.30468099, 0.29149009])
        self.assertTrue(np.allclose(x_sigma, expected_x_sigma))

                                    
class test_softmax(unittest.TestCase):

    def setUp(self):
        pass

    def test_matrix(self):
        x = np.array(
            [[9, 2, 5, 0, 0],
             [7, 5, 0, 0 ,0]]
        )
        expected_s = np.array(
            [[9.80897665e-01, 8.94462891e-04, 1.79657674e-02, 1.21052389e-04, 1.21052389e-04],
             [8.78679856e-01, 1.18916387e-01, 8.01252314e-04, 8.01252314e-04, 8.01252314e-04]]
        )
        s = softmax(x)
        self.assertTrue(np.allclose(s, expected_s))
        self.assertTrue(s.shape == (2,5))
        # the rows of softmax are a normalized probability space so should sum up to 0
        self.assertTrue(np.allclose(np.sum(s, axis=1), np.array([1.0, 1.0]))) 
        # check that original x was not modified
        self.assertFalse(np.allclose(s, x))
        
    def test_random_matrix(self):
        x = np.random.random((20,10))
        s = softmax(x)
        self.assertTrue(s.shape == (20,10))
        # the rows of softmax are a normalized probability space so should sum up to 0
        self.assertTrue(np.allclose(np.sum(s, axis=1), np.ones((20,)))) 
        # check that original x was not modified
        self.assertFalse(np.allclose(s, x))

class test_one_hot(unittest.TestCase):

    def setUp(self):
        pass

    def test_example_case(self):
        category = np.array(['Setosa', 'Versicolor', 'Setosa', 'Virginica', 'Virginica'])
        one_hot_array = one_hot(category)
        expected_one_hot_array = np.array(
            [[1., 0., 0.],
             [0., 1., 0.],
             [1., 0., 0.],
             [0., 0., 1.],
             [0., 0., 1.]])
        self.assertTrue(one_hot_array.shape == (5,3))
        self.assertTrue(np.allclose(one_hot_array, expected_one_hot_array))

    def test_bigger_case(self):
        random.seed(1)
        np.random.seed(1)
        words = ["apple", "book", "desk", "pen", "cat", "dog", "tree", "house", "car", "phone"]
        category = np.array([random.choice(words) for _ in range(30)])

        one_hot_array = one_hot(category)
        expected_one_hot_array = np.array(
           [[0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
            [0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
            [0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 1.],
            [0., 0., 0., 0., 0., 0., 0., 1., 0., 0.],
            [0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
            [1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 1.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 1.],
            [0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
            [1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
            [0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 1., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
            [0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
            [1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
            [1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 1.],
            [0., 0., 0., 0., 0., 0., 0., 1., 0., 0.]]            
        )
        self.assertTrue(one_hot_array.shape == (30, 10))
        self.assertTrue(np.allclose(one_hot_array, expected_one_hot_array))

class test_mae(unittest.TestCase):

    def setUp(self):
        pass

    def test_case1(self):
        y_pred = np.array([.9, 0.2, 0.1, .4, .9])
        y_true = np.array([1, 0, 0, 1, 1])
        loss = mae(y_pred, y_true)
        self.assertAlmostEqual(loss, 0.22000000000000003)

    def test_case2(self):
        y_pred = np.array([0.8, 0.3, 0.7, 0.5, 0.8, 0.6, 0.5, 0.2])
        y_true = np.array([1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
        loss = mae(y_pred, y_true)
        self.assertAlmostEqual(loss, 0.35)


class test_rmse(unittest.TestCase):

    def setUp(self):
        pass

    def test_case1(self):
        y_pred = np.array([.9, 0.2, 0.1, .4, .9])
        y_true = np.array([1, 0, 0, 1, 1])
        loss = rmse(y_pred, y_true)
        self.assertAlmostEqual(loss, 0.29325756597230357)

    def test_case2(self):
        y_pred = np.array([0.8, 0.3, 0.7, 0.5, 0.8, 0.6, 0.5, 0.2])
        y_true = np.array([1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
        loss = rmse(y_pred, y_true)
        self.assertAlmostEqual(loss, 0.3807886552931954)

