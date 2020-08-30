import unittest,math
from datatype import DataTypes
import numpy as np
from scipy import stats
class TDD_GETTING_STARTED(unittest.TestCase):
    def test_mse(self):
        a=[1,2,3]
        b=[4,5,6]
        self.assertRaises(TypeError,DataTypes({'a':a}).getMSE,b)

    def test_datatypes(self):
        d=DataTypes(5)
        self.assertTrue(d.Numerical())
        self.assertTrue(d.Discrete())
        self.assertFalse(d.Continuous())
        d=DataTypes(5.)
        self.assertTrue(d.Numerical())
        self.assertFalse(d.Discrete())
        self.assertTrue(d.Continuous())
        d=DataTypes({"speed": [99,86,87,88,111,86,103,87,94,78,77,85,86]})
        d1=DataTypes({"speed": [99,86,87,88,86,103,87,94,78,77,85,86]})
        m=d.getMean()
        self.assertAlmostEqual(m, 89.77, 1)
        median = d.getMedian()
        median1 = d1.getMedian()
        self.assertEqual(median,87)
        self.assertEqual(median1, 86.5)
        mode = d.getMode()
        # print(mode)
        self.assertEqual(mode[0],86)
        self.assertEqual(mode.mode,86)
        self.assertEqual(mode[1],3)
        self.assertEqual(mode.count, 3)
    def test_standard_deviation(self):
        d = DataTypes({'speed': [86, 87, 88, 86, 87, 85, 86]})
        d1 = DataTypes({'speed': [32,111,138,28,59,77,97]})
        s = d.getStd()
        s1 = d1.getStd()
        self.assertAlmostEqual(s,.9,2)
        self.assertAlmostEqual(s1, 37.85, 2)
        v=d1.getVariance()
        self.assertAlmostEqual(v,1432.2,1)
        # the formula to find the standard deviation is the square root of the variance:
        self.assertEqual(s1,math.sqrt(v))
        self.assertEqual(s1 ** 2, (v))
if __name__ == '__main__':
    unittest.main()
