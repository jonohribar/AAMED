# distutils: language = c++
# distutils: sources = pyAAMED.cpp

import numpy as np
cimport numpy as np
import cv2

cdef extern from "../src/FLED.h":
    cdef cppclass FLED:
        FLED(int, int)
        int run_FLED(unsigned char*, int, int)
        # takes in Mat Img_G, vector<Point> edgePoints
        int run_AAMED_WithoutImage(unsigned char*, int, int, unsigned char*, int)
        void release()
        void drawAAMED(unsigned char*, int, int)
        void SetParameters(double, double, double)
        void UpdateResults(float*)


cdef class pyAAMED:
    cdef FLED* _fled;
    cdef int drows;
    cdef int dcols;
    def __cinit__(self, int drows, int dcols):
        self._fled = new FLED(drows, dcols)
        self.drows = drows
        self.dcols = dcols

    def run_AAMED(self, np.ndarray[np.uint8_t, ndim=2] imgG):
        cdef int rows = imgG.shape[0]
        cdef int cols = imgG.shape[1]
        assert rows < self.drows and cols < self.dcols, \
            'The size ({:d}, {:d}) of an input image must be smaller than ({:d}, {:d})'.format(rows, cols, self.drows, self.dcols)
        cdef int det_num = 0
        det_num = self._fled.run_FLED(&imgG[0, 0], rows, cols)
        if det_num == 0:
            return []
        cdef np.ndarray[np.float32_t, ndim=2] detEllipse = np.zeros(shape=(det_num, 6), dtype=np.float32)
        self._fled.UpdateResults(&detEllipse[0, 0])
        return detEllipse

    def run_AMMED_imageless(self, np.ndarray[np.uint8_t, ndim=2] imgG, np.ndarray[np.uint8_t, ndim=2] points):
        cdef int rows = imgG.shape[0]
        cdef int cols = imgG.shape[1]
        cdef int numpoints = int(points.shape[0])
        assert rows < self.drows and cols < self.dcols, \
            'The size ({:d}, {:d}) of an input image must be smaller than ({:d}, {:d})'.format(rows, cols, self.drows, self.dcols)
        cdef int det_num = 0
        det_num = self._fled.run_AAMED_WithoutImage(&imgG[0, 0], rows, cols, &points[0, 0], numpoints)
        if det_num == 0:
            return []
        cdef np.ndarray[np.float32_t, ndim=2] detEllipse = np.zeros(shape=(det_num, 6), dtype=np.float32)
        self._fled.UpdateResults(&detEllipse[0, 0])
        return detEllipse





    def release(self):
        #self._fled.release()
        del self._fled

    def drawAAMED(self, np.ndarray[np.uint8_t, ndim=2] imgG):
        cdef int rows = imgG.shape[0]
        cdef int cols = imgG.shape[1]
        self._fled.drawAAMED(&imgG[0, 0], rows, cols)

    def setParameters(self, double theta_fsa, double length_fsa, double T_val):
        self._fled.SetParameters(theta_fsa, length_fsa, T_val)
