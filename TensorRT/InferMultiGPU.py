import pycuda.driver as cuda
import tensorrt as trt
import numpy as np
from ctypes import cdll, c_char_p
libcudart = cdll.LoadLibrary('libcudart.so')
libcudart.cudaGetErrorString.restype = c_char_p



TRT_LOGGER = trt.Logger(min_severity =trt.ILogger.INTERNAL_ERROR)


class GPU_infer():

    def __init__(self,engine_file_path,batch_size,data_type,gpu_id):
        cuda.init()
        self.ctx = cuda.Device(gpu_id).make_context() 
        self.batch_size = batch_size

        with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.h_input = cuda.pagelocked_empty(1 * trt.volume(self.engine .get_binding_shape(0)), dtype=trt.nptype(data_type))
        self.h_output = cuda.pagelocked_empty(1 * trt.volume(self.engine .get_binding_shape(1)), dtype=trt.nptype(data_type))
        self.d_input = cuda.mem_alloc(self.h_input.nbytes)
        self.d_output = cuda.mem_alloc(self.h_output.nbytes)
        self.stream = cuda.Stream()



    # def cudaSetDevice(device_idx):
    #     ret = libcudart.cudaSetDevice(device_idx)
    #     if ret != 0:
    #         error_string = libcudart.cudaGetErrorString(ret)
    #         raise RuntimeError("cudaSetDevice: " + error_string)

    def run_inf(self,pics):
        self.ctx.push()
        preprocessed = np.asarray(pics).ravel()
        np.copyto(self.h_input, preprocessed) 
        with self.engine.create_execution_context() as context:
            # input_shape = (self.batch_size ,3,256,256)
            # context.set_binding_shape(0, input_shape)
            cuda.memcpy_htod(self.d_input, self.h_input)
            context.execute_v2(bindings=[int(self.d_input), int(self.d_output)])
            cuda.memcpy_dtoh(self.h_output, self.d_output)
            out = self.h_output.reshape((self.batch_size,10, 164, 164))
            return out 
    def clean(self):
        # self.ctx.pop()
        self.ctx.synchronize()
        self.ctx.detach()
        # del self.ctx


