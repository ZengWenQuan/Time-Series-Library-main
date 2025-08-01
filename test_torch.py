import torch
import os
def test_torch_and_cuda():
    # 检查PyTorch是否正确安装
    print("PyTorch版本:", torch.__version__)
    
    # 检查CUDA是否可用
    cuda_available = torch.cuda.is_available()
    print("CUDA是否可用:", cuda_available)
    
    if cuda_available:
        # 显示CUDA版本
        print("CUDA版本:", torch.version.cuda)
        
        # 显示可用的GPU数量和信息
        gpu_count = torch.cuda.device_count()
        print("可用GPU数量:", gpu_count)
        
        for i in range(gpu_count):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            
        # 测试GPU内存分配
        try:
            test_tensor = torch.tensor([1.0]).cuda()
            print("GPU内存分配测试成功")
        except Exception as e:
            print("GPU内存分配测试失败:", str(e))
    else:
        print("警告: CUDA不可用，PyTorch将使用CPU运行")

    print('TORCH_CUDA_ARCH_LIST:',os.environ.get('TORCH_CUDA_ARCH_LIST', None))
    #print(os.environ.get('TORCH_CUDA_ARCH_LIST'))
    print('支持的计算框架',torch.cuda.get_device_capability())
if __name__ == "__main__":
    test_torch_and_cuda() 