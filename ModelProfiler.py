import torch
import torch.nn as nn
import time
import psutil
import os
from torch.profiler import profile, record_function, ProfilerActivity
import numpy as np
from collections import OrderedDict
from model import MyModule, TokenCodeBERT
from parse import HTN_single_layer_parse, BERT_parse
from CWEutils import CWEfile_parse, CWEBalancefile_parse, MyCWEDataset
from torch.utils.data import DataLoader


class ModelProfiler:
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        模型性能分析器

        Args:
            model: PyTorch模型
            device: 运行设备 ('cuda' or 'cpu')
        """
        self.model = model.to(device)
        self.device = device
        self.model.eval()

    def count_parameters(self):
        """统计模型参数量"""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        non_trainable_params = total_params - trainable_params

        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'non_trainable_parameters': non_trainable_params,
            'total_parameters_M': total_params / 1e6,  # 百万参数
            'model_size_MB': total_params * 4 / (1024 ** 2)  # 假设float32，每个参数4字节
        }

    def get_model_size(self):
        """获取模型大小（字节）"""
        param_size = 0
        buffer_size = 0

        for param in self.model.parameters():
            param_size += param.nelement() * param.element_size()

        for buffer in self.model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()

        model_size = param_size + buffer_size
        return {
            'model_size_bytes': model_size,
            'model_size_MB': model_size / (1024 ** 2),
            'param_size_MB': param_size / (1024 ** 2),
            'buffer_size_MB': buffer_size / (1024 ** 2)
        }

    def measure_inference_time(self, input_tensor, retraction=None, num_runs=100, warmup_runs=10):
        """
                测量推理时间

                Args:
                    input_tensor: 输入张量
                    retraction:缩进张量
                    num_runs: 测试运行次数
                    warmup_runs: 预热运行次数
                """
        input_tensor = input_tensor.to(self.device)

        # GPU预热
        if self.device == 'cuda':
            for _ in range(warmup_runs):
                with torch.no_grad():
                    if retraction is not None:
                        _ = self.model(input_tensor, retraction)
                    else:
                        _ = self.model(input_tensor)
            torch.cuda.synchronize()

        # 测量推理时间
        times = []

        with torch.no_grad():
            for _ in range(num_runs):
                if self.device == 'cuda':
                    torch.cuda.synchronize()
                    start_time = time.time()
                    if retraction is not None:
                        _ = self.model(input_tensor, retraction)
                    else:
                        _ = self.model(input_tensor)
                    torch.cuda.synchronize()
                    end_time = time.time()
                else:
                    start_time = time.time()
                    if retraction is not None:
                        _ = self.model(input_tensor, retraction)
                    else:
                        _ = self.model(input_tensor)
                    end_time = time.time()

                times.append(end_time - start_time)

        return {
            'avg_inference_time': np.mean(times),
            'std_inference_time': np.std(times),
            'min_inference_time': np.min(times),
            'max_inference_time': np.max(times),
            'fps': 1.0 / np.mean(times)  # 每秒帧数
        }

    def measure_memory_usage(self, input_tensor, retraction=None):
        """测量内存使用情况"""
        input_tensor = input_tensor.to(self.device)

        if self.device == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            # 测量推理前的内存
            memory_before = torch.cuda.memory_allocated()

            with torch.no_grad():
                if retraction is not None:
                    _ = self.model(input_tensor, retraction)
                else:
                    _ = self.model(input_tensor)

            # 测量推理后的内存
            memory_after = torch.cuda.memory_allocated()
            peak_memory = torch.cuda.max_memory_allocated()

            return {
                'gpu_memory_before_MB': memory_before / (1024 ** 2),
                'gpu_memory_after_MB': memory_after / (1024 ** 2),
                'gpu_peak_memory_MB': peak_memory / (1024 ** 2),
                'gpu_memory_used_MB': (memory_after - memory_before) / (1024 ** 2)
            }
        else:
            # CPU内存使用情况
            process = psutil.Process(os.getpid())
            memory_before = process.memory_info().rss / (1024 ** 2)

            with torch.no_grad():
                if retraction is not None:
                    _ = self.model(input_tensor, retraction)
                else:
                    _ = self.model(input_tensor)

            memory_after = process.memory_info().rss / (1024 ** 2)

            return {
                'cpu_memory_before_MB': memory_before,
                'cpu_memory_after_MB': memory_after,
                'cpu_memory_used_MB': memory_after - memory_before
            }

    def detailed_profiling(self, input_tensor, retraction=None, num_runs=10):
        """详细的性能分析"""
        input_tensor = input_tensor.to(self.device)

        activities = [ProfilerActivity.CPU]
        if self.device == 'cuda':
            activities.append(ProfilerActivity.CUDA)

        with profile(
                activities=activities,
                record_shapes=True,
                profile_memory=True,
                with_stack=True
        ) as prof:
            with record_function("model_inference"):
                for _ in range(num_runs):
                    with torch.no_grad():
                        if retraction is not None:
                            _ = self.model(input_tensor, retraction)
                        else:
                            _ = self.model(input_tensor)

        return prof

    def get_flops_estimate(self, input_tensor, retraction=None):
        """估算FLOPs（需要安装thop库: pip install thop）"""
        try:
            from thop import profile
            input_tensor = input_tensor.to(self.device)
            if retraction is not None:
                flops, params = profile(self.model, inputs=(input_tensor, retraction), verbose=False)
            else:
                flops, params = profile(self.model, inputs=(input_tensor,), verbose=False)
            return {
                'flops': flops,
                'flops_G': flops / 1e9,  # GFLOPs
                'params_from_thop': params
            }
        except ImportError:
            print("Warning: thop library not installed. Cannot calculate FLOPs.")
            return None

    def comprehensive_benchmark(self, input_tensor, retraction=None, num_runs=100):
        """
        综合性能测试

        Args:
            input_tensor: 张量
            num_runs: 运行次数
        """
        print("=" * 60)
        print("PyTorch Model Performance Benchmark")
        print("=" * 60)

        # # 创建测试输入
        # full_input_shape = (batch_size,) + input_shape
        # input_tensor = torch.randn(full_input_shape)

        print(f"Model: {self.model.__class__.__name__}")
        print(f"Device: {self.device}")
        # print(f"Input shape: {full_input_shape}")
        # print(f"Batch size: {batch_size}")
        print("-" * 60)

        # 1. 参数统计
        param_stats = self.count_parameters()
        print("Parameter Statistics:")
        print(f"  Total parameters: {param_stats['total_parameters']:,}")
        print(f"  Trainable parameters: {param_stats['trainable_parameters']:,}")
        print(f"  Non-trainable parameters: {param_stats['non_trainable_parameters']:,}")
        print(f"  Model size: {param_stats['model_size_MB']:.2f} MB")
        print()

        # 2. 模型大小
        size_stats = self.get_model_size()
        print("Model Size:")
        print(f"  Total size: {size_stats['model_size_MB']:.2f} MB")
        print(f"  Parameters: {size_stats['param_size_MB']:.2f} MB")
        print(f"  Buffers: {size_stats['buffer_size_MB']:.2f} MB")
        print()

        # 3. 推理时间
        time_stats = self.measure_inference_time(input_tensor, retraction, num_runs)
        print("Inference Time:")
        print(f"  Average: {time_stats['avg_inference_time'] * 1000:.2f} ms")
        print(f"  Std: {time_stats['std_inference_time'] * 1000:.2f} ms")
        print(f"  Min: {time_stats['min_inference_time'] * 1000:.2f} ms")
        print(f"  Max: {time_stats['max_inference_time'] * 1000:.2f} ms")
        print(f"  FPS: {time_stats['fps']:.2f}")
        print()

        # 4. 内存使用
        memory_stats = self.measure_memory_usage(input_tensor, retraction)
        print("Memory Usage:")
        for key, value in memory_stats.items():
            print(f"  {key}: {value:.2f} MB")
        print()

        # 5. FLOPs估算
        flops_stats = self.get_flops_estimate(input_tensor,retraction)
        if flops_stats:
            print("FLOPs:")
            print(f"  Total FLOPs: {flops_stats['flops']:,}")
            print(f"  GFLOPs: {flops_stats['flops_G']:.2f}")
            print()

        # # 6. 吞吐量测试（不同batch size）
        # print("Throughput Analysis:")
        # batch_sizes = [1, 4, 8, 16, 32] if self.device == 'cuda' else [1, 2, 4, 8]
        #
        # for bs in batch_sizes:
        #     try:
        #         test_input = torch.randn((bs,) + input_shape)
        #         bs_time_stats = self.measure_inference_time(test_input, num_runs=50)
        #         throughput = bs / bs_time_stats['avg_inference_time']
        #         print(f"  Batch size {bs:2d}: {throughput:6.1f} samples/sec")
        #     except RuntimeError as e:
        #         print(f"  Batch size {bs:2d}: OOM or Error")
        #         break

        print("=" * 60)

        return {
            'parameters': param_stats,
            'model_size': size_stats,
            'inference_time': time_stats,
            'memory_usage': memory_stats,
        }


# 使用示例
def example_usage(model, input, retraction=None, runNum=100):
    # 创建分析器
    profiler = ModelProfiler(model)

    # 运行综合测试
    results = profiler.comprehensive_benchmark(
        input,
        retraction,
        num_runs=runNum
    )
    detailed_prof = profiler.detailed_profiling(input, retraction)

    # 打印详细分析结果
    print("\nDetailed Profiling Results:")
    print(detailed_prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))


def staticHTN():
    # 性能统计  待测试
    myModel_parse = HTN_single_layer_parse()
    myModel = MyModule(myModel_parse)
    path_parse = CWEBalancefile_parse()

    test_data2D = MyCWEDataset(path_parse.split_data_path + "//test.json", path_parse.dict_path, myModel_parse, dim=2,
                               need_clean=False)
    merged_test2D = DataLoader(dataset=test_data2D, batch_size=myModel_parse.batch_size, shuffle=True, drop_last=True)

    for batch in merged_test2D:
        data, tag, ret = batch
        if torch.cuda.is_available():
            data = data.cuda()
            # tag1 = tag1.cuda()
            tag = tag.cuda()
            ret = ret.cuda()
        example_usage(myModel, data, retraction=ret)
        # 处理第一个batch后跳出
        break


def staticBERT():
    # BERT
    bert_parse = BERT_parse()
    bert = TokenCodeBERT(bert_parse)
    path_parse = CWEBalancefile_parse()


    test_data1D = MyCWEDataset(path_parse.split_data_path + "//test.json", path_parse.dict_path, bert_parse, dim=1,
                               need_clean=False)
    merged_test1D = DataLoader(dataset=test_data1D, batch_size=bert_parse.batch_size, shuffle=True, drop_last=True)

    for batch in merged_test1D:
        data, tag = batch
        if torch.cuda.is_available():
            data = data.cuda()
            # tag1 = tag1.cuda()
            tag = tag.cuda()
        example_usage(bert, data)
        break


if __name__ == "__main__":
    print("---------------------HTN-------------------------")
    staticHTN()
    print("---------------------Transformer-------------------------")
    staticBERT()
