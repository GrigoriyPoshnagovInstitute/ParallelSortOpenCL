#include <CL/cl2.hpp>
#include <algorithm>
#include <chrono>
#include <execution>
#include <fstream>
#include <iostream>
#include <random>
#include <vector>

static std::string LoadKernelSource(const std::string &filename)
{
	std::ifstream file(filename);
	if (!file.is_open())
	{
		throw std::runtime_error("Cannot open kernel file");
	}
	return std::string(std::istreambuf_iterator(file),
	                   std::istreambuf_iterator<char>());
}

void SortArrayOnCpu(std::vector<int> &inputData, double &cpuTimeMs)
{
	auto startTime = std::chrono::high_resolution_clock::now();
	std::sort(std::execution::par,  inputData.begin(),  inputData.end());
	auto endTime = std::chrono::high_resolution_clock::now();
	cpuTimeMs = std::chrono::duration<double, std::milli>(endTime - startTime).count();
}

void SortArrayOnGpu(cl::Context &context,
                    cl::CommandQueue &queue,
                    cl::Program &program,
                    std::vector<int> &inputData,
                    double &gpuTimeMs)
{
	const size_t arraySize =  inputData.size();
	cl::Kernel kernel(program, "BitonicSort");
	auto gpuStart = std::chrono::high_resolution_clock::now();
	cl::Buffer  deviceBuffer(context,
	                          CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
	                          sizeof(int) * arraySize,
	                           inputData.data());
	// будет ли ускорение если мы будем асинхронно сортировать
	queue.finish();
	queue.flush();


	for (unsigned int stage = 1; stage <= static_cast<unsigned>(std::log2(arraySize)); ++stage)
	{
		for (unsigned int pass = 1; pass <= stage; ++pass)
		{
			kernel.setArg(0,  deviceBuffer);
			kernel.setArg(1, stage);
			kernel.setArg(2, pass);
			kernel.setArg(3, static_cast<unsigned>(arraySize));
			// возможо ли ускорить за счет исползования локальной памяти (разделить на ворк группы?)
			queue.enqueueNDRangeKernel(
				kernel,
				cl::NullRange,
				cl::NDRange(arraySize / 2),
				cl::NullRange);
		}
	}

	queue.finish();
	auto gpuEnd = std::chrono::high_resolution_clock::now();

	queue.enqueueReadBuffer(deviceBuffer,
	                        CL_TRUE,
	                        0,
	                        sizeof(int) * arraySize,
	                         inputData.data());

	gpuTimeMs = std::chrono::duration<double, std::milli>(gpuEnd - gpuStart).count();
}

int main()
{
	system("chcp 65001");
	try
	{
		const size_t arraySize = 1 << 27; // 1M элементов
		std::vector<int>  inputData(arraySize);
		std::mt19937 rng(12345);
		std::uniform_int_distribution dist(0, 1000000);
		for (auto &value:  inputData)
		{
			value = dist(rng);
		}
		auto  cpuData =  inputData;

		std::vector<cl::Platform> platforms;
		cl::Platform::get(&platforms);
		if (platforms.empty())
		{
			throw std::runtime_error("No OpenCL platforms found");
		}
		cl::Platform platform = platforms.front();

		std::vector<cl::Device> devices;
		platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
		if (devices.empty())
		{
			throw std::runtime_error("No GPU devices found");
		}
		cl::Device device = devices.front();

		cl::Context context(device);
		cl_command_queue_properties props = CL_QUEUE_PROFILING_ENABLE;
		cl::CommandQueue queue(context, device, props);

		std::string kernelSource = LoadKernelSource("BitonicSort.cl");
		cl::Program::Sources sources;
		sources.push_back({kernelSource.c_str(), kernelSource.length()});
		cl::Program program(context, sources);
		program.build({device});

		cl_int buildStatus = program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(device);
		if (buildStatus != CL_SUCCESS)
		{
			throw std::runtime_error("Kernel build failed");
		}

		double gpuTimeMs = 0.0;
		SortArrayOnGpu(context, queue, program,  inputData, gpuTimeMs);

		double cpuTimeMs = 0.0;
		SortArrayOnCpu(cpuData, cpuTimeMs);

		if (inputData !=  cpuData)
		{
			std::cerr << "Ошибка: результаты сортировки не совпадают!\n";

			return 1;
		}

		std::cout << "GPU Sort time: " << gpuTimeMs << " ms\n";
		std::cout << "CPU Sort time: " << cpuTimeMs << " ms\n";
	}
	catch (const std::exception &ex)
	{
		std::cerr << "Exception: " << ex.what() << "\n";
		return 1;
	}

	return 0;
}
