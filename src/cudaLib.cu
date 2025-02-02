
#include "cudaLib.cuh"

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort)
{
	if (code != cudaSuccess) 
	{
		fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

__global__ 
void saxpy_gpu (float* x, float* y, float scale, int size) {
	//	Insert GPU SAXPY kernel code here
	int tid = (blockIdx.x * blockDim.x) + threadIdx.x; //Calculate thread ID
	
	if(tid < size){
		y[tid] = scale * x[tid] + y[tid]; //SAX
	}
}

int runGpuSaxpy(int vectorSize) {

	std::cout << "Hello GPU Saxpy!\n";
	auto tStart = std::chrono::high_resolution_clock::now();
	//	Insert code here
	size_t v_size = vectorSize * sizeof(float); //Memory Size of Vector

	float *cpu_x, *cpu_y, *cpu_z, *cpu_solve; //Host side vectors
	float *gpu_x, *gpu_z; //Device side vectors

	cpu_x = (float*)malloc(v_size); //Malloc host side vectors
	cpu_y = (float*)malloc(v_size);
	cpu_z = (float*)malloc(v_size);
	cpu_solve = (float*)malloc(v_size);

	cudaMalloc(&gpu_x, v_size); //cudaMalloc device side vectors
	cudaMalloc(&gpu_z, v_size);

	for(int i = 0; i < vectorSize; i++){ //Generate Random Matrix Values
		cpu_x[i] = (float)(rand() % 100);
		cpu_y[i] = (float)(rand() % 100);
		cpu_z[i] = cpu_y[i];
		cpu_solve[i] = cpu_y[i];
	}

	// float scale = (float)(rand() % 100); //Generate scale value
	// float scale = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
	float scale = 1.7f;
	std::cout << "Scale: " << scale << "\n";
	

	cudaMemcpy(gpu_x, cpu_x, v_size, cudaMemcpyHostToDevice);//Copy memory block to Device
	cudaMemcpy(gpu_z, cpu_z, v_size, cudaMemcpyHostToDevice);

	int tb_size = 256; //Thread block size
	int num_blocks = ceil(float(vectorSize) / tb_size); //Make sure we round up partial thread blocks

	saxpy_gpu<<<num_blocks, tb_size>>>(gpu_x, gpu_z, scale, vectorSize);//Kernel launch

	cudaMemcpy(cpu_z, gpu_z, v_size, cudaMemcpyDeviceToHost);//Copy result back to Host	

	
	// std::cout << "Lazy, you are!\n";
	// std::cout << "Write code, you must\n";
	saxpy_cpu(cpu_x, cpu_solve, scale, vectorSize); // Run saxpy on CPU to compare with GPU code
	
	int errorCount = 0;
	for(int i = 0; i < vectorSize; i++){
		if(std::abs(cpu_z[i] - cpu_solve[i]) > 0.0001f){ //compare GPU result (cpu_z) with CPU result (cpu_solve)
			errorCount++;
		}
	}
	std::cout << "Found " << errorCount << " / " << vectorSize << " errors \n";

	free(cpu_x); // Free everything we malloced
	free(cpu_y);
	free(cpu_z);
	free(cpu_solve);
	cudaFree(gpu_x); // cudaFree device memory
	cudaFree(gpu_z);
	auto tEnd= std::chrono::high_resolution_clock::now();

	std::chrono::duration<double> time_span = (tEnd- tStart);
	std::cout << "It took " << time_span.count() << " seconds.";
	return 0;
}

/* 
 Some helpful definitions

 generateThreadCount is the number of threads spawned initially. Each thread is responsible for sampleSize points. 
 *pSums is a pointer to an array that holds the number of 'hit' points for each thread. The length of this array is pSumSize.

 reduceThreadCount is the number of threads used to reduce the partial sums.
 *totals is a pointer to an array that holds reduced values.
 reduceSize is the number of partial sums that each reduceThreadCount reduces.

*/

__global__
void generatePoints (uint64_t * pSums, uint64_t pSumSize, uint64_t sampleSize) {
	//	Insert code here
	int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

	if(tid < pSumSize){ // thread boundary check
		curandState_t rng;
		curand_init(clock64(), tid, 0, &rng); // set curand seed
		uint64_t hit_count = 0;

		for(uint64_t i = 0; i < sampleSize; i++){ //generate sampleSize coordinate pairs
			double x = curand_uniform(&rng); // [0, 1)
			double y = curand_uniform(&rng);
			if ( int(x * x + y * y) == 0 ){ //unit circle check (taken from cpulib.cpp file)
				hit_count++;
			}
		}
		pSums[tid] = hit_count; //store partial sum in array
	}
}

__global__ 
void reduceCounts (uint64_t * pSums, uint64_t * totals, uint64_t pSumSize, uint64_t reduceSize) {
	//	Insert code here
	int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
	if(tid < pSumSize / reduceSize){ //boundary check	
		uint64_t partial = 0;
		if(tid * reduceSize < pSumSize){ //boundary check for pSum array index
			for(uint64_t i = tid * reduceSize; i < tid * reduceSize + reduceSize; i++){ //sum inidicies
				partial += pSums[i];
			}
		}
		totals[tid] = partial; //store result in reduced array
	}
}

int runGpuMCPi (uint64_t generateThreadCount, uint64_t sampleSize, 
	uint64_t reduceThreadCount, uint64_t reduceSize) {

	//  Check CUDA device presence
	int numDev;
	cudaGetDeviceCount(&numDev);
	if (numDev < 1) {
		std::cout << "CUDA device missing!\n";
		return -1;
	}

	auto tStart = std::chrono::high_resolution_clock::now();
		
	float approxPi = estimatePi(generateThreadCount, sampleSize, 
		reduceThreadCount, reduceSize);
	
	std::cout << "Estimated Pi = " << approxPi << "\n";

	auto tEnd= std::chrono::high_resolution_clock::now();

	std::chrono::duration<double> time_span = (tEnd- tStart);
	std::cout << "It took " << time_span.count() << " seconds.";

	return 0;
}

double estimatePi(uint64_t generateThreadCount, uint64_t sampleSize, 
	uint64_t reduceThreadCount, uint64_t reduceSize) {
	
	double approxPi = 0;

	//      Insert code here
	uint64_t *cpu_pSums, *gpu_pSums; // host side vectors
	uint64_t *cpu_totals, *gpu_totals; // device side vectors
	
	size_t pSumMem = sizeof(uint64_t) * generateThreadCount; //memory size of pSums
	size_t reduceSizeMem = sizeof(uint64_t) * reduceThreadCount; //memory size of reduced pSums
	
	cpu_pSums = (uint64_t*)malloc(pSumMem); //malloc host side
	cpu_totals = (uint64_t*)malloc(reduceSizeMem);
	
	cudaMalloc(&gpu_pSums, pSumMem); //cudaMalloc device side
	cudaMalloc(&gpu_totals, reduceSizeMem);

	cudaMemcpy(gpu_pSums, cpu_pSums, pSumMem, cudaMemcpyHostToDevice); //copy pSums to device memory

	uint64_t tb_size = 256; //tb size
	uint64_t num_blocks = ceil(float(generateThreadCount) / tb_size);
	uint64_t num_blocks_reduce = ceil(float(reduceThreadCount) / tb_size);
	generatePoints<<<num_blocks, tb_size>>>(gpu_pSums, generateThreadCount, sampleSize); //kernel launch
	cudaDeviceSynchronize();
	reduceCounts<<<num_blocks_reduce, tb_size>>>(gpu_pSums, gpu_totals, generateThreadCount, reduceSize);

	// cudaMemcpy(cpu_pSums, gpu_pSums, pSumMem, cudaMemcpyDeviceToHost); //copy out unreduced pSums
	cudaMemcpy(cpu_totals, gpu_totals, reduceSizeMem, cudaMemcpyDeviceToHost); //copy out reduced pSums from device

	uint64_t totalHitCount = 0;
	uint64_t totalSampleCount = generateThreadCount * sampleSize;
	// for(uint64_t i = 0; i < generateThreadCount; i++) //used for calcing unreduced pSums
	// {
	// 	totalHitCount += cpu_pSums[i];
	// }
	for(int i = 0; i < reduceThreadCount; i++){ //sum reduced pSums
		totalHitCount += cpu_totals[i];
	}
	
	approxPi = (double)totalHitCount / totalSampleCount; //estimate pi similar to cpuLib.cpp
	approxPi = approxPi * 4.0f;

	free(cpu_pSums); //free host memory
	free(cpu_totals);
	cudaFree(gpu_totals); //free device memory
	cudaFree(gpu_pSums);
	// std::cout << "Sneaky, you are ...\n";
	// std::cout << "Compute pi, you must!\n";
	return approxPi;
}
