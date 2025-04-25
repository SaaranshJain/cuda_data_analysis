#include <iostream>
#include <fstream>
#include <unordered_map>
#include <vector>
#include <sstream>

// CUDA runtime includes
#include <cuda_runtime.h>

// Thrust includes
#include <thrust/sort.h>
#include <thrust/device_vector.h>

// Define constants
#define ITEMS_PER_THREAD 4

__global__ void calc_avg(int *ids, double *rating_sums, int *counts, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        rating_sums[idx] = rating_sums[idx] / counts[idx];
    }
}

__global__ void calc_avg_optimized(int *ids, double *rating_sums, int *counts, int n)
{
    // Keep each thread more busy than before, make it average 4 elements instead of one
    // 4 is an arbitrary choice here
    // Process multiple elements per thread
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    const int items_per_thread = 4;
    for (int i = 0; i < items_per_thread; i++)
    {
        int current_idx = idx * items_per_thread + i;
        if (current_idx < n)
        {
            rating_sums[current_idx] = rating_sums[current_idx] / counts[current_idx];
        }
    }
}

__global__ void calc_avg_coalesced_multi(int *ids, double *rating_sums, int *counts, int n)
{
    // Coalesced access refers to threads of the same block accessing contiguous memory locations
    // Shared memory for coalesced access
    extern __shared__ char shared_mem[];
    double *s_sums = (double *)shared_mem;
    int *s_counts = (int *)&s_sums[blockDim.x * ITEMS_PER_THREAD];

    int tid = threadIdx.x;
    int base_idx = blockIdx.x * blockDim.x * ITEMS_PER_THREAD + tid;

// Each thread loads multiple items in a coalesced manner
#pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; i++)
    {
        int idx = base_idx + i * blockDim.x;
        if (idx < n)
        {
            // Coalesced global memory read
            s_sums[tid + i * blockDim.x] = rating_sums[idx];
            s_counts[tid + i * blockDim.x] = counts[idx];
        }
    }

    __syncthreads();

// Process multiple items per thread
#pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; i++)
    {
        int idx = base_idx + i * blockDim.x;
        if (idx < n)
        {
            // Compute in shared memory
            s_sums[tid + i * blockDim.x] = s_sums[tid + i * blockDim.x] / s_counts[tid + i * blockDim.x];
        }
    }

    __syncthreads();

// Write back to global memory in a coalesced manner
#pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; i++)
    {
        int idx = base_idx + i * blockDim.x;
        if (idx < n)
        {
            // Coalesced global memory write
            rating_sums[idx] = s_sums[tid + i * blockDim.x];
        }
    }
}

// Optimized kernel using shared memory for coalesced access
__global__ void calc_avg_coalesced(int *ids, double *rating_sums, int *counts, int n)
{
    extern __shared__ char shared_mem[];
    double *s_sums = (double *)shared_mem;
    int *s_counts = (int *)&s_sums[blockDim.x];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Coalesced load from global memory
    if (idx < n)
    {
        s_sums[threadIdx.x] = rating_sums[idx];
        s_counts[threadIdx.x] = counts[idx];
    }

    __syncthreads();

    // Perform computation in shared memory
    if (idx < n)
    {
        s_sums[threadIdx.x] = s_sums[threadIdx.x] / s_counts[threadIdx.x];
    }

    __syncthreads();

    // Coalesced write back to global memory
    if (idx < n)
    {
        rating_sums[idx] = s_sums[threadIdx.x];
    }
}

std::vector<std::string> split(std::string &str, char delimiter)
{
    std::stringstream ss(str);
    std::string item;
    std::vector<std::string> result;

    while (std::getline(ss, item, delimiter))
    {
        result.push_back(item);
    }

    return result;
}

int main()
{
    // Read json file on cpu itself
    std::ifstream input_file("./datasets/e5_product_and_rating.csv");
    if (!input_file.is_open())
    {
        std::cerr << "Error opening file!" << std::endl;
        return -1;
    }
    // An int is big enough to store 5 * 400M, so no issue using it for rating sum
    std::unordered_map<std::string, std::pair<double, int>> id_to_stats;

    std::string line;
    std::vector<std::string> parts;
    // Read the file line by line
    while (std::getline(input_file, line))
    {
        try
        {
            parts = split(line, ';');
            std::string asin = parts[0];
            double rating = std::stof(parts[1]);
            auto &pair = id_to_stats[asin];
            pair.first += rating;
            pair.second += 1;
        }
        catch (const std::exception &e)
        {
            std::cerr << "Error processing line: " << e.what() << std::endl;
        }
    }

    std::unordered_map<std::string, int> asin_to_int;
    std::unordered_map<int, std::string> int_to_asin;
    std::vector<int> ids;
    std::vector<double> rating_sums;
    std::vector<int> rating_counts;

    int current_id = 0;
    for (const auto &pair : id_to_stats)
    {
        auto asin = pair.first;
        auto stats = pair.second;
        if (asin_to_int.find(asin) == asin_to_int.end())
        {
            asin_to_int[asin] = current_id++;
            int_to_asin[current_id - 1] = asin;
        }
        ids.push_back(asin_to_int[asin]);
        rating_sums.push_back(stats.first);
        rating_counts.push_back(stats.second);
    }

    id_to_stats.clear();
    // GPU data items
    int *gpu_ids;
    double *gpu_rating_sums;
    int *gpu_rating_counts;

    int n = ids.size();

    cudaEvent_t start, stop;
    float milliseconds = 0;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    // allocate space for ids, sums and counts
    cudaMalloc(&gpu_ids, n * sizeof(int));
    cudaMalloc(&gpu_rating_sums, n * sizeof(double)); // Re-use this for storing averages as well(no point allocating extra mem.)
    cudaMalloc(&gpu_rating_counts, n * sizeof(int));

    // Move to GPU
    cudaMemcpy(gpu_ids, ids.data(), n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_rating_sums, rating_sums.data(), n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_rating_counts, rating_counts.data(), n * sizeof(int), cudaMemcpyHostToDevice);

    // Before kernel launch
    int blockSize;
    int minGridSize;

    // Get optimal block size for calc_avg kernel
    cudaOccupancyMaxPotentialBlockSize(
        &minGridSize,
        &blockSize,
        calc_avg,
        0, // No dynamic shared memory
        0  // No block size limit
    );

    // Calculate grid size based on input size
    int gridSize = (n + blockSize - 1) / blockSize;

    // Calculate shared memory size
    size_t sharedMemSize = blockSize * ITEMS_PER_THREAD * (sizeof(double) + sizeof(int));

    // Launch optimized kernel
    calc_avg_coalesced_multi<<<gridSize, blockSize, sharedMemSize>>>(
        gpu_ids, gpu_rating_sums, gpu_rating_counts, n);

    cudaFree(gpu_rating_counts); // No need of these after calculating average ratings
    // Error check
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }

    // Use thrust::device_vector instead of devicePtr, since devicePtr is not supported by earlier cuda version
    thrust::device_vector<int> d_ids(gpu_ids, gpu_ids + n);
    thrust::device_vector<double> d_ratings(gpu_rating_sums, gpu_rating_sums + n);

    // Number of top rated products
    // int k = 10000;
    int k = 10;

    // sorting in descending order
    // Thrust optimizes sorting, figuring out block size and number of threads on its own
    thrust::sort_by_key(d_ratings.begin(), d_ratings.end(), d_ids.begin(), thrust::greater<double>());

    std::vector<int> top_ids(n);
    std::vector<double> top_ratings(n);

    // Copy from device to host
    // cudaMemcpy(top_k_ids, d_ids ,k * sizeof(int) , cudaMemcpyDeviceToHost);
    // cudaMemcpy(top_k_ratings, d_ratings ,k * sizeof(double) , cudaMemcpyDeviceToHost);
    thrust::copy(d_ids.begin(), d_ids.end(), top_ids.begin());
    thrust::copy(d_ratings.begin(), d_ratings.end(), top_ratings.begin());

    cudaFree(gpu_ids);
    cudaFree(gpu_rating_sums);

    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Execution time: %f ms\n", milliseconds);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Print to verify
    for (int i = 0; i < k; i++)
    {
        std::cout << "Top id: " << i << " is: " << int_to_asin[top_ids[i]] << " with rating: " << top_ratings[i] << std::endl;
    }
    // Print to verify
    for (int i = n - 1; i > n - k; i--)
    {
        std::cout << "Bottom id: " << n - i << " is: " << int_to_asin[top_ids[i]] << " with rating: " << top_ratings[i] << std::endl;
    }
    return 0;
}