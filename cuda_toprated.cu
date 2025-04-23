#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda.h>
#include <unordered_map>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>

#define MAX_LINE_LEN 1024

using namespace std;

// CUDA kernel to compute average rating
__global__ void compute_averages(float *d_rating_sums, int *d_review_counts, float *d_avg_ratings, int asin_count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < asin_count && d_review_counts[idx] > 0) {
        d_avg_ratings[idx] = d_rating_sums[idx] / d_review_counts[idx];
    }
}

int main() {
    ifstream file("data/cleaned_ratings.csv");
    if (!file.is_open()) {
        cerr << "Failed to open file.\n";
        return 1;
    }

    unordered_map<string, int> asin_to_index;
    vector<string> asin_list;
    vector<float> rating_sums;
    vector<int> review_counts;

    string line;
    getline(file, line); // Skip header

    while (getline(file, line)) {
        stringstream ss(line);
        string rating_str, asin;

        getline(ss, rating_str, ',');
        getline(ss, asin, ',');

        if (rating_str.empty() || asin.empty()) continue;

        float rating = atof(rating_str.c_str());

        if (asin_to_index.find(asin) == asin_to_index.end()) {
            asin_to_index[asin] = asin_list.size();
            asin_list.push_back(asin);
            rating_sums.push_back(0.0f);
            review_counts.push_back(0);
        }

        int idx = asin_to_index[asin];
        rating_sums[idx] += rating;
        review_counts[idx] += 1;
    }

    file.close();

    int asin_count = asin_list.size();

    // Allocate GPU memory
    float *d_rating_sums, *d_avg_ratings;
    int *d_review_counts;

    cudaMalloc((void**)&d_rating_sums, asin_count * sizeof(float));
    cudaMalloc((void**)&d_review_counts, asin_count * sizeof(int));
    cudaMalloc((void**)&d_avg_ratings, asin_count * sizeof(float));

    cudaMemcpy(d_rating_sums, rating_sums.data(), asin_count * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_review_counts, review_counts.data(), asin_count * sizeof(int), cudaMemcpyHostToDevice);

    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (asin_count + threadsPerBlock - 1) / threadsPerBlock;
    compute_averages<<<blocksPerGrid, threadsPerBlock>>>(d_rating_sums, d_review_counts, d_avg_ratings, asin_count);

    cudaDeviceSynchronize();

    // Copy back
    vector<float> avg_ratings(asin_count);
    cudaMemcpy(avg_ratings.data(), d_avg_ratings, asin_count * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_rating_sums);
    cudaFree(d_review_counts);
    cudaFree(d_avg_ratings);

    // Sort and get top 10
    vector<int> indices(asin_count);
    for (int i = 0; i < asin_count; i++) indices[i] = i;

    sort(indices.begin(), indices.end(), [&](int a, int b) {
        return avg_ratings[a] > avg_ratings[b];
    });

    // Write results
    ofstream out("top_10_rated.csv");
    out << "ASIN,Avg Rating,Review Count\n";
    for (int i = 0, count = 0; i < asin_count && count < 10; i++) {
        int idx = indices[i];
        if (review_counts[idx] == 0) continue;
        out << asin_list[idx] << "," << avg_ratings[idx] << "," << review_counts[idx] << "\n";
        count++;
    }
    out.close();

    cout << "Top 10 results written to top_10_rated.csv\n";
    return 0;
}
