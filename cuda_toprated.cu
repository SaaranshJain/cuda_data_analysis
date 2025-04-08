#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <cuda.h>
#include <algorithm>

#define ASIN_LEN 10
#define HASH_TABLE_SIZE 2048
#define TOP_K 10

typedef struct
{
    float overall;
    char *asin;
    // int asinlen;
    char *reviewtext;
    int reviewtextlen;
} Review;

typedef struct
{
    char asin[ASIN_LEN];
    float total_rating;
    int count;
    bool occupied;
} HashEntry;

typedef struct
{
    char asin[ASIN_LEN];
    float avg_rating;
} TopEntry;

Review **read_reviews(char *filename, int *num_reviews, int *reviews_cap)
{
    FILE *reviews_fp = fopen(filename, "r");

    if (!reviews_fp)
    {
        perror("Failed to open file");
        return NULL;
    }

    *num_reviews = 0;
    *reviews_cap = 32;

    Review **reviews;
    if (cudaMallocHost(&reviews, *reviews_cap * sizeof(Review *)) != cudaSuccess)
    {
        perror("cudaMallocHost failed");
        fclose(reviews_fp);
        return NULL;
    }

    char line[4096];
    while (fgets(line, sizeof(line), reviews_fp) != NULL)
    {
        Review *r;
        if (cudaMallocHost(&r, sizeof(Review)) != cudaSuccess)
        {
            perror("cudaMallocHost for Review failed");
            continue;
        }

        // Initialize pointers
        r->asin = NULL;
        r->reviewtext = NULL;

        // Parse "overall"
        char *overall_ptr = strstr(line, "\"overall\": ");
        if (overall_ptr)
        {
            overall_ptr += 10;
            r->overall = atof(overall_ptr);
        }

        // Parse "asin"
        char *asin_ptr = strstr(line, "\"asin\": \"");
        if (asin_ptr)
        {
            asin_ptr += ASIN_LEN - 1;
            if (cudaMallocHost(&(r->asin), (ASIN_LEN + 1) * sizeof(char)) != cudaSuccess)
            {
                perror("cudaMallocHost for asin failed");
                cudaFreeHost(r);
                continue;
            }
            strncpy(r->asin, asin_ptr, ASIN_LEN);
            r->asin[10] = '\0';
        }

        // Parse "reviewText"
        char *review_ptr = strstr(line, "\"reviewText\": \"");
        if (review_ptr)
        {
            review_ptr += 15;
            char *end_quote = strchr(review_ptr, '"');
            int len = end_quote ? (end_quote - review_ptr) : (int)(strlen(review_ptr));
            r->reviewtextlen = len;

            if (cudaMallocHost(&(r->reviewtext), (len + 1) * sizeof(char)) != cudaSuccess)
            {
                perror("cudaMallocHost for reviewText failed");
                if (r->asin)
                    cudaFreeHost(r->asin);
                cudaFreeHost(r);
                continue;
            }

            strncpy(r->reviewtext, review_ptr, len);
            r->reviewtext[len] = '\0';
        }

        if (*num_reviews >= *reviews_cap)
        {
            *reviews_cap <<= 1;
            Review **new_reviews;
            if (cudaMallocHost(&new_reviews, *reviews_cap * sizeof(Review *)) != cudaSuccess)
            {
                perror("Realloc failed");
                if (r->asin)
                    cudaFreeHost(r->asin);
                if (r->reviewtext)
                    cudaFreeHost(r->reviewtext);
                cudaFreeHost(r);
                break;
            }
            cudaMemcpy(new_reviews, reviews, *num_reviews * sizeof(Review *), cudaMemcpyHostToHost);
            cudaFreeHost(reviews);
            reviews = new_reviews;
        }

        reviews[(*num_reviews)++] = r;
    }

    fclose(reviews_fp);
    return reviews;
}

__device__ bool asin_equal(const char a[10], const char *b)
{
    for (int i = 0; i < ASIN_LEN; ++i)
    {
        if (a[i] != b[i]) return false;
    }
    return true;
}

__device__ unsigned int hash_asin(const char *asin)
{
    const unsigned int FNV_PRIME = 16777619u;
    const unsigned int OFFSET_BASIS = 2166136261u;
    unsigned int hash = OFFSET_BASIS;

    for (int i = 0; i < ASIN_LEN && asin[i]; ++i)
    {
        hash ^= (unsigned int)asin[i];
        hash *= FNV_PRIME;
    }

    return hash & (HASH_TABLE_SIZE - 1);
}

__global__ void compute_ratings(Review **reviews, int *num_reviews_ptr, int *work_per_thread_ptr, HashEntry *d_table)
{
    int num_reviews = *num_reviews_ptr;
    int work_per_thread = *work_per_thread_ptr;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int start = tid * work_per_thread;
    int end = min(start + work_per_thread, num_reviews);

    for (int i = start; i < end; ++i)
    {
        Review *r = reviews[i];
        char *asin = r->asin;
        float rating = r->overall;

        unsigned int idx = hash_asin(asin);

        // linear probing
        while (true)
        {
            if (!d_table[idx].occupied)
            {
                if (atomicCAS((int *)&d_table[idx].occupied, 0, 1) == 0)
                {
                    // we got the slot
                    for (int j = 0; j < ASIN_LEN; ++j)
                        d_table[idx].asin[j] = asin[j];

                    d_table[idx].total_rating = rating;
                    d_table[idx].count = 1;
                    break;
                }
            }
            
            if (asin_equal(d_table[idx].asin, asin))
            {
                atomicAdd(&d_table[idx].total_rating, rating);
                atomicAdd(&d_table[idx].count, 1);
                break;
            }

            idx = (idx + 1) & (HASH_TABLE_SIZE - 1); // wrap around
        }
    }
}

__global__ void find_top_k(HashEntry *d_table, TopEntry *d_top_k)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < HASH_TABLE_SIZE && d_table[tid].occupied)
    {
        float avg_rating = d_table[tid].total_rating / d_table[tid].count;
        
        // Use atomic operations to update top-k
        for (int i = 0; i < TOP_K; i++)
        {
            if (avg_rating > d_top_k[i].avg_rating)
            {
                // Swap with current entry
                TopEntry temp = d_top_k[i];
                d_top_k[i].avg_rating = avg_rating;
                for (int j = 0; j < ASIN_LEN; j++)
                    d_top_k[i].asin[j] = d_table[tid].asin[j];
                
                // Move the replaced entry down
                for (int j = i + 1; j < TOP_K; j++)
                {
                    TopEntry temp2 = d_top_k[j];
                    d_top_k[j] = temp;
                    temp = temp2;
                }
                break;
            }
        }
    }
}

int main()
{
    int num_reviews, reviews_cap;
    char filename[24] = "data/sample.json";
    Review **h_reviews = read_reviews(filename, &num_reviews, &reviews_cap);

    if (!h_reviews)
    {
        printf("Failed to read reviews\n");
        return 1;
    }

    // Allocate device memory for hash table
    HashEntry *d_table;
    cudaMalloc(&d_table, HASH_TABLE_SIZE * sizeof(HashEntry));
    cudaMemset(d_table, 0, HASH_TABLE_SIZE * sizeof(HashEntry));

    // Allocate device memory for top-k results
    TopEntry *d_top_k;
    cudaMalloc(&d_top_k, TOP_K * sizeof(TopEntry));
    cudaMemset(d_top_k, 0, TOP_K * sizeof(TopEntry));

    // Calculate work per thread
    int threads_per_block = 256;
    int num_blocks = (num_reviews + threads_per_block - 1) / threads_per_block;
    int work_per_thread = (num_reviews + num_blocks * threads_per_block - 1) / (num_blocks * threads_per_block);

    // Launch kernel to compute ratings
    compute_ratings<<<num_blocks, threads_per_block>>>(h_reviews, &num_reviews, &work_per_thread, d_table);
    cudaDeviceSynchronize();

    // Launch kernel to find top-k
    find_top_k<<<1, HASH_TABLE_SIZE>>>(d_table, d_top_k);
    cudaDeviceSynchronize();

    // Copy results back to host
    TopEntry *h_top_k = (TopEntry *)malloc(TOP_K * sizeof(TopEntry));
    cudaMemcpy(h_top_k, d_top_k, TOP_K * sizeof(TopEntry), cudaMemcpyDeviceToHost);

    // Print top 10 rated products
    printf("Top 10 Rated Products:\n");
    printf("=====================\n");
    for (int i = 0; i < TOP_K; i++)
    {
        if (h_top_k[i].avg_rating > 0)
        {
            printf("%d. ASIN: %s, Average Rating: %.2f\n", 
                   i + 1, h_top_k[i].asin, h_top_k[i].avg_rating);
        }
    }

    // Cleanup
    free(h_top_k);
    cudaFree(d_top_k);
    cudaFree(d_table);
    
    // Free host memory
    for (int i = 0; i < num_reviews; i++)
    {
        if (h_reviews[i]->asin)
            cudaFreeHost(h_reviews[i]->asin);
        if (h_reviews[i]->reviewtext)
            cudaFreeHost(h_reviews[i]->reviewtext);
        cudaFreeHost(h_reviews[i]);
    }
    cudaFreeHost(h_reviews);

    return 0;
}
