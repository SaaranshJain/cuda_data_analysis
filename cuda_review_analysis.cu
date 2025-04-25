#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>

#define MAX_REVIEWS 6740000       // Corrected from 6.74 million
#define MAX_REVIEW_LEN 36000     // Actual max review length
#define MAX_LEXICON_WORDS 7600
#define MAX_WORD_LEN 32
#define CHUNK_SIZE 10000         // Process 10k reviews at a time

// __device__ bool is_valid_char(char c) {
//     return (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '_' || c == '-';
// }


__device__ bool custom_strncmp(const char* a, const char* b, int max_len) {
    for(int i=0; i<max_len; i++) {
        if(a[i] != b[i]) return false;
        if(a[i] == '\0' && b[i] == '\0') return true;
    }
    return b[max_len] == '\0';
}

__device__ char to_lower(char c) {
    return (c >= 'A' && c <= 'Z') ? c + 32 : c;
}

__device__ bool is_alphanum(char c) {
    c = to_lower(c);
    return (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9');
}

__global__ void sentiment_kernel(char* d_reviews, int* d_offsets, int num_reviews,
                                 char* d_lex_words, float* d_lex_scores, 
                                 int* d_lex_offsets, int lexicon_size,
                                 int* d_sentiment_result) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_reviews) return;

    char* review = &d_reviews[d_offsets[i]];
    float score = 0.0f;
    int start = 0, end = 0;

    while (review[end] != '\0') {
        // Skip non-alphanumeric characters
        while (review[end] != '\0' && !is_alphanum(review[end])) end++;
        start = end;
        
        // Extract token and lowercase
        while (review[end] != '\0' && is_alphanum(review[end])) {
            review[end] = to_lower(review[end]);
            end++;
        }

        int word_len = end - start;
        if (word_len > 0 && word_len < MAX_WORD_LEN) {
            for (int j = 0; j < lexicon_size; j++) {
                char* lex_word = &d_lex_words[d_lex_offsets[j]];
                bool match = true;
                
                // Manual string comparison
                for (int k = 0; k < word_len; k++) {
                    if (lex_word[k] == '\0' || review[start + k] != lex_word[k]) {
                        match = false;
                        break;
                    }
                }
                
                if (match && lex_word[word_len] == '\0') {
                    atomicAdd(&score, d_lex_scores[j]);
                }
            }
        }
    }
    
    d_sentiment_result[i] = (score > 0) ? 1 : (score < 0) ? -1 : 0;
}


void read_lexicon(const char* filename, char* lex_words, float* lex_scores, 
                 int* lex_offsets, int* lex_size) {
    FILE* file = fopen(filename, "r");
    if (!file) { perror("Lexicon open failed"); exit(1); }
    
    int offset = 0;
    *lex_size = 0;
    char word[MAX_WORD_LEN];
    float score;
    
    while (fscanf(file, "%s %f", word, &score) == 2 && *lex_size < MAX_LEXICON_WORDS) {
        lex_offsets[*lex_size] = offset;
        strcpy(&lex_words[offset], word);
        lex_scores[*lex_size] = score;
        offset += strlen(word) + 1;
        (*lex_size)++;
    }
    fclose(file);
}

void process_chunk(char* h_reviews, int* h_offsets, int chunk_size,
                  char* d_lex_words, float* d_lex_scores,
                  int* d_lex_offsets, int lexicon_size,
                  FILE* output_fp) {
    // Allocate GPU memory for chunk
    char* d_reviews;
    int* d_offsets;
    int* d_sentiment_result;
    
    size_t review_bytes = h_offsets[chunk_size-1] + strlen(&h_reviews[h_offsets[chunk_size-1]]) + 1;
    
    cudaMalloc(&d_reviews, review_bytes);
    cudaMalloc(&d_offsets, sizeof(int)*chunk_size);
    cudaMalloc(&d_sentiment_result, sizeof(int)*chunk_size);

    // Copy data to GPU
    cudaMemcpy(d_reviews, h_reviews, review_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_offsets, h_offsets, sizeof(int)*chunk_size, cudaMemcpyHostToDevice);

    // Launch kernel
    int threads = 256;
    int blocks = (chunk_size + threads - 1) / threads;
    sentiment_kernel<<<blocks, threads>>>(d_reviews, d_offsets, chunk_size,
                                         d_lex_words, d_lex_scores, 
                                         d_lex_offsets, lexicon_size,
                                         d_sentiment_result);

    // Copy results back
    int* h_sentiment = (int*)malloc(sizeof(int)*chunk_size);
    cudaMemcpy(h_sentiment, d_sentiment_result, sizeof(int)*chunk_size, cudaMemcpyDeviceToHost);

    // Write results
    for(int i = 0; i < chunk_size; i++) {
        const char* sentiment;
        if (h_sentiment[i] > 0) sentiment = "Positive";
        else if (h_sentiment[i] < 0) sentiment = "Negative";
        else sentiment = "Neutral";
        fprintf(output_fp, "%s\n", sentiment);
    }

    // Cleanup
    free(h_sentiment);
    cudaFree(d_reviews);
    cudaFree(d_offsets);
    cudaFree(d_sentiment_result);
}

int main() {
    // Initialize lexicon
    char lex_words[MAX_LEXICON_WORDS * MAX_WORD_LEN] = {0};
    float lex_scores[MAX_LEXICON_WORDS] = {0};
    int lex_offsets[MAX_LEXICON_WORDS] = {0};
    int lexicon_size;
    read_lexicon("data/cleaned_lexicon.txt", lex_words, lex_scores, lex_offsets, &lexicon_size);

    // Copy lexicon to GPU (once)
    char* d_lex_words;
    float* d_lex_scores;
    int* d_lex_offsets;
    cudaMalloc(&d_lex_words, MAX_LEXICON_WORDS * MAX_WORD_LEN);
    cudaMalloc(&d_lex_scores, sizeof(float) * MAX_LEXICON_WORDS);
    cudaMalloc(&d_lex_offsets, sizeof(int) * MAX_LEXICON_WORDS);
    cudaMemcpy(d_lex_words, lex_words, MAX_LEXICON_WORDS * MAX_WORD_LEN, cudaMemcpyHostToDevice);
    cudaMemcpy(d_lex_scores, lex_scores, sizeof(float) * MAX_LEXICON_WORDS, cudaMemcpyHostToDevice);
    cudaMemcpy(d_lex_offsets, lex_offsets, sizeof(int) * MAX_LEXICON_WORDS, cudaMemcpyHostToDevice);

    FILE* review_fp = fopen("data/reviewData.csv", "r");
    FILE* output_fp = fopen("score_3.txt", "w");
    if (!review_fp || !output_fp) { perror("File open failed"); return 1; }

    char* line = (char*)malloc(MAX_REVIEW_LEN);
    fgets(line, MAX_REVIEW_LEN, review_fp); // Skip header

    int total_processed = 0;
    while (total_processed < MAX_REVIEWS) {
        // Allocate host memory for chunk
        char* h_reviews = (char*)malloc(MAX_REVIEW_LEN * CHUNK_SIZE);
        int* h_offsets = (int*)malloc(sizeof(int) * CHUNK_SIZE);
        if (!h_reviews || !h_offsets) {
            perror("Chunk allocation failed");
            break;
        }

        int chunk_size = 0;
        int cursor = 0;
        
        // Read chunk with improved CSV parsing
        while (chunk_size < CHUNK_SIZE && fgets(line, MAX_REVIEW_LEN, review_fp)) {
            char* review = strchr(line, ',') + 1; // Simple CSV parsing
            char* end = line + strlen(line);
            
            // Handle quoted fields
            if (*review == '"') {
                char* closing_quote = strchr(review + 1, '"');
                if (closing_quote) {
                    review++;
                    *closing_quote = '\0';
                    end = closing_quote;
                }
            }
            
            // Remove trailing newline
            if (end[-1] == '\n') end[-1] = '\0';
            
            size_t len = end - review;
            if (cursor + len + 1 > (size_t)MAX_REVIEW_LEN * CHUNK_SIZE) break;
            
            h_offsets[chunk_size] = cursor;
            memcpy(h_reviews + cursor, review, len + 1);
            cursor += len + 1;
            chunk_size++;
        }

        if (chunk_size == 0) break;
        
        // Process chunk
        process_chunk(h_reviews, h_offsets, chunk_size,
                     d_lex_words, d_lex_scores, d_lex_offsets, lexicon_size,
                     output_fp);
        
        // Cleanup and prepare for next chunk
        free(h_reviews);
        free(h_offsets);
        total_processed += chunk_size;
        printf("Processed %d reviews (total: %d)\n", chunk_size, total_processed);
    }

    // Final cleanup
    fclose(review_fp);
    fclose(output_fp);
    free(line);
    cudaFree(d_lex_words);
    cudaFree(d_lex_scores);
    cudaFree(d_lex_offsets);
    
    return 0;
}
