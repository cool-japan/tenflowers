/**
 * @file c_api_basic.c
 * @brief Basic C API usage example for TenfloweRS
 *
 * This example demonstrates basic tensor operations using the TenfloweRS C API.
 *
 * Build:
 *   gcc -o c_api_basic c_api_basic.c -L../target/release -ltenflowers -lm
 *
 * Run:
 *   LD_LIBRARY_PATH=../target/release ./c_api_basic
 */

#include "../include/tenflowers.h"
#include <stdio.h>
#include <stdlib.h>

int main() {
    printf("TenfloweRS C API Basic Example\n");
    printf("==============================\n\n");

    // Initialize TenfloweRS
    TF_Status* status = TF_NewStatus();
    if (status == NULL) {
        fprintf(stderr, "Failed to create status object\n");
        return 1;
    }

    printf("1. Creating tensors...\n");

    // Create shape for 2x3 tensor
    size_t shape[] = {2, 3};
    size_t ndim = 2;

    // Create zero tensor
    TF_Tensor* zeros = TF_Zeros(shape, ndim, TF_FLOAT32, status);
    if (TF_GetStatusCode(status) != TF_OK) {
        fprintf(stderr, "Failed to create zeros tensor: %s\n",
                TF_GetStatusMessage(status));
        TF_DeleteStatus(status);
        return 1;
    }
    printf("   Created zeros tensor: shape [%zu, %zu]\n", shape[0], shape[1]);

    // Create ones tensor
    TF_Tensor* ones = TF_Ones(shape, ndim, TF_FLOAT32, status);
    if (TF_GetStatusCode(status) != TF_OK) {
        fprintf(stderr, "Failed to create ones tensor: %s\n",
                TF_GetStatusMessage(status));
        TF_DeleteTensor(zeros);
        TF_DeleteStatus(status);
        return 1;
    }
    printf("   Created ones tensor: shape [%zu, %zu]\n", shape[0], shape[1]);

    printf("\n2. Performing tensor addition...\n");

    // Add tensors
    TF_Tensor* result = TF_Add(zeros, ones, status);
    if (TF_GetStatusCode(status) != TF_OK) {
        fprintf(stderr, "Failed to add tensors: %s\n",
                TF_GetStatusMessage(status));
        TF_DeleteTensor(zeros);
        TF_DeleteTensor(ones);
        TF_DeleteStatus(status);
        return 1;
    }
    printf("   Addition successful!\n");

    // Get result data
    float* data = (float*)TF_TensorData(result);
    printf("   Result values: [");
    for (size_t i = 0; i < 6; i++) {
        printf("%.1f", data[i]);
        if (i < 5) printf(", ");
    }
    printf("]\n");

    printf("\n3. Matrix multiplication...\n");

    // Create matrices for matmul
    size_t shape_a[] = {2, 3};
    size_t shape_b[] = {3, 4};

    TF_Tensor* mat_a = TF_Ones(shape_a, 2, TF_FLOAT32, status);
    TF_Tensor* mat_b = TF_Ones(shape_b, 2, TF_FLOAT32, status);

    TF_Tensor* mat_result = TF_MatMul(mat_a, mat_b, status);
    if (TF_GetStatusCode(status) != TF_OK) {
        fprintf(stderr, "Failed to multiply matrices: %s\n",
                TF_GetStatusMessage(status));
    } else {
        printf("   Matrix multiplication successful!\n");
        printf("   Result shape: [2, 4]\n");
    }

    printf("\n4. Cleanup...\n");

    // Cleanup
    TF_DeleteTensor(zeros);
    TF_DeleteTensor(ones);
    TF_DeleteTensor(result);
    TF_DeleteTensor(mat_a);
    TF_DeleteTensor(mat_b);
    TF_DeleteTensor(mat_result);
    TF_DeleteStatus(status);

    printf("   Cleanup complete!\n");
    printf("\nExample completed successfully!\n");

    return 0;
}
