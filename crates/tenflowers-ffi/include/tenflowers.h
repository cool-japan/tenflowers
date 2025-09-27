#ifndef TENFLOWERS_H
#define TENFLOWERS_H

#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// ===== Status Codes =====

typedef enum {
    TENSOR_SUCCESS = 0,
    TENSOR_INVALID_ARGUMENT = 1,
    TENSOR_OUT_OF_MEMORY = 2,
    TENSOR_DEVICE_ERROR = 3,
    TENSOR_SHAPE_MISMATCH = 4,
    TENSOR_DTYPE_MISMATCH = 5,
    TENSOR_UNKNOWN = 999
} TensorStatus;

// ===== Device Types =====

typedef enum {
    TENSOR_DEVICE_CPU = 0,
    TENSOR_DEVICE_GPU = 1
} TensorDevice;

// ===== Opaque Types =====

typedef struct CTensor CTensor;

// ===== Core Functions =====

/**
 * Initialize the TenfloweRS library
 * Must be called before using any other functions
 */
TensorStatus tensor_init(void);

/**
 * Cleanup the TenfloweRS library
 * Should be called when done using the library
 */
TensorStatus tensor_cleanup(void);

/**
 * Get the library version string
 * Returns a pointer to the version string that must be freed with tensor_free_version
 */
const char* tensor_version(void);

/**
 * Free the version string returned by tensor_version
 */
void tensor_free_version(char* version);

// ===== Tensor Management =====

/**
 * Create a new tensor filled with zeros
 * @param shape Array of dimension sizes
 * @param ndim Number of dimensions
 * @param device Device to create tensor on
 * @return Pointer to new tensor or NULL on error
 */
CTensor* tensor_zeros(const size_t* shape, size_t ndim, TensorDevice device);

/**
 * Create a new tensor filled with ones
 * @param shape Array of dimension sizes
 * @param ndim Number of dimensions
 * @param device Device to create tensor on
 * @return Pointer to new tensor or NULL on error
 */
CTensor* tensor_ones(const size_t* shape, size_t ndim, TensorDevice device);

/**
 * Create a new tensor from raw data
 * @param data Pointer to raw data (f32 values)
 * @param shape Array of dimension sizes
 * @param ndim Number of dimensions
 * @param device Device to create tensor on
 * @return Pointer to new tensor or NULL on error
 */
CTensor* tensor_from_data(const float* data, const size_t* shape, size_t ndim, TensorDevice device);

/**
 * Destroy a tensor and free its memory
 * @param tensor Tensor to destroy
 * @return Status code
 */
TensorStatus tensor_destroy(CTensor* tensor);

/**
 * Get tensor shape information
 * @param tensor Tensor to query
 * @param shape_out Array to store shape (must be allocated by caller)
 * @param ndim_out Pointer to store number of dimensions
 * @return Status code
 */
TensorStatus tensor_shape(const CTensor* tensor, size_t* shape_out, size_t* ndim_out);

/**
 * Get read-only pointer to tensor data
 * @param tensor Tensor to query
 * @return Pointer to data or NULL if not available on CPU
 */
const float* tensor_data(const CTensor* tensor);

/**
 * Get mutable pointer to tensor data
 * @param tensor Tensor to query
 * @return Pointer to data or NULL if not available on CPU
 */
float* tensor_data_mut(CTensor* tensor);

/**
 * Get tensor size (total number of elements)
 * @param tensor Tensor to query
 * @return Number of elements
 */
size_t tensor_size(const CTensor* tensor);

/**
 * Get tensor device
 * @param tensor Tensor to query
 * @return Device type
 */
TensorDevice tensor_device(const CTensor* tensor);

/**
 * Move tensor to different device
 * @param tensor Tensor to move
 * @param device Target device
 * @return Status code
 */
TensorStatus tensor_to_device(CTensor* tensor, TensorDevice device);

// ===== Tensor Operations =====

/**
 * Add two tensors element-wise
 * @param a First tensor
 * @param b Second tensor
 * @param out Output tensor (must be pre-allocated)
 * @return Status code
 */
TensorStatus tensor_add(const CTensor* a, const CTensor* b, CTensor* out);

/**
 * Subtract two tensors element-wise
 * @param a First tensor
 * @param b Second tensor
 * @param out Output tensor (must be pre-allocated)
 * @return Status code
 */
TensorStatus tensor_sub(const CTensor* a, const CTensor* b, CTensor* out);

/**
 * Multiply two tensors element-wise
 * @param a First tensor
 * @param b Second tensor
 * @param out Output tensor (must be pre-allocated)
 * @return Status code
 */
TensorStatus tensor_mul(const CTensor* a, const CTensor* b, CTensor* out);

/**
 * Matrix multiplication
 * @param a First tensor
 * @param b Second tensor
 * @param out Output tensor (must be pre-allocated)
 * @return Status code
 */
TensorStatus tensor_matmul(const CTensor* a, const CTensor* b, CTensor* out);

/**
 * Sum tensor along specified axes
 * @param tensor Input tensor
 * @param axis Array of axes to sum over (NULL for all axes)
 * @param axis_len Length of axis array
 * @param keepdims Whether to keep dimensions
 * @param out Output tensor (must be pre-allocated)
 * @return Status code
 */
TensorStatus tensor_sum(const CTensor* tensor, const int* axis, size_t axis_len, bool keepdims, CTensor* out);

/**
 * Mean tensor along specified axes
 * @param tensor Input tensor
 * @param axis Array of axes to average over (NULL for all axes)
 * @param axis_len Length of axis array
 * @param keepdims Whether to keep dimensions
 * @param out Output tensor (must be pre-allocated)
 * @return Status code
 */
TensorStatus tensor_mean(const CTensor* tensor, const int* axis, size_t axis_len, bool keepdims, CTensor* out);

/**
 * Reshape tensor
 * @param tensor Input tensor
 * @param new_shape Array of new dimension sizes
 * @param new_ndim Number of new dimensions
 * @param out Output tensor (must be pre-allocated)
 * @return Status code
 */
TensorStatus tensor_reshape(const CTensor* tensor, const size_t* new_shape, size_t new_ndim, CTensor* out);

/**
 * Transpose tensor
 * @param tensor Input tensor
 * @param axes Array of axis permutation (NULL for reverse order)
 * @param axes_len Length of axes array
 * @param out Output tensor (must be pre-allocated)
 * @return Status code
 */
TensorStatus tensor_transpose(const CTensor* tensor, const size_t* axes, size_t axes_len, CTensor* out);

// ===== Error Handling =====

/**
 * Get the last error message
 * @return Pointer to error string or NULL if no error
 * The returned pointer must be freed with tensor_free_error
 */
const char* tensor_last_error(void);

/**
 * Free error message string
 * @param error Error string to free
 */
void tensor_free_error(char* error);

// ===== Utility Functions =====

/**
 * Check if GPU is available
 * @return true if GPU is available, false otherwise
 */
bool tensor_is_gpu_available(void);

/**
 * Get number of available devices
 * @return Number of devices
 */
size_t tensor_get_device_count(void);

#ifdef __cplusplus
}
#endif

#endif // TENFLOWERS_H