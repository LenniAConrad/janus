/*
 * Persistent OpenCL 1.2 worker for Janus's pinned BT4J-v2 evaluator.
 *
 * This executable deliberately sits outside the Rust process.  Janus keeps
 * `unsafe_code = "forbid"`; the vendor runtime, model-sized device allocation,
 * and kernel failures are isolated behind a bounded binary pipe protocol.
 */

#include <CL/cl.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#if defined(_WIN32)
#include <fcntl.h>
#include <io.h>
#endif

namespace {

constexpr std::uint32_t kProtocolVersion = 1;
constexpr std::size_t kHeaderBytes = 24;
constexpr std::size_t kInputChannels = 112;
constexpr std::size_t kTokens = 64;
constexpr std::size_t kInputFloats = kInputChannels * kTokens;
constexpr std::size_t kEmbedding = 1024;
constexpr std::size_t kEncoderLayers = 15;
constexpr std::size_t kAttentionHeads = 32;
constexpr std::size_t kFfnHidden = 1536;
constexpr std::size_t kSmolgenChannels = 32;
constexpr std::size_t kSmolgenHidden = 256;
constexpr std::size_t kSmolgenPerHead = 256;
constexpr std::size_t kAttentionMap = kTokens * kTokens;
constexpr std::size_t kPreprocChannels = 12;
constexpr std::size_t kPreprocPerToken = 512;
constexpr std::size_t kInternalPolicy = 67 * kTokens;
constexpr std::size_t kWdl = 3;
constexpr std::size_t kResponseFloats = kInternalPolicy + kWdl;
constexpr std::size_t kPredictRequestBytes = kInputFloats * sizeof(float);
constexpr std::size_t kPredictResponseBytes = kResponseFloats * sizeof(float);
constexpr std::size_t kMaxFramePayload = 64 * 1024;
constexpr std::uint64_t kMaxModelBytes = 1ULL << 30U;
constexpr std::uint32_t kMaxStringBytes = 1024;

constexpr std::uint16_t kOpcodeHello = 1;
constexpr std::uint16_t kOpcodePredict = 2;
constexpr std::uint16_t kOpcodeShutdown = 3;
constexpr std::uint32_t kStatusOk = 0;
constexpr std::uint32_t kStatusProtocol = 1;
constexpr std::uint32_t kStatusInput = 2;
constexpr std::uint32_t kStatusBackend = 3;

[[noreturn]] void fail(const std::string& message) {
    throw std::runtime_error(message);
}

std::string lower(std::string value) {
    for (char& c : value) {
        const unsigned char byte = static_cast<unsigned char>(c);
        c = static_cast<char>(std::tolower(byte));
    }
    return value;
}

void check_cl(cl_int error, const char* operation) {
    if (error != CL_SUCCESS) {
        std::ostringstream message;
        message << operation << " failed with OpenCL error " << error;
        fail(message.str());
    }
}

class Mem {
public:
    Mem() = default;
    explicit Mem(cl_mem value) : value_(value) {}
    ~Mem() {
        if (value_ != nullptr) {
            clReleaseMemObject(value_);
        }
    }
    Mem(const Mem&) = delete;
    Mem& operator=(const Mem&) = delete;
    Mem(Mem&& other) noexcept : value_(other.value_) {
        other.value_ = nullptr;
    }
    Mem& operator=(Mem&& other) noexcept {
        if (this != &other) {
            if (value_ != nullptr) {
                clReleaseMemObject(value_);
            }
            value_ = other.value_;
            other.value_ = nullptr;
        }
        return *this;
    }
    cl_mem get() const { return value_; }

private:
    cl_mem value_ = nullptr;
};

class Kernel {
public:
    Kernel() = default;
    explicit Kernel(cl_kernel value) : value_(value) {}
    ~Kernel() {
        if (value_ != nullptr) {
            clReleaseKernel(value_);
        }
    }
    Kernel(const Kernel&) = delete;
    Kernel& operator=(const Kernel&) = delete;
    Kernel(Kernel&& other) noexcept : value_(other.value_) {
        other.value_ = nullptr;
    }
    Kernel& operator=(Kernel&& other) noexcept {
        if (this != &other) {
            if (value_ != nullptr) {
                clReleaseKernel(value_);
            }
            value_ = other.value_;
            other.value_ = nullptr;
        }
        return *this;
    }
    cl_kernel get() const { return value_; }

private:
    cl_kernel value_ = nullptr;
};

std::string device_string(cl_device_id device, cl_device_info key) {
    std::size_t bytes = 0;
    check_cl(clGetDeviceInfo(device, key, 0, nullptr, &bytes), "clGetDeviceInfo(size)");
    std::vector<char> buffer(bytes == 0 ? 1 : bytes, '\0');
    check_cl(clGetDeviceInfo(device, key, buffer.size(), buffer.data(), nullptr), "clGetDeviceInfo(value)");
    return std::string(buffer.data());
}

std::string platform_string(cl_platform_id platform, cl_platform_info key) {
    std::size_t bytes = 0;
    check_cl(clGetPlatformInfo(platform, key, 0, nullptr, &bytes), "clGetPlatformInfo(size)");
    std::vector<char> buffer(bytes == 0 ? 1 : bytes, '\0');
    check_cl(clGetPlatformInfo(platform, key, buffer.size(), buffer.data(), nullptr), "clGetPlatformInfo(value)");
    return std::string(buffer.data());
}

struct Candidate {
    cl_platform_id platform = nullptr;
    cl_device_id device = nullptr;
    std::string platform_name;
    std::string vendor;
    std::string name;
    cl_device_type type = 0;
    cl_ulong global_memory = 0;
};

bool vendor_matches(const std::string& requested, const Candidate& candidate) {
    if (requested == "auto") {
        return true;
    }
    const std::string text = lower(candidate.vendor + " " + candidate.platform_name + " " + candidate.name);
    if (requested == "nvidia") {
        return text.find("nvidia") != std::string::npos;
    }
    if (requested == "amd") {
        return text.find("advanced micro devices") != std::string::npos
                || text.find(" amd") != std::string::npos
                || text.rfind("amd", 0) == 0;
    }
    return requested == "intel" && text.find("intel") != std::string::npos;
}

std::uint32_t vendor_code(const Candidate& candidate) {
    const std::string text = lower(candidate.vendor + " " + candidate.platform_name + " " + candidate.name);
    if (text.find("nvidia") != std::string::npos) return 1;
    if (text.find("advanced micro devices") != std::string::npos
            || text.find(" amd") != std::string::npos || text.rfind("amd", 0) == 0) return 2;
    if (text.find("intel") != std::string::npos) return 3;
    return 0;
}

std::vector<Candidate> enumerate_devices(const std::string& requested_vendor, bool allow_cpu) {
    cl_uint platform_count = 0;
    const cl_int platform_status = clGetPlatformIDs(0, nullptr, &platform_count);
    if (platform_status != CL_SUCCESS || platform_count == 0) {
        return {};
    }
    std::vector<cl_platform_id> platforms(platform_count);
    check_cl(clGetPlatformIDs(platform_count, platforms.data(), nullptr), "clGetPlatformIDs");
    std::vector<Candidate> candidates;
    const cl_device_type accepted = allow_cpu ? (CL_DEVICE_TYPE_GPU | CL_DEVICE_TYPE_CPU) : CL_DEVICE_TYPE_GPU;
    for (cl_platform_id platform : platforms) {
        cl_uint device_count = 0;
        const cl_int status = clGetDeviceIDs(platform, accepted, 0, nullptr, &device_count);
        if (status == CL_DEVICE_NOT_FOUND || device_count == 0) {
            continue;
        }
        check_cl(status, "clGetDeviceIDs(count)");
        std::vector<cl_device_id> devices(device_count);
        check_cl(clGetDeviceIDs(platform, accepted, device_count, devices.data(), nullptr), "clGetDeviceIDs(list)");
        for (cl_device_id device : devices) {
            Candidate candidate;
            candidate.platform = platform;
            candidate.device = device;
            candidate.platform_name = platform_string(platform, CL_PLATFORM_NAME);
            candidate.vendor = device_string(device, CL_DEVICE_VENDOR);
            candidate.name = device_string(device, CL_DEVICE_NAME);
            check_cl(clGetDeviceInfo(device, CL_DEVICE_TYPE, sizeof(candidate.type), &candidate.type, nullptr),
                    "clGetDeviceInfo(type)");
            check_cl(clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(candidate.global_memory),
                            &candidate.global_memory, nullptr),
                    "clGetDeviceInfo(memory)");
            if (!allow_cpu && (candidate.type & CL_DEVICE_TYPE_GPU) == 0) {
                continue;
            }
            if (vendor_code(candidate) != 0 && vendor_matches(requested_vendor, candidate)) {
                candidates.push_back(std::move(candidate));
            }
        }
    }
    std::stable_sort(candidates.begin(), candidates.end(), [](const Candidate& left, const Candidate& right) {
        const bool left_gpu = (left.type & CL_DEVICE_TYPE_GPU) != 0;
        const bool right_gpu = (right.type & CL_DEVICE_TYPE_GPU) != 0;
        if (left_gpu != right_gpu) {
            return left_gpu;
        }
        return left.global_memory > right.global_memory;
    });
    return candidates;
}

const char* kKernelSource = R"CLC(
inline float bt4_softplus(float x) {
    if (x > 20.0f) return x;
    if (x < -20.0f) return exp(x);
    return log1p(exp(x));
}

inline float bt4_activate_one(float x, int activation) {
    if (activation == 1) return fmax(x, 0.0f);
    if (activation == 2) return x * tanh(bt4_softplus(x));
    if (activation == 3) return x / (1.0f + exp(-x));
    if (activation == 4) return tanh(x);
    return x;
}

__kernel void dense_fm(__global const float* input, __global const float* weights,
        __global const float* bias, __global float* output, uint in_features,
        uint out_features, uint tokens) {
    const size_t index = get_global_id(0);
    const size_t total = (size_t)out_features * tokens;
    if (index >= total) return;
    const uint out_feature = (uint)(index / tokens);
    const uint token = (uint)(index - (size_t)out_feature * tokens);
    float sum = bias[out_feature];
    const size_t weight_base = (size_t)out_feature * in_features;
    for (uint in_feature = 0; in_feature < in_features; ++in_feature) {
        sum += weights[weight_base + in_feature] * input[(size_t)in_feature * tokens + token];
    }
    output[index] = sum;
}

__kernel void activate(__global float* values, uint count, int activation) {
    const size_t index = get_global_id(0);
    if (index < count) values[index] = bt4_activate_one(values[index], activation);
}

__kernel void layer_norm_fm(__global float* values, __global const float* gamma,
        __global const float* beta, uint tokens, uint features, float epsilon) {
    const uint token = (uint)get_global_id(0);
    if (token >= tokens) return;
    float mean = 0.0f;
    for (uint feature = 0; feature < features; ++feature) {
        mean += values[(size_t)feature * tokens + token];
    }
    mean /= (float)features;
    float variance = 0.0f;
    for (uint feature = 0; feature < features; ++feature) {
        const float centered = values[(size_t)feature * tokens + token] - mean;
        variance += centered * centered;
    }
    const float inverse_std = 1.0f / sqrt(variance / (float)features + epsilon);
    for (uint feature = 0; feature < features; ++feature) {
        const size_t index = (size_t)feature * tokens + token;
        values[index] = (values[index] - mean) * inverse_std * gamma[feature] + beta[feature];
    }
}

__kernel void prepare_preproc(__global const float* encoded, __global float* output) {
    const size_t index = get_global_id(0);
    if (index >= 768) return;
    const uint token = (uint)(index / 12);
    const uint channel = (uint)(index - (size_t)token * 12);
    output[index] = encoded[(size_t)channel * 64 + token];
}

__kernel void prepare_embedding(__global const float* encoded, __global const float* preprocessed,
        __global float* output) {
    const size_t index = get_global_id(0);
    if (index >= (size_t)624 * 64) return;
    const uint feature = (uint)(index / 64);
    const uint token = (uint)(index - (size_t)feature * 64);
    if (feature < 112) output[index] = encoded[(size_t)feature * 64 + token];
    else output[index] = preprocessed[(size_t)token * 512 + (feature - 112)];
}

__kernel void apply_gate(__global float* values, __global const float* mult,
        __global const float* add, uint count) {
    const size_t index = get_global_id(0);
    if (index < count) values[index] = values[index] * mult[index] + add[index];
}

__kernel void residual_add(__global float* output, __global const float* residual,
        float alpha, uint count) {
    const size_t index = get_global_id(0);
    if (index < count) output[index] = output[index] * alpha + residual[index];
}

__kernel void fm_to_token(__global const float* input, __global float* output,
        uint features, uint tokens) {
    const size_t index = get_global_id(0);
    const size_t total = (size_t)features * tokens;
    if (index >= total) return;
    const uint token = (uint)(index / features);
    const uint feature = (uint)(index - (size_t)token * features);
    output[index] = input[(size_t)feature * tokens + token];
}

__kernel void smolgen_project(__global const float* shared, __global const float* generated,
        __global float* output) {
    const size_t index = get_global_id(0);
    if (index >= (size_t)32 * 4096) return;
    const uint head = (uint)(index / 4096);
    const uint cell = (uint)(index - (size_t)head * 4096);
    float sum = 0.0f;
    const size_t generated_base = (size_t)head * 256;
    const size_t weight_base = (size_t)cell * 256;
    for (uint dimension = 0; dimension < 256; ++dimension) {
        sum += generated[generated_base + dimension] * shared[weight_base + dimension];
    }
    output[index] = sum;
}

__kernel void attention_heads(__global const float* query, __global const float* key,
        __global const float* value, __global const float* bias, __global float* output) {
    const size_t index = get_global_id(0);
    if (index >= (size_t)32 * 64) return;
    const uint head = (uint)(index / 64);
    const uint query_token = (uint)(index - (size_t)head * 64);
    const uint feature_base = head * 32;
    const size_t bias_base = (size_t)head * 4096 + (size_t)query_token * 64;
    const float inverse_scale = 0.1767766952966369f;
    float scores[64];
    float maximum = -3.402823466e+38f;
    for (uint key_token = 0; key_token < 64; ++key_token) {
        float sum = 0.0f;
        for (uint dimension = 0; dimension < 32; ++dimension) {
            const uint feature = feature_base + dimension;
            sum += query[(size_t)feature * 64 + query_token]
                    * key[(size_t)feature * 64 + key_token];
        }
        scores[key_token] = sum * inverse_scale + bias[bias_base + key_token];
        maximum = fmax(maximum, scores[key_token]);
    }
    float denominator = 0.0f;
    for (uint key_token = 0; key_token < 64; ++key_token) {
        scores[key_token] = exp(scores[key_token] - maximum);
        denominator += scores[key_token];
    }
    if (denominator > 0.0f) {
        for (uint key_token = 0; key_token < 64; ++key_token) scores[key_token] /= denominator;
    }
    for (uint dimension = 0; dimension < 32; ++dimension) {
        const uint feature = feature_base + dimension;
        float sum = 0.0f;
        for (uint key_token = 0; key_token < 64; ++key_token) {
            sum += scores[key_token] * value[(size_t)feature * 64 + key_token];
        }
        output[(size_t)feature * 64 + query_token] = sum;
    }
}

__kernel void policy_from_to(__global const float* query, __global const float* key,
        __global float* output, uint dimension) {
    const size_t index = get_global_id(0);
    if (index >= 4096) return;
    const uint from = (uint)(index / 64);
    const uint to = (uint)(index - (size_t)from * 64);
    float sum = 0.0f;
    for (uint feature = 0; feature < dimension; ++feature) {
        sum += query[(size_t)feature * 64 + from] * key[(size_t)feature * 64 + to];
    }
    output[index] = sum / sqrt((float)dimension);
}

__kernel void promotion_logits(__global const float* key, __global const float* weights,
        __global float* internal, uint dimension) {
    const size_t index = get_global_id(0);
    if (index >= 192) return;
    const uint promotion = (uint)(index % 3);
    const uint to_file = (uint)((index / 3) % 8);
    const uint from_file = (uint)(index / 24);
    if (to_file + 1 < from_file || to_file > from_file + 1) return;
    const uint from = 48 + from_file;
    const uint to = 56 + to_file;
    float shared = 0.0f;
    float piece = 0.0f;
    for (uint feature = 0; feature < dimension; ++feature) {
        const float k = key[(size_t)feature * 64 + to];
        shared += k * weights[(size_t)3 * dimension + feature];
        piece += k * weights[(size_t)promotion * dimension + feature];
    }
    internal[4096 + (size_t)from_file * 24 + (size_t)to_file * 3 + promotion]
            = internal[(size_t)from * 64 + to] + shared + piece;
}

__kernel void softmax3(__global float* values) {
    if (get_global_id(0) != 0) return;
    const float maximum = fmax(values[0], fmax(values[1], values[2]));
    const float a = exp(values[0] - maximum);
    const float b = exp(values[1] - maximum);
    const float c = exp(values[2] - maximum);
    const float sum = a + b + c;
    values[0] = a / sum;
    values[1] = b / sum;
    values[2] = c / sum;
}
)CLC";

class Runtime {
public:
    explicit Runtime(const Candidate& selected) : selected_(selected) {
        const cl_context_properties properties[] = {
            CL_CONTEXT_PLATFORM,
            reinterpret_cast<cl_context_properties>(selected.platform),
            0
        };
        cl_int error = CL_SUCCESS;
        context_ = clCreateContext(properties, 1, &selected.device, nullptr, nullptr, &error);
        check_cl(error, "clCreateContext");
        queue_ = clCreateCommandQueue(context_, selected.device, 0, &error);
        check_cl(error, "clCreateCommandQueue");
        const char* source = kKernelSource;
        const std::size_t source_size = std::strlen(source);
        program_ = clCreateProgramWithSource(context_, 1, &source, &source_size, &error);
        check_cl(error, "clCreateProgramWithSource");
        error = clBuildProgram(program_, 1, &selected.device, "-cl-std=CL1.2", nullptr, nullptr);
        if (error != CL_SUCCESS) {
            std::size_t log_size = 0;
            clGetProgramBuildInfo(program_, selected.device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
            std::vector<char> log(log_size == 0 ? 1 : log_size, '\0');
            clGetProgramBuildInfo(program_, selected.device, CL_PROGRAM_BUILD_LOG, log.size(), log.data(), nullptr);
            std::ostringstream message;
            message << "OpenCL program build failed with " << error << ":\n" << log.data();
            fail(message.str());
        }
        dense = make_kernel("dense_fm");
        activation = make_kernel("activate");
        layer_norm = make_kernel("layer_norm_fm");
        prepare_preproc = make_kernel("prepare_preproc");
        prepare_embedding = make_kernel("prepare_embedding");
        gate = make_kernel("apply_gate");
        residual = make_kernel("residual_add");
        transpose = make_kernel("fm_to_token");
        smolgen_project = make_kernel("smolgen_project");
        attention = make_kernel("attention_heads");
        policy = make_kernel("policy_from_to");
        promotion = make_kernel("promotion_logits");
        softmax = make_kernel("softmax3");
    }

    ~Runtime() {
        if (queue_ != nullptr) clFinish(queue_);
        if (program_ != nullptr) clReleaseProgram(program_);
        if (queue_ != nullptr) clReleaseCommandQueue(queue_);
        if (context_ != nullptr) clReleaseContext(context_);
    }
    Runtime(const Runtime&) = delete;
    Runtime& operator=(const Runtime&) = delete;

    const Candidate& selected() const { return selected_; }
    cl_command_queue queue() const { return queue_; }

    Mem upload(const std::vector<float>& values, const std::string& label) const {
        if (values.empty()) fail(label + " is empty");
        cl_int error = CL_SUCCESS;
        cl_mem memory = clCreateBuffer(context_, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                values.size() * sizeof(float), const_cast<float*>(values.data()), &error);
        if (error != CL_SUCCESS) {
            std::ostringstream message;
            message << "uploading " << label << " (" << values.size() << " f32) failed with OpenCL error " << error;
            fail(message.str());
        }
        return Mem(memory);
    }

    Mem scratch(std::size_t floats, const std::string& label) const {
        if (floats == 0) fail(label + " requested a zero-sized buffer");
        cl_int error = CL_SUCCESS;
        cl_mem memory = clCreateBuffer(context_, CL_MEM_READ_WRITE, floats * sizeof(float), nullptr, &error);
        if (error != CL_SUCCESS) {
            std::ostringstream message;
            message << "allocating " << label << " (" << floats << " f32) failed with OpenCL error " << error;
            fail(message.str());
        }
        return Mem(memory);
    }

    void set_mem(Kernel& kernel, cl_uint index, const Mem& memory) const {
        const cl_mem raw = memory.get();
        check_cl(clSetKernelArg(kernel.get(), index, sizeof(raw), &raw), "clSetKernelArg(mem)");
    }
    template <typename T>
    void set_value(Kernel& kernel, cl_uint index, const T& value) const {
        check_cl(clSetKernelArg(kernel.get(), index, sizeof(value), &value), "clSetKernelArg(value)");
    }
    void launch(Kernel& kernel, std::size_t global) const {
        check_cl(clEnqueueNDRangeKernel(queue_, kernel.get(), 1, nullptr, &global, nullptr, 0, nullptr, nullptr),
                "clEnqueueNDRangeKernel");
    }
    void write(const Mem& memory, const float* values, std::size_t count) const {
        check_cl(clEnqueueWriteBuffer(queue_, memory.get(), CL_FALSE, 0, count * sizeof(float), values,
                        0, nullptr, nullptr),
                "clEnqueueWriteBuffer");
    }
    void read(const Mem& memory, float* values, std::size_t count) const {
        check_cl(clEnqueueReadBuffer(queue_, memory.get(), CL_TRUE, 0, count * sizeof(float), values,
                        0, nullptr, nullptr),
                "clEnqueueReadBuffer");
    }
    void zero(const Mem& memory, std::size_t count) const {
        const float value = 0.0F;
        check_cl(clEnqueueFillBuffer(queue_, memory.get(), &value, sizeof(value), 0, count * sizeof(float),
                        0, nullptr, nullptr),
                "clEnqueueFillBuffer");
    }

    Kernel dense;
    Kernel activation;
    Kernel layer_norm;
    Kernel prepare_preproc;
    Kernel prepare_embedding;
    Kernel gate;
    Kernel residual;
    Kernel transpose;
    Kernel smolgen_project;
    Kernel attention;
    Kernel policy;
    Kernel promotion;
    Kernel softmax;

private:
    Kernel make_kernel(const char* name) const {
        cl_int error = CL_SUCCESS;
        cl_kernel kernel = clCreateKernel(program_, name, &error);
        check_cl(error, name);
        return Kernel(kernel);
    }

    Candidate selected_;
    cl_context context_ = nullptr;
    cl_command_queue queue_ = nullptr;
    cl_program program_ = nullptr;
};

struct Header {
    std::string name;
    float epsilon = 0.0F;
};

class Reader {
public:
    explicit Reader(const std::string& path) : input_(path, std::ios::binary) {
        if (!input_) fail("cannot open BT4 model " + path);
        input_.seekg(0, std::ios::end);
        const std::streamoff length = input_.tellg();
        if (length <= 0 || static_cast<std::uint64_t>(length) > kMaxModelBytes) {
            fail("BT4 model size is outside the 1 GiB safety bound");
        }
        file_bytes_ = static_cast<std::uint64_t>(length);
        input_.seekg(0, std::ios::beg);
        header_ = read_header();
    }

    const Header& header() const { return header_; }

    std::uint32_t u32(const std::string& label) {
        std::array<unsigned char, 4> bytes{};
        read_exact(bytes.data(), bytes.size(), label);
        return static_cast<std::uint32_t>(bytes[0])
                | (static_cast<std::uint32_t>(bytes[1]) << 8U)
                | (static_cast<std::uint32_t>(bytes[2]) << 16U)
                | (static_cast<std::uint32_t>(bytes[3]) << 24U);
    }

    float f32(const std::string& label) {
        const std::uint32_t bits = u32(label);
        float value = 0.0F;
        static_assert(sizeof(value) == sizeof(bits), "f32 size");
        std::memcpy(&value, &bits, sizeof(value));
        if (!std::isfinite(value)) fail(label + " is not finite");
        return value;
    }

    std::string string(const std::string& label) {
        const std::uint32_t length = u32(label + " length");
        if (length > kMaxStringBytes) fail(label + " exceeds string safety bound");
        std::string value(length, '\0');
        if (length != 0) read_exact(value.data(), length, label);
        return value;
    }

    bool boolean(const std::string& label) {
        unsigned char value = 0;
        read_exact(&value, 1, label);
        if (value > 1) fail(label + " is not a canonical boolean");
        return value == 1;
    }

    std::vector<float> tensor(std::size_t expected, const std::string& label) {
        const std::uint32_t count = u32(label + " length");
        if (count != expected) {
            std::ostringstream message;
            message << label << " has " << count << " elements; expected " << expected;
            fail(message.str());
        }
        std::vector<float> values(expected);
        std::vector<unsigned char> bytes(expected * sizeof(float));
        if (!bytes.empty()) read_exact(bytes.data(), bytes.size(), label);
        for (std::size_t index = 0; index < expected; ++index) {
            const std::size_t base = index * 4;
            const std::uint32_t bits = static_cast<std::uint32_t>(bytes[base])
                    | (static_cast<std::uint32_t>(bytes[base + 1]) << 8U)
                    | (static_cast<std::uint32_t>(bytes[base + 2]) << 16U)
                    | (static_cast<std::uint32_t>(bytes[base + 3]) << 24U);
            std::memcpy(&values[index], &bits, sizeof(bits));
            if (!std::isfinite(values[index])) fail(label + " contains a non-finite weight");
        }
        return values;
    }

    void count(std::size_t expected, const std::string& label) {
        const std::uint32_t actual = u32(label);
        if (actual != expected) {
            std::ostringstream message;
            message << label << " is " << actual << "; expected " << expected;
            fail(message.str());
        }
    }

    void activation(const char* expected, const std::string& label) {
        const std::string actual = string(label);
        if (actual != expected) fail(label + " is " + actual + "; expected " + expected);
    }

    void finish() {
        const int extra = input_.peek();
        if (extra != std::char_traits<char>::eof()) fail("BT4 model has trailing bytes");
        if (position_ != file_bytes_) fail("BT4 reader position does not match file length");
    }

private:
    Header read_header() {
        std::array<char, 4> magic{};
        read_exact(magic.data(), magic.size(), "BT4 magic");
        if (magic != std::array<char, 4>{'B', 'T', '4', 'J'}) fail("invalid BT4 magic");
        if (u32("BT4 version") != 2) fail("OpenCL worker requires BT4J v2");
        Header header;
        header.name = string("architecture name");
        if (header.name.empty()) fail("BT4 architecture name is empty");
        if (string("input format") != "CLASSICAL_112") fail("worker requires CLASSICAL_112");
        if (string("input embedding") != "PE_DENSE") fail("worker requires PE_DENSE");
        exact(u32("input channels"), kInputChannels, "input channels");
        exact(u32("tokens"), kTokens, "tokens");
        exact(u32("embedding"), kEmbedding, "embedding");
        exact(u32("encoder layers"), kEncoderLayers, "encoder layers");
        exact(u32("attention heads"), kAttentionHeads, "attention heads");
        exact(u32("policy size"), 1858, "policy size");
        header.epsilon = f32("layer norm epsilon");
        if (!(header.epsilon > 0.0F)) fail("layer norm epsilon must be positive");
        exact(u32("FFN hidden"), kFfnHidden, "FFN hidden");
        exact(u32("smolgen channels"), kSmolgenChannels, "smolgen channels");
        exact(u32("smolgen hidden"), kSmolgenHidden, "smolgen hidden");
        exact(u32("smolgen per head"), kSmolgenPerHead, "smolgen per head");
        exact(u32("smolgen global"), kAttentionMap, "smolgen global");
        activation("MISH", "default activation");
        activation("SWISH", "smolgen activation");
        activation("MISH", "FFN activation");
        if (!boolean("has input preproc") || !boolean("has input FFN")
                || !boolean("has input gates") || !boolean("has smolgen")) {
            fail("worker requires all pinned BT4J-v2 extensions");
        }
        return header;
    }

    static void exact(std::uint32_t actual, std::size_t expected, const char* label) {
        if (actual != expected) {
            std::ostringstream message;
            message << label << " is " << actual << "; expected " << expected;
            fail(message.str());
        }
    }

    void read_exact(void* destination, std::size_t bytes, const std::string& label) {
        if (bytes > file_bytes_ - position_) fail("truncated BT4 model while reading " + label);
        input_.read(static_cast<char*>(destination), static_cast<std::streamsize>(bytes));
        if (!input_) fail("cannot read " + label);
        position_ += bytes;
    }

    std::ifstream input_;
    std::uint64_t file_bytes_ = 0;
    std::uint64_t position_ = 0;
    Header header_;
};

struct Dense {
    std::uint32_t input = 0;
    std::uint32_t output = 0;
    Mem weights;
    Mem bias;
};

struct Norm {
    Mem gamma;
    Mem beta;
};

struct Smolgen {
    Dense compress;
    Dense dense1;
    Norm norm1;
    Dense dense2;
    Norm norm2;
};

struct AttentionWeights {
    Dense query;
    Dense key;
    Dense value;
    Dense output;
    Smolgen smolgen;
};

struct Block {
    AttentionWeights attention;
    Dense ffn_in;
    Dense ffn_out;
    Norm norm1;
    Norm norm2;
    float alpha = 0.0F;
};

struct InputWeights {
    Dense preproc;
    Dense embedding;
    Norm embedding_norm;
    Mem mult_gate;
    Mem add_gate;
    Dense ffn_in;
    Dense ffn_out;
    Norm ffn_norm;
};

struct PolicyWeights {
    Dense embedding;
    Dense query;
    Dense key;
    Mem promotion;
};

struct ValueWeights {
    Dense embedding;
    Dense fc1;
    Dense fc2;
};

class Model {
public:
    Model(Runtime& runtime, Reader& reader) : runtime_(runtime), epsilon_(reader.header().epsilon) {
        const float expected_alpha = static_cast<float>(std::pow(2.0 * static_cast<double>(kEncoderLayers), -0.25));
        input_.preproc = read_dense(reader, kPreprocChannels * kTokens, kPreprocPerToken * kTokens,
                "input.preproc");
        input_.embedding = read_dense(reader, kInputChannels + kPreprocPerToken, kEmbedding, "input.embedding");
        input_.embedding_norm = read_norm(reader, kEmbedding, "input.embedding_ln");
        input_.mult_gate = read_gate(reader, "input.mult_gate");
        input_.add_gate = read_gate(reader, "input.add_gate");
        input_.ffn_in = read_dense(reader, kEmbedding, kFfnHidden, "input.ffn.in");
        input_.ffn_out = read_dense(reader, kFfnHidden, kEmbedding, "input.ffn.out");
        input_.ffn_norm = read_norm(reader, kEmbedding, "input.ffn_ln");

        reader.count(kEncoderLayers, "body encoder count");
        blocks_.reserve(kEncoderLayers);
        for (std::size_t index = 0; index < kEncoderLayers; ++index) {
            const std::string prefix = "body.encoder[" + std::to_string(index) + "]";
            reader.count(kAttentionHeads, prefix + ".attention.heads");
            Block block;
            block.attention.query = read_dense(reader, kEmbedding, kEmbedding, prefix + ".attention.query");
            block.attention.key = read_dense(reader, kEmbedding, kEmbedding, prefix + ".attention.key");
            block.attention.value = read_dense(reader, kEmbedding, kEmbedding, prefix + ".attention.value");
            block.attention.output = read_dense(reader, kEmbedding, kEmbedding, prefix + ".attention.out");
            block.attention.smolgen.compress = read_dense(reader, kEmbedding, kSmolgenChannels,
                    prefix + ".smolgen.compress");
            block.attention.smolgen.dense1 = read_dense(reader, kSmolgenChannels * kTokens, kSmolgenHidden,
                    prefix + ".smolgen.dense1");
            block.attention.smolgen.norm1 = read_norm(reader, kSmolgenHidden, prefix + ".smolgen.ln1");
            block.attention.smolgen.dense2 = read_dense(reader, kSmolgenHidden,
                    kAttentionHeads * kSmolgenPerHead, prefix + ".smolgen.dense2");
            block.attention.smolgen.norm2 = read_norm(reader, kAttentionHeads * kSmolgenPerHead,
                    prefix + ".smolgen.ln2");
            block.ffn_in = read_dense(reader, kEmbedding, kFfnHidden, prefix + ".ffn.in");
            block.ffn_out = read_dense(reader, kFfnHidden, kEmbedding, prefix + ".ffn.out");
            block.norm1 = read_norm(reader, kEmbedding, prefix + ".ln1");
            block.norm2 = read_norm(reader, kEmbedding, prefix + ".ln2");
            reader.activation("MISH", prefix + ".activation");
            block.alpha = reader.f32(prefix + ".alpha");
            if (std::fabs(block.alpha - expected_alpha) > 1.0e-5F) {
                fail(prefix + " has unexpected residual alpha");
            }
            // The safe-Rust reference intentionally executes the pinned
            // architecture constant. Accept the exporter rounding above, then
            // use that same value so CPU and OpenCL cannot drift by metadata.
            block.alpha = expected_alpha;
            blocks_.push_back(std::move(block));
        }
        shared_smolgen_ = runtime_.upload(reader.tensor(kSmolgenPerHead * kAttentionMap, "body.smolgen_w"),
                "body.smolgen_w");

        policy_.embedding = read_dense(reader, kEmbedding, kEmbedding, "policy.embedding");
        reader.count(0, "policy encoder count");
        policy_.query = read_dense(reader, kEmbedding, kEmbedding, "policy.query");
        policy_.key = read_dense(reader, kEmbedding, kEmbedding, "policy.key");
        policy_.promotion = runtime_.upload(reader.tensor(4 * kEmbedding, "policy.promotion_weights"),
                "policy.promotion_weights");
        reader.activation("MISH", "policy.activation");

        value_.embedding = read_dense(reader, kEmbedding, 128, "value.embedding");
        value_.fc1 = read_dense(reader, 128 * kTokens, 128, "value.fc1");
        value_.fc2 = read_dense(reader, 128, kWdl, "value.fc2");
        reader.activation("MISH", "value.activation");
        reader.finish();
        allocate_scratch();
    }

    void predict(const std::array<float, kInputFloats>& encoded,
            std::array<float, kInternalPolicy>& policy,
            std::array<float, kWdl>& wdl) {
        runtime_.write(encoded_, encoded.data(), encoded.size());
        run_body();
        run_policy();
        run_value();
        runtime_.read(internal_policy_, policy.data(), policy.size());
        runtime_.read(value_logits_, wdl.data(), wdl.size());
        for (float value : policy) {
            if (!std::isfinite(value)) fail("OpenCL BT4 policy contains a non-finite value");
        }
        float sum = 0.0F;
        for (float value : wdl) {
            if (!std::isfinite(value)) fail("OpenCL BT4 WDL contains a non-finite value");
            sum += value;
        }
        if (!(sum > 0.999F && sum < 1.001F)) fail("OpenCL BT4 WDL is not normalized");
    }

private:
    Dense read_dense(Reader& reader, std::size_t expected_input, std::size_t expected_output,
            const std::string& label) {
        const std::uint32_t input = reader.u32(label + ".input");
        const std::uint32_t output = reader.u32(label + ".output");
        if (input != expected_input || output != expected_output) {
            std::ostringstream message;
            message << label << " is " << input << "x" << output << "; expected "
                    << expected_input << "x" << expected_output;
            fail(message.str());
        }
        Dense dense;
        dense.input = input;
        dense.output = output;
        dense.weights = runtime_.upload(reader.tensor(expected_input * expected_output, label + ".weights"),
                label + ".weights");
        dense.bias = runtime_.upload(reader.tensor(expected_output, label + ".bias"), label + ".bias");
        return dense;
    }

    Norm read_norm(Reader& reader, std::size_t width, const std::string& label) {
        Norm norm;
        norm.gamma = runtime_.upload(reader.tensor(width, label + "_gamma"), label + "_gamma");
        norm.beta = runtime_.upload(reader.tensor(width, label + "_beta"), label + "_beta");
        return norm;
    }

    Mem read_gate(Reader& reader, const std::string& label) {
        const std::vector<float> serialized = reader.tensor(kTokens * kEmbedding, label);
        std::vector<float> transposed(serialized.size());
        for (std::size_t token = 0; token < kTokens; ++token) {
            for (std::size_t feature = 0; feature < kEmbedding; ++feature) {
                transposed[feature * kTokens + token] = serialized[token * kEmbedding + feature];
            }
        }
        return runtime_.upload(transposed, label);
    }

    void allocate_scratch() {
        encoded_ = runtime_.scratch(kInputFloats, "encoded");
        preproc_input_ = runtime_.scratch(kPreprocChannels * kTokens, "preproc_input");
        preproc_output_ = runtime_.scratch(kPreprocPerToken * kTokens, "preproc_output");
        embedding_input_ = runtime_.scratch((kInputChannels + kPreprocPerToken) * kTokens, "embedding_input");
        flow_ = runtime_.scratch(kEmbedding * kTokens, "flow");
        next_ = runtime_.scratch(kEmbedding * kTokens, "next");
        query_ = runtime_.scratch(kEmbedding * kTokens, "query");
        key_ = runtime_.scratch(kEmbedding * kTokens, "key");
        value_activation_ = runtime_.scratch(kEmbedding * kTokens, "attention_value");
        combined_ = runtime_.scratch(kEmbedding * kTokens, "attention_combined");
        ffn_hidden_ = runtime_.scratch(kFfnHidden * kTokens, "ffn_hidden");
        smolgen_compressed_ = runtime_.scratch(kSmolgenChannels * kTokens, "smolgen_compressed");
        smolgen_flat_ = runtime_.scratch(kSmolgenChannels * kTokens, "smolgen_flat");
        smolgen_mid_ = runtime_.scratch(kSmolgenHidden, "smolgen_mid");
        smolgen_generated_ = runtime_.scratch(kAttentionHeads * kSmolgenPerHead, "smolgen_generated");
        smolgen_bias_ = runtime_.scratch(kAttentionHeads * kAttentionMap, "smolgen_bias");
        policy_flow_ = runtime_.scratch(kEmbedding * kTokens, "policy_flow");
        policy_query_ = runtime_.scratch(kEmbedding * kTokens, "policy_query");
        policy_key_ = runtime_.scratch(kEmbedding * kTokens, "policy_key");
        internal_policy_ = runtime_.scratch(kInternalPolicy, "internal_policy");
        value_embedding_ = runtime_.scratch(128 * kTokens, "value_embedding");
        value_flat_ = runtime_.scratch(128 * kTokens, "value_flat");
        value_hidden_ = runtime_.scratch(128, "value_hidden");
        value_logits_ = runtime_.scratch(kWdl, "value_logits");
    }

    static cl_uint narrow(std::size_t value) {
        if (value > std::numeric_limits<cl_uint>::max()) fail("OpenCL dimension overflow");
        return static_cast<cl_uint>(value);
    }

    void dense(const Dense& layer, const Mem& input, std::size_t tokens, const Mem& output) {
        runtime_.set_mem(runtime_.dense, 0, input);
        runtime_.set_mem(runtime_.dense, 1, layer.weights);
        runtime_.set_mem(runtime_.dense, 2, layer.bias);
        runtime_.set_mem(runtime_.dense, 3, output);
        runtime_.set_value(runtime_.dense, 4, layer.input);
        runtime_.set_value(runtime_.dense, 5, layer.output);
        const cl_uint token_count = narrow(tokens);
        runtime_.set_value(runtime_.dense, 6, token_count);
        runtime_.launch(runtime_.dense, static_cast<std::size_t>(layer.output) * tokens);
    }

    void activate(const Mem& values, std::size_t count, cl_int kind) {
        runtime_.set_mem(runtime_.activation, 0, values);
        const cl_uint size = narrow(count);
        runtime_.set_value(runtime_.activation, 1, size);
        runtime_.set_value(runtime_.activation, 2, kind);
        runtime_.launch(runtime_.activation, count);
    }

    void normalize(const Mem& values, const Norm& norm, std::size_t tokens, std::size_t features) {
        runtime_.set_mem(runtime_.layer_norm, 0, values);
        runtime_.set_mem(runtime_.layer_norm, 1, norm.gamma);
        runtime_.set_mem(runtime_.layer_norm, 2, norm.beta);
        const cl_uint token_count = narrow(tokens);
        const cl_uint feature_count = narrow(features);
        runtime_.set_value(runtime_.layer_norm, 3, token_count);
        runtime_.set_value(runtime_.layer_norm, 4, feature_count);
        runtime_.set_value(runtime_.layer_norm, 5, epsilon_);
        runtime_.launch(runtime_.layer_norm, tokens);
    }

    void residual(const Mem& output, const Mem& input, float alpha, std::size_t count) {
        runtime_.set_mem(runtime_.residual, 0, output);
        runtime_.set_mem(runtime_.residual, 1, input);
        runtime_.set_value(runtime_.residual, 2, alpha);
        const cl_uint size = narrow(count);
        runtime_.set_value(runtime_.residual, 3, size);
        runtime_.launch(runtime_.residual, count);
    }

    void transpose(const Mem& input, const Mem& output, std::size_t features, std::size_t tokens) {
        runtime_.set_mem(runtime_.transpose, 0, input);
        runtime_.set_mem(runtime_.transpose, 1, output);
        const cl_uint feature_count = narrow(features);
        const cl_uint token_count = narrow(tokens);
        runtime_.set_value(runtime_.transpose, 2, feature_count);
        runtime_.set_value(runtime_.transpose, 3, token_count);
        runtime_.launch(runtime_.transpose, features * tokens);
    }

    void run_body() {
        runtime_.set_mem(runtime_.prepare_preproc, 0, encoded_);
        runtime_.set_mem(runtime_.prepare_preproc, 1, preproc_input_);
        runtime_.launch(runtime_.prepare_preproc, kPreprocChannels * kTokens);
        dense(input_.preproc, preproc_input_, 1, preproc_output_);

        runtime_.set_mem(runtime_.prepare_embedding, 0, encoded_);
        runtime_.set_mem(runtime_.prepare_embedding, 1, preproc_output_);
        runtime_.set_mem(runtime_.prepare_embedding, 2, embedding_input_);
        runtime_.launch(runtime_.prepare_embedding, (kInputChannels + kPreprocPerToken) * kTokens);
        dense(input_.embedding, embedding_input_, kTokens, flow_);
        activate(flow_, kEmbedding * kTokens, 2);
        normalize(flow_, input_.embedding_norm, kTokens, kEmbedding);

        runtime_.set_mem(runtime_.gate, 0, flow_);
        runtime_.set_mem(runtime_.gate, 1, input_.mult_gate);
        runtime_.set_mem(runtime_.gate, 2, input_.add_gate);
        const cl_uint trunk_count = narrow(kEmbedding * kTokens);
        runtime_.set_value(runtime_.gate, 3, trunk_count);
        runtime_.launch(runtime_.gate, kEmbedding * kTokens);

        dense(input_.ffn_in, flow_, kTokens, ffn_hidden_);
        activate(ffn_hidden_, kFfnHidden * kTokens, 2);
        dense(input_.ffn_out, ffn_hidden_, kTokens, next_);
        const float alpha = static_cast<float>(std::pow(2.0 * static_cast<double>(kEncoderLayers), -0.25));
        residual(next_, flow_, alpha, kEmbedding * kTokens);
        normalize(next_, input_.ffn_norm, kTokens, kEmbedding);
        std::swap(flow_, next_);

        for (const Block& block : blocks_) {
            run_attention(block);
            residual(next_, flow_, block.alpha, kEmbedding * kTokens);
            normalize(next_, block.norm1, kTokens, kEmbedding);
            dense(block.ffn_in, next_, kTokens, ffn_hidden_);
            activate(ffn_hidden_, kFfnHidden * kTokens, 2);
            dense(block.ffn_out, ffn_hidden_, kTokens, flow_);
            residual(flow_, next_, block.alpha, kEmbedding * kTokens);
            normalize(flow_, block.norm2, kTokens, kEmbedding);
        }
    }

    void run_attention(const Block& block) {
        dense(block.attention.query, flow_, kTokens, query_);
        dense(block.attention.key, flow_, kTokens, key_);
        dense(block.attention.value, flow_, kTokens, value_activation_);
        const Smolgen& smolgen = block.attention.smolgen;
        dense(smolgen.compress, flow_, kTokens, smolgen_compressed_);
        transpose(smolgen_compressed_, smolgen_flat_, kSmolgenChannels, kTokens);
        dense(smolgen.dense1, smolgen_flat_, 1, smolgen_mid_);
        activate(smolgen_mid_, kSmolgenHidden, 3);
        normalize(smolgen_mid_, smolgen.norm1, 1, kSmolgenHidden);
        dense(smolgen.dense2, smolgen_mid_, 1, smolgen_generated_);
        activate(smolgen_generated_, kAttentionHeads * kSmolgenPerHead, 3);
        normalize(smolgen_generated_, smolgen.norm2, 1, kAttentionHeads * kSmolgenPerHead);
        runtime_.set_mem(runtime_.smolgen_project, 0, shared_smolgen_);
        runtime_.set_mem(runtime_.smolgen_project, 1, smolgen_generated_);
        runtime_.set_mem(runtime_.smolgen_project, 2, smolgen_bias_);
        runtime_.launch(runtime_.smolgen_project, kAttentionHeads * kAttentionMap);
        runtime_.set_mem(runtime_.attention, 0, query_);
        runtime_.set_mem(runtime_.attention, 1, key_);
        runtime_.set_mem(runtime_.attention, 2, value_activation_);
        runtime_.set_mem(runtime_.attention, 3, smolgen_bias_);
        runtime_.set_mem(runtime_.attention, 4, combined_);
        runtime_.launch(runtime_.attention, kAttentionHeads * kTokens);
        dense(block.attention.output, combined_, kTokens, next_);
    }

    void run_policy() {
        dense(policy_.embedding, flow_, kTokens, policy_flow_);
        activate(policy_flow_, kEmbedding * kTokens, 2);
        dense(policy_.query, policy_flow_, kTokens, policy_query_);
        dense(policy_.key, policy_flow_, kTokens, policy_key_);
        runtime_.zero(internal_policy_, kInternalPolicy);
        runtime_.set_mem(runtime_.policy, 0, policy_query_);
        runtime_.set_mem(runtime_.policy, 1, policy_key_);
        runtime_.set_mem(runtime_.policy, 2, internal_policy_);
        const cl_uint dimension = narrow(kEmbedding);
        runtime_.set_value(runtime_.policy, 3, dimension);
        runtime_.launch(runtime_.policy, kTokens * kTokens);
        runtime_.set_mem(runtime_.promotion, 0, policy_key_);
        runtime_.set_mem(runtime_.promotion, 1, policy_.promotion);
        runtime_.set_mem(runtime_.promotion, 2, internal_policy_);
        runtime_.set_value(runtime_.promotion, 3, dimension);
        runtime_.launch(runtime_.promotion, 192);
    }

    void run_value() {
        dense(value_.embedding, flow_, kTokens, value_embedding_);
        activate(value_embedding_, 128 * kTokens, 2);
        transpose(value_embedding_, value_flat_, 128, kTokens);
        dense(value_.fc1, value_flat_, 1, value_hidden_);
        activate(value_hidden_, 128, 2);
        dense(value_.fc2, value_hidden_, 1, value_logits_);
        runtime_.set_mem(runtime_.softmax, 0, value_logits_);
        runtime_.launch(runtime_.softmax, 1);
    }

    Runtime& runtime_;
    float epsilon_ = 0.0F;
    InputWeights input_;
    std::vector<Block> blocks_;
    Mem shared_smolgen_;
    PolicyWeights policy_;
    ValueWeights value_;
    Mem encoded_;
    Mem preproc_input_;
    Mem preproc_output_;
    Mem embedding_input_;
    Mem flow_;
    Mem next_;
    Mem query_;
    Mem key_;
    Mem value_activation_;
    Mem combined_;
    Mem ffn_hidden_;
    Mem smolgen_compressed_;
    Mem smolgen_flat_;
    Mem smolgen_mid_;
    Mem smolgen_generated_;
    Mem smolgen_bias_;
    Mem policy_flow_;
    Mem policy_query_;
    Mem policy_key_;
    Mem internal_policy_;
    Mem value_embedding_;
    Mem value_flat_;
    Mem value_hidden_;
    Mem value_logits_;
};

struct Options {
    std::string model;
    std::string vendor = "auto";
    std::uint32_t device = 0;
    bool allow_cpu = false;
    bool list = false;
    bool worker = false;
    std::uint32_t protocol = 0;
    bool help = false;
};

std::uint32_t parse_device(const std::string& text) {
    std::size_t consumed = 0;
    unsigned long value = 0;
    try {
        value = std::stoul(text, &consumed, 10);
    } catch (const std::exception&) {
        fail("--device requires an integer from 0 to 15");
    }
    if (consumed != text.size() || value > 15UL) fail("--device requires an integer from 0 to 15");
    return static_cast<std::uint32_t>(value);
}

Options parse_options(int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        auto value = [&](const char* flag) -> std::string {
            if (++index >= argc) fail(std::string(flag) + " requires a value");
            return argv[index];
        };
        if (argument == "--model") options.model = value("--model");
        else if (argument == "--vendor") options.vendor = lower(value("--vendor"));
        else if (argument == "--device") options.device = parse_device(value("--device"));
        else if (argument == "--protocol") options.protocol = parse_device(value("--protocol"));
        else if (argument == "--worker") options.worker = true;
        else if (argument == "--allow-cpu") options.allow_cpu = true;
        else if (argument == "--list") options.list = true;
        else if (argument == "--help" || argument == "-h") options.help = true;
        else fail("unknown argument: " + argument);
    }
    if (options.vendor != "auto" && options.vendor != "nvidia"
            && options.vendor != "amd" && options.vendor != "intel") {
        fail("--vendor must be auto, nvidia, amd, or intel");
    }
    return options;
}

void print_usage() {
    std::cout
        << "Usage: janus-bt4-opencl --worker --protocol 1 --model PATH"
           " [--vendor auto|nvidia|amd|intel] [--device 0..15]\n"
        << "       janus-bt4-opencl --list [--vendor auto|nvidia|amd|intel] [--allow-cpu]\n"
        << "\n--allow-cpu is test-only; protocol mode otherwise accepts GPU devices only.\n";
}

void put_u16(std::array<unsigned char, kHeaderBytes>& bytes, std::size_t offset, std::uint16_t value) {
    bytes[offset] = static_cast<unsigned char>(value & 0xffU);
    bytes[offset + 1] = static_cast<unsigned char>((value >> 8U) & 0xffU);
}

void put_u32(std::array<unsigned char, kHeaderBytes>& bytes, std::size_t offset, std::uint32_t value) {
    for (std::size_t byte = 0; byte < 4; ++byte) {
        bytes[offset + byte] = static_cast<unsigned char>((value >> (8U * byte)) & 0xffU);
    }
}

void put_u64(std::array<unsigned char, kHeaderBytes>& bytes, std::size_t offset, std::uint64_t value) {
    for (std::size_t byte = 0; byte < 8; ++byte) {
        bytes[offset + byte] = static_cast<unsigned char>((value >> (8U * byte)) & 0xffU);
    }
}

std::uint16_t get_u16(const std::array<unsigned char, kHeaderBytes>& bytes, std::size_t offset) {
    return static_cast<std::uint16_t>(bytes[offset])
            | static_cast<std::uint16_t>(static_cast<std::uint16_t>(bytes[offset + 1]) << 8U);
}

std::uint32_t get_u32(const std::array<unsigned char, kHeaderBytes>& bytes, std::size_t offset) {
    return static_cast<std::uint32_t>(bytes[offset])
            | (static_cast<std::uint32_t>(bytes[offset + 1]) << 8U)
            | (static_cast<std::uint32_t>(bytes[offset + 2]) << 16U)
            | (static_cast<std::uint32_t>(bytes[offset + 3]) << 24U);
}

std::uint64_t get_u64(const std::array<unsigned char, kHeaderBytes>& bytes, std::size_t offset) {
    std::uint64_t value = 0;
    for (std::size_t byte = 0; byte < 8; ++byte) {
        value |= static_cast<std::uint64_t>(bytes[offset + byte]) << (8U * byte);
    }
    return value;
}

void write_all(const void* data, std::size_t bytes) {
    const unsigned char* cursor = static_cast<const unsigned char*>(data);
    while (bytes != 0) {
        const std::size_t written = std::fwrite(cursor, 1, bytes, stdout);
        if (written == 0) fail("cannot write worker protocol response");
        cursor += written;
        bytes -= written;
    }
}

bool read_header(std::array<unsigned char, kHeaderBytes>& header) {
    std::size_t offset = 0;
    while (offset != header.size()) {
        const std::size_t got =
                std::fread(header.data() + offset, 1, header.size() - offset, stdin);
        if (got == 0) {
            if (offset == 0 && std::feof(stdin) != 0) return false;
            fail("truncated worker request header");
        }
        offset += got;
    }
    return true;
}

void read_all(void* data, std::size_t bytes) {
    unsigned char* cursor = static_cast<unsigned char*>(data);
    while (bytes != 0) {
        const std::size_t got = std::fread(cursor, 1, bytes, stdin);
        if (got == 0) fail("truncated worker request payload");
        cursor += got;
        bytes -= got;
    }
}

void append_u32(std::vector<unsigned char>& bytes, std::uint32_t value) {
    bytes.push_back(static_cast<unsigned char>(value & 0xffU));
    bytes.push_back(static_cast<unsigned char>((value >> 8U) & 0xffU));
    bytes.push_back(static_cast<unsigned char>((value >> 16U) & 0xffU));
    bytes.push_back(static_cast<unsigned char>((value >> 24U) & 0xffU));
}

std::vector<unsigned char> hello_payload(const Candidate& selected, std::uint32_t device_index) {
    if (selected.name.size() > 256) fail("OpenCL device name is too long");
    const std::uint32_t vendor = vendor_code(selected);
    if (vendor == 0) fail("OpenCL device vendor is unsupported");
    std::vector<unsigned char> payload;
    payload.reserve(24 + selected.name.size());
    append_u32(payload, vendor);
    append_u32(payload, device_index);
    append_u32(payload, static_cast<std::uint32_t>(kInputFloats));
    append_u32(payload, static_cast<std::uint32_t>(kInternalPolicy));
    append_u32(payload, static_cast<std::uint32_t>(kWdl));
    append_u32(payload, static_cast<std::uint32_t>(selected.name.size()));
    payload.insert(payload.end(), selected.name.begin(), selected.name.end());
    return payload;
}

void send_response(std::uint16_t opcode, std::uint32_t status, std::uint64_t request_id,
        const std::vector<unsigned char>& payload) {
    if (payload.size() > kMaxFramePayload) fail("worker response exceeds the protocol payload bound");
    std::array<unsigned char, kHeaderBytes> header{};
    std::memcpy(header.data(), "JGPU", 4);
    put_u16(header, 4, static_cast<std::uint16_t>(kProtocolVersion));
    put_u16(header, 6, static_cast<std::uint16_t>(opcode | 0x8000U));
    put_u64(header, 8, request_id);
    put_u32(header, 16, static_cast<std::uint32_t>(payload.size()));
    put_u32(header, 20, status);
    write_all(header.data(), header.size());
    if (!payload.empty()) write_all(payload.data(), payload.size());
    if (std::fflush(stdout) != 0) fail("cannot flush worker response");
}

std::vector<unsigned char> diagnostic_payload(const std::string& message) {
    const std::size_t count = std::min(message.size(), kMaxFramePayload);
    return std::vector<unsigned char>(message.begin(), message.begin() + static_cast<std::ptrdiff_t>(count));
}

void send_error(std::uint16_t opcode, std::uint32_t status, std::uint64_t request_id,
        const std::string& message) {
    send_response(opcode, status, request_id, diagnostic_payload(message));
}

std::array<float, kInputFloats> decode_input(const std::vector<unsigned char>& payload) {
    std::array<float, kInputFloats> values{};
    for (std::size_t index = 0; index < values.size(); ++index) {
        const std::size_t base = index * 4;
        const std::uint32_t bits = static_cast<std::uint32_t>(payload[base])
                | (static_cast<std::uint32_t>(payload[base + 1]) << 8U)
                | (static_cast<std::uint32_t>(payload[base + 2]) << 16U)
                | (static_cast<std::uint32_t>(payload[base + 3]) << 24U);
        std::memcpy(&values[index], &bits, sizeof(bits));
        if (!std::isfinite(values[index])) fail("predict request contains a non-finite input");
    }
    return values;
}

std::vector<unsigned char> encode_prediction(const std::array<float, kInternalPolicy>& policy,
        const std::array<float, kWdl>& wdl) {
    std::vector<unsigned char> payload(kPredictResponseBytes);
    auto encode = [&](std::size_t index, float value) {
        std::uint32_t bits = 0;
        std::memcpy(&bits, &value, sizeof(bits));
        const std::size_t base = index * 4;
        payload[base] = static_cast<unsigned char>(bits & 0xffU);
        payload[base + 1] = static_cast<unsigned char>((bits >> 8U) & 0xffU);
        payload[base + 2] = static_cast<unsigned char>((bits >> 16U) & 0xffU);
        payload[base + 3] = static_cast<unsigned char>((bits >> 24U) & 0xffU);
    };
    for (std::size_t index = 0; index < policy.size(); ++index) encode(index, policy[index]);
    for (std::size_t index = 0; index < wdl.size(); ++index) encode(policy.size() + index, wdl[index]);
    return payload;
}

int serve(Model& model, const Candidate& selected, std::uint32_t device_index) {
    std::array<float, kInputFloats> zero_input{};
    std::array<float, kInternalPolicy> warm_policy{};
    std::array<float, kWdl> warm_wdl{};
    const auto start = std::chrono::steady_clock::now();
    model.predict(zero_input, warm_policy, warm_wdl);
    const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - start).count();
    std::cerr << "BT4 OpenCL warmup passed on " << selected.vendor << " " << selected.name
              << " in " << elapsed << " ms\n";

    std::array<unsigned char, kHeaderBytes> header{};
    while (read_header(header)) {
        if (std::memcmp(header.data(), "JGPU", 4) != 0) fail("invalid worker request magic");
        const std::uint16_t version = get_u16(header, 4);
        const std::uint16_t opcode = get_u16(header, 6);
        const std::uint64_t request_id = get_u64(header, 8);
        const std::uint32_t payload_bytes = get_u32(header, 16);
        const std::uint32_t request_status = get_u32(header, 20);
        if (version != kProtocolVersion) {
            send_error(opcode, kStatusProtocol, request_id, "unsupported worker protocol version");
            fail("unsupported worker protocol version");
        }
        if (request_status != 0) {
            send_error(opcode, kStatusProtocol, request_id, "worker request status must be zero");
            fail("worker request status must be zero");
        }
        if (payload_bytes > kMaxFramePayload) {
            send_error(opcode, kStatusProtocol, request_id,
                    "worker request exceeds the fixed payload bound");
            fail("worker request exceeds the fixed payload bound");
        }
        std::vector<unsigned char> payload(payload_bytes);
        if (!payload.empty()) read_all(payload.data(), payload.size());
        if (opcode == kOpcodeHello || opcode == kOpcodeShutdown) {
            if (payload_bytes != 0) {
                send_error(opcode, kStatusProtocol, request_id,
                        "hello/shutdown requests must not carry a payload");
                fail("hello/shutdown requests must not carry a payload");
            }
            if (opcode == kOpcodeHello) {
                send_response(opcode, kStatusOk, request_id, hello_payload(selected, device_index));
            } else {
                send_response(opcode, kStatusOk, request_id, {});
                return 0;
            }
            continue;
        }
        if (opcode != kOpcodePredict || payload_bytes != kPredictRequestBytes) {
            send_error(opcode, kStatusProtocol, request_id,
                    "invalid predict opcode or payload size");
            fail("invalid predict opcode or payload size");
        }
        std::array<float, kInputFloats> input{};
        try {
            input = decode_input(payload);
        } catch (const std::exception& error) {
            std::cerr << "input rejected: " << error.what() << '\n';
            send_error(opcode, kStatusInput, request_id, error.what());
            continue;
        }
        std::array<float, kInternalPolicy> policy{};
        std::array<float, kWdl> wdl{};
        try {
            model.predict(input, policy, wdl);
            send_response(opcode, kStatusOk, request_id, encode_prediction(policy, wdl));
        } catch (const std::exception& error) {
            std::cerr << "OpenCL inference failed: " << error.what() << '\n';
            send_error(opcode, kStatusBackend, request_id, error.what());
            return 5;
        }
    }
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    try {
#if defined(_WIN32)
        _setmode(_fileno(stdin), _O_BINARY);
        _setmode(_fileno(stdout), _O_BINARY);
#endif
        const Options options = parse_options(argc, argv);
        if (options.help) {
            print_usage();
            return 0;
        }
        const std::vector<Candidate> candidates = enumerate_devices(options.vendor, options.allow_cpu);
        if (options.list) {
            for (std::size_t index = 0; index < candidates.size(); ++index) {
                const Candidate& candidate = candidates[index];
                const char* kind = (candidate.type & CL_DEVICE_TYPE_GPU) != 0 ? "gpu" : "cpu-test-only";
                std::cout << index << '\t' << kind << '\t' << candidate.vendor << '\t' << candidate.name
                          << '\t' << (candidate.global_memory / (1024ULL * 1024ULL)) << " MiB\n";
            }
            return 0;
        }
        if (!options.worker) fail("protocol mode requires --worker");
        if (options.protocol != kProtocolVersion) fail("protocol mode requires --protocol 1");
        if (options.model.empty()) fail("--model PATH is required");
        if (options.device >= candidates.size()) {
            std::ostringstream message;
            message << "OpenCL device index " << options.device << " is unavailable; "
                    << candidates.size() << " matching device(s) found";
            fail(message.str());
        }
        const Candidate selected = candidates[options.device];
        if (!options.allow_cpu && (selected.type & CL_DEVICE_TYPE_GPU) == 0) {
            fail("CPU OpenCL devices require the test-only --allow-cpu option");
        }
        Reader reader(options.model);
        std::cerr << "loading " << reader.header().name << " on " << selected.vendor << " " << selected.name << '\n';
        Runtime runtime(selected);
        Model model(runtime, reader);
        return serve(model, selected, options.device);
    } catch (const std::exception& error) {
        std::cerr << "janus-bt4-opencl: " << error.what() << '\n';
        return 2;
    }
}
