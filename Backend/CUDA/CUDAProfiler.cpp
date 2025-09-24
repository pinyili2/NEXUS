#include "CUDAProfiler.h"
#include <pthread.h>

#if defined(USE_CUDA) && defined(USE_NVTX)

namespace ARBD {
namespace CUDA {

// Static member initialization
thread_local size_t Profiler::color_index_ = 0;

constexpr std::array<uint32_t, 12> Profiler::color_palette_;

void Profiler::push_range(std::string_view message) {
	push_range(message, get_next_color());
}

void Profiler::push_range(std::string_view message, Color color) {
	push_range(message, static_cast<uint32_t>(color));
}

void Profiler::push_range(std::string_view message, uint32_t color_value) {
	nvtxEventAttributes_t attributes = create_attributes(message, color_value);
	nvtxRangePushEx(&attributes);
}

void Profiler::pop_range() {
	nvtxRangePop();
}

void Profiler::mark(std::string_view message) {
	mark(message, get_next_color());
}

void Profiler::mark(std::string_view message, Color color) {
	mark(message, static_cast<uint32_t>(color));
}

void Profiler::mark(std::string_view message, uint32_t color_value) {
	nvtxEventAttributes_t attributes = create_attributes(message, color_value);
	nvtxMarkEx(&attributes);
}

void Profiler::name_thread(std::string_view name) {
	// Convert string_view to null-terminated string for NVTX
	std::string name_str(name);
	nvtxNameOsThreadA(pthread_self(), name_str.c_str());
}

void Profiler::set_category(std::string_view category) {
	// NVTX doesn't have a direct category concept, but we can use it in range names
	// This is more of a utility function for user code organization
	static thread_local std::string current_category;
	current_category = category;
}

uint32_t Profiler::get_next_color() {
	uint32_t color = color_palette_[color_index_ % color_palette_.size()];
	color_index_++;
	return color;
}

nvtxEventAttributes_t Profiler::create_attributes(std::string_view message, uint32_t color_value) {
	nvtxEventAttributes_t attributes = {};
	attributes.version = NVTX_VERSION;
	attributes.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;

	// Set color
	attributes.colorType = NVTX_COLOR_ARGB;
	attributes.color = color_value;

	// Set message
	attributes.messageType = NVTX_MESSAGE_TYPE_ASCII;

	// NVTX expects null-terminated strings, but string_view might not be null-terminated
	// We need to create a persistent string for the lifetime of the range
	// For performance, we'll use a thread-local buffer for temporary storage
	static thread_local std::string message_buffer;
	message_buffer.assign(message.data(), message.size());
	attributes.message.ascii = message_buffer.c_str();

	return attributes;
}

} // namespace CUDA
} // namespace ARBD

#endif // USE_CUDA && USE_NVTX
