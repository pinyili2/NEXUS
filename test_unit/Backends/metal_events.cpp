#ifdef USE_METAL

// @TODO: Confirm std::move is correct for metal event

#include "../catch_boiler.h"
#include "Backend/Events.h"
#include "Backend/METAL/METALManager.h"

using namespace ARBD;

TEST_CASE("Metal Event Basic Operations", "[events]") {
	// Setup: Initialize Metal manager with device selection
	METAL::Manager::load_info();
	auto& device = const_cast<METAL::Manager::Device&>(METAL::Manager::get_current_device());
	auto& queue = device.get_next_queue();
	Resource metal_resource(ResourceType::METAL, 0);

	SECTION("Default Event Construction") {
		Event default_event;
		REQUIRE_FALSE(default_event.is_valid());
		REQUIRE(default_event.is_complete()); // Default events are considered complete
	}

	SECTION("Null Event Construction") {
		Event null_event(nullptr, metal_resource);
		REQUIRE_FALSE(null_event.is_valid());
		REQUIRE(null_event.is_complete());
		REQUIRE(null_event.get_resource().type == ResourceType::METAL);
	}

	SECTION("Metal Event Construction and Operations") {
		// Create a command buffer
		void* cmd_buffer_ptr = queue.create_command_buffer();
		REQUIRE(cmd_buffer_ptr != nullptr);

		// Create Metal event with command buffer
		METAL::Event metal_event(cmd_buffer_ptr);

		// Commit the command buffer before creating the backend event
		metal_event.commit();

		// Create backend Event with the Metal event
		Event backend_event(std::move(metal_event), metal_resource); //not sure if I should explicitly move the metal event

		REQUIRE(backend_event.is_valid());
		REQUIRE(backend_event.get_resource().type == ResourceType::METAL);
		REQUIRE(backend_event.get_resource().id == 0);

		// Test wait functionality - now safe since command buffer is committed
		backend_event.wait();
		REQUIRE(backend_event.is_complete());
	}

	SECTION("Event Implementation Access") {
		void* cmd_buffer_ptr = queue.create_command_buffer();
		METAL::Event metal_event(cmd_buffer_ptr);
		Event backend_event(std::move(metal_event), metal_resource);

		// Verify we can access the implementation
		REQUIRE(backend_event.is_valid());
		REQUIRE(backend_event.get_resource().type == ResourceType::METAL);

		// Don't wait on this one - just test construction and validity
	}
}

TEST_CASE("Metal EventList Operations", "[events]") {
	// Setup
	METAL::Manager::load_info();
	auto& device = const_cast<METAL::Manager::Device&>(METAL::Manager::get_current_device());
	auto& queue = device.get_next_queue();
	Resource metal_resource(ResourceType::METAL, 0);

	SECTION("Default EventList Construction") {
		EventList list;
		REQUIRE(list.empty());
		REQUIRE(list.all_complete());
	}

	SECTION("EventList with Multiple Events") {
		EventList list;

		// Add multiple events with valid command buffers
		for (int i = 0; i < 3; ++i) {
			void* cmd_buffer_ptr = queue.create_command_buffer();
			METAL::Event metal_event(cmd_buffer_ptr);

			// Commit each command buffer
			metal_event.commit();
			Event backend_event(std::move(metal_event), metal_resource);
			list.add(backend_event);
		}

		REQUIRE_FALSE(list.empty());
		REQUIRE(list.get_events().size() == 3);

		// Test waiting on all events - now safe since all are committed
		list.wait_all();
		REQUIRE(list.all_complete());
	}

	SECTION("EventList Management") {
		EventList list;

		// Add an event with valid command buffer (no commit needed for this test)
		void* cmd_buffer_ptr = queue.create_command_buffer();
		METAL::Event metal_event(cmd_buffer_ptr);
		Event backend_event(std::move(metal_event), metal_resource);
		list.add(backend_event);

		REQUIRE_FALSE(list.empty());

		// Clear the list
		list.clear();
		REQUIRE(list.empty());
	}

	SECTION("Metal Event Extraction") {
		EventList list;

		// Add a Metal event with valid command buffer (no commit needed for this test)
		void* cmd_buffer_ptr = queue.create_command_buffer();
		METAL::Event metal_event(cmd_buffer_ptr);
		Event backend_event(std::move(metal_event), metal_resource);
		list.add(backend_event);

		// Extract Metal events
		auto metal_events = list.get_metal_events();
		REQUIRE_FALSE(metal_events.empty());
		REQUIRE(metal_events.size() == 1);
	}

	// Cleanup
	METAL::Manager::finalize();
}

#endif // USE_METAL
