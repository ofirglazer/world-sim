# world-sim

## TODOs
* rounding in serializer to reflect 5-10 cm and not 2.

## check
* after pressing new scenario for a second time, the view resets automatically after zoom or pan, if opening a new browser tab it behaves OK
* in world view the road network is not visible
* in world view the waypoints are not visible
* in c4i view all entities are visible, not only UAV and detections
* with pytest.ini 8 tests fail, without pytest.ini only 1 test fail
* in test_m3 the NFZ violated test fails. Also, it seems that the UAV never checks if it is violating NFZ with the nfz_violated method.
* test_uav_low_fuel_shown_in_panel fails
* test_uav_specific_fields_in_panel fails
* mismatch between wedge visualization and actual detection
* what is the meaning: For the initial implementation, hardcode the scenario configuration to match run_simulation.py's defaults — the body can be empty {}
* what is the meaning: Use --workers 1. The simulation worker is a thread, not a process — multiple uvicorn workers would each create their own SessionRegistry with no shared state, so sessions created on one worker would not be visible to another. A single worker is correct until Redis pub/sub is introduced.
* what is the meaning: No Redis service yet

## add to PRD
* real map including raster and vector
* ...