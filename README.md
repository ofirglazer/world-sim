# world-sim

## How to run
1. In a terminal tab run: uvicorn sim3dves.web.server:app --host 0.0.0.0 --port 8000 --reload . Watch server start running in the terminal.
3. In a browser tab go to http://localhost:8000/ . Verify the rendering the client.
5. Press "New Scenario". Verify connecting to the server.

## TODOs
* data panels for selected entities should update and not remain static.
* rounding in serializer to reflect 5-10 cm and not 2.

## check
* after pressing new scenario for a second time, the view resets automatically after zoom or pan, if opening a new browser tab it behaves OK
* with pytest.ini 8 tests fail, without pytest.ini only 1 test fail
* test_uav_low_fuel_shown_in_panel fails
* test_uav_specific_fields_in_panel fails
* mismatch between wedge visualization and actual detection
* what is the meaning: For the initial implementation, hardcode the scenario configuration to match run_simulation.py's defaults — the body can be empty {}
* what is the meaning: Use --workers 1. The simulation worker is a thread, not a process — multiple uvicorn workers would each create their own SessionRegistry with no shared state, so sessions created on one worker would not be visible to another. A single worker is correct until Redis pub/sub is introduced.
* what is the meaning: No Redis service yet

## add to PRD
* real map including raster and vector
* ...