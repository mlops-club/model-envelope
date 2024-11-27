.PHONY: build-and-run redeploy clean

# Default target - build and run latest model
build-and-run:
	bash ./run.sh build_and_run 1

# Force rebuild and redeploy of latest model
redeploy:
	./run.sh clean
	./run.sh build_and_run 1

# Clean up containers, images, and downloaded models
clean:
	./run.sh clean
