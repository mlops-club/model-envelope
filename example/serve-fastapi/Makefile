.PHONY: build-and-run redeploy clean

# Default target - build and run latest model
build-and-run:
	bash ./run.sh build_and_run 3

pull-model:
	bash ./run.sh pull_model 3

build:
	bash ./run.sh build 3

# Force rebuild and redeploy of latest model
run:
	./run.sh run 3

# Clean up containers, images, and downloaded models
clean:
	./run.sh clean
