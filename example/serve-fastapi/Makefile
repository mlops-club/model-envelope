.PHONY: build-and-run redeploy clean

install:
	bash ./run.sh install

run:
	bash ./run.sh run

# Default target - build and run latest model
build-and-run:
	bash ./run.sh build_and_run 3

pull-model:
	bash ./run.sh pull_model 3

build:
	bash ./run.sh build 3

# Force rebuild and redeploy of latest model
run-docker:
	./run.sh run 3

# Clean up containers, images, and downloaded models
clean:
	./run.sh clean
