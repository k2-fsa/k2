.PHONY: all clean

all: build/bin/torch_api_test
	./build/bin/torch_api_test

build/bin/torch_api_test: torch_api_test.cc
	mkdir -p build
	cd build && \
			cmake .. && \
			make VERBOSE=1 -j2 && \
			./bin/torch_api_test

clean:
	$(RM) -r build
