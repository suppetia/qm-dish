CC := $(shell which gcc)
FC := $(shell which gfortran)
PY := $(shell which python3)


SRC_DIR := src
BUILD_DIR := dist

PACKAGE_NAME := qm-dish

# check for the fortran compiler
MISSING_COMPILATION_DEPENDENCIES :=
ifeq ($(FC),)
	MISSING_COMPILATION_DEPENDENCIES += gfortran
endif
# check if numpy is installed
ifeq (,$(shell $(PY) -c "import numpy.f2py; print(numpy.f2py.__version__.version)" 2>/dev/null))
	MISSING_COMPILATION_DEPENDENCIES += numpy
endif



build: build_dir
ifdef MISSING_COMPILATION_DEPENDENCIES
	@echo "'"$(MISSING_COMPILATION_DEPENDENCIES)"' is/are not installed. Install the dependencies to compile adams.f90 and significantly speed up the program"
else
	cd $(BUILD_DIR)/dish/util/numeric; \
	$(PY) -m numpy.f2py -c adams_f.f90 -m adams_f -llapack --f90exec=$(FC) --f77exec=$(FC)
endif
	cp LICENSE MANIFEST.in pyproject.toml README.md $(BUILD_DIR)
	$(PY) -m build $(BUILD_DIR) --outdir $(BUILD_DIR)

source_dir:
ifeq ($(wildcard $(SRC_DIR)),)
	$(error "sources not present. '$(SRC_DIR) missing'")
endif

build_dir: source_dir
	mkdir -p $(BUILD_DIR)
	cp -R $(SRC_DIR)/ $(BUILD_DIR)/
	rm -rf $$(find $(BUILD_DIR) -type d -name __pycache__)


install:
ifeq ($(wildcard $(BUILD_DIR)/$(PACKAGE_NAME)-*.*),)
	$(error "package wasn't build. Make sure to run 'make [build]' first")
endif
	$(PY) -m pip install $(PACKAGE_NAME) --find-links $(BUILD_DIR)/

uninstall:
	$(PY) -m pip uninstall $(PACKAGE_NAME) -y


# remove all build stuff
.PHONY: clean
clean:
	rm -rf $(BUILD_DIR)
	rm -rf $(SRC_DIR)/*.egg-info
	@echo "removed all build dependencies"
