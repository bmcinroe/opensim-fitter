#!/bin/bash
# Exit when an error happens instead of continue.
set -e

# Default values for flags.
DEBUG_TYPE="Release"
NUM_JOBS=${OPENSIM_BUILD_JOBS:-24}
MOCO="on"
ORG="nickbianco"
BRANCH="3ecb9f235fa347fa3ce2285d857594fac9b81cf9"

# Separate generators for dependencies and core.
DEPS_GENERATOR="Unix Makefiles"
CORE_GENERATOR="Ninja"

PYTHON_ROOT_DIR=$1
WORKING_DIR="$(pwd)/opensim"

# Preflight checks.
missing=""
command -v make      >/dev/null 2>&1 || missing="$missing make"
command -v ninja     >/dev/null 2>&1 || missing="$missing ninja-build"
command -v pkg-config >/dev/null 2>&1 || missing="$missing pkg-config"
command -v gfortran  >/dev/null 2>&1 || missing="$missing gfortran"
if [ -n "$missing" ]; then
    echo "ERROR: missing required build tools:$missing" >&2
    echo "On Debian/Ubuntu: sudo apt install$missing" >&2
    exit 1
fi

if [ -z "$PYTHON_ROOT_DIR" ]; then
    echo "ERROR: pass the Python executable as the first argument." >&2
    exit 1
fi

if [ -d "$WORKING_DIR" ]; then
    rm -rf "$WORKING_DIR"
fi
mkdir -p "$WORKING_DIR"

# Get opensim-core.
git clone https://github.com/$ORG/opensim-core.git "$WORKING_DIR/opensim-core"
cd "$WORKING_DIR/opensim-core"
git checkout $BRANCH

# Build opensim-core dependencies.
mkdir -p "$WORKING_DIR/opensim-core/dependencies/build"
cd "$WORKING_DIR/opensim-core/dependencies/build"
cmake "$WORKING_DIR/opensim-core/dependencies" \
    -G "$DEPS_GENERATOR" \
    -DCMAKE_BUILD_TYPE=$DEBUG_TYPE \
    -DCMAKE_INSTALL_PREFIX="$WORKING_DIR/opensim_dependencies_install/" \
    -DSUPERBUILD_ezc3d=off \
    -DOPENSIM_WITH_CASADI=$MOCO \
    -DBUILD_PYTHON_WRAPPING=on \
    -DPython3_EXECUTABLE="$PYTHON_ROOT_DIR"

# Confirm the Moco dependency chain actually is switched on.
echo "--- dependency superbuild configuration ---"
cmake . -LA 2>/dev/null | grep -Ei "casadi|ipopt|SUPERBUILD" || true
echo "-------------------------------------------"

cmake --build . --config $DEBUG_TYPE -j$NUM_JOBS

# Build and install opensim-core.
export PKG_CONFIG_PATH="$WORKING_DIR/opensim_dependencies_install/ipopt/lib/pkgconfig:$PKG_CONFIG_PATH"

mkdir -p "$WORKING_DIR/opensim-core/build"
cd "$WORKING_DIR/opensim-core/build"
cmake "$WORKING_DIR/opensim-core" \
    -G "$CORE_GENERATOR" \
    -DCMAKE_BUILD_TYPE=$DEBUG_TYPE \
    -DOPENSIM_DEPENDENCIES_DIR="$WORKING_DIR/opensim_dependencies_install/" \
    -DOPENSIM_C3D_PARSER=None \
    -DBUILD_TESTING=off \
    -DCMAKE_INSTALL_PREFIX="$WORKING_DIR/opensim_core_install" \
    -DOPENSIM_INSTALL_UNIX_FHS=off \
    -DOPENSIM_WITH_CASADI=$MOCO \
    -DBUILD_PYTHON_WRAPPING=on \
    -DPython3_EXECUTABLE="$PYTHON_ROOT_DIR"

cmake --build . --config $DEBUG_TYPE -j$NUM_JOBS
cmake --install .

# Verify CasADi is compiled in.
if [ "$MOCO" = "on" ]; then
    echo "--- verifying MocoCasADiSolver availability ---"
    if ldd "$WORKING_DIR/opensim_core_install/sdk/lib/libosimMoco.so" 2>/dev/null \
        | grep -qi casadi; then
        echo "OK: libosimMoco.so links libcasadi"
    else
        echo "WARNING: libosimMoco.so does not link libcasadi." >&2
        echo "Moco will throw 'MocoCasADiSolver is not available' at runtime." >&2
        exit 1
    fi
fi
